import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import numbers
from layers import *

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



# SACU
class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()

        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True),
            eca_layer(self.channels)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1, groups=self.channels * 2, bias=True),
            eca_layer(self.channels * 2)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            eca_layer(self.channels)
        )

    def forward(self, pre, cur):
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k, v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        out = self.conv_out(out) + cur
        return out


class BasicBlock(torch.nn.Module):
    def __init__(self, dim=32):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.lb_max = nn.Parameter(torch.Tensor([0.6]))  # 0.6
        self.lb_min = nn.Parameter(torch.Tensor([0.4]))  # 0.4

        self.atten = Atten(dim)
        self.atten2 = Atten(dim)
        self.thres = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

        self.shallow_feat1 = Pre(dim)
        self.shallow_feat2 = Post(dim)

    def forward(self, x, pre, pre2, PhiWeight, PhiTb, i, x_record, g_record):
        if i != 0:
            b, c, w, h = x.shape
            g = g_record[i]
            x_pre = x_record[i - 1]
            g_pre = g_record[i - 1]
            x = x.view(b, -1)
            x_pre = x_pre.view(b, -1)
            g = g.view(b, -1)
            g_pre = g_pre.view(b, -1)

            delta_x = x - x_pre
            delta_g = g - g_pre
            res_min = float('inf')
            delta_x_min = delta_x[0]
            for t in range(b):
                delta_x_t = torch.unsqueeze(delta_x[t], 1)
                delta_g_t = torch.unsqueeze(delta_g[t], 1)
                result = torch.mm(torch.transpose(delta_x_t, 0, 1), delta_g_t).item()
                if result < res_min:
                    res_min = result
                    delta_x_min = delta_x_t
            if res_min > 0:
                res = torch.min(torch.mm(torch.transpose(delta_x_min, 0, 1), delta_x_min) / res_min)
                self.lambda_step.data = torch.min(self.lb_max, torch.max(self.lb_min, res))
            else:
                self.lambda_step.data = self.lb_max
            x = x.view(b, c, w, h)

        t = PhiTPhi_fun(x, PhiWeight)
        x = x - self.lambda_step * t
        x_g = x + self.lambda_step * PhiTb
        g = t - PhiTb
        x_input = self.shallow_feat1(x_g)

        x_mid = self.atten(pre, x_input)
        x_input = x_mid + x_g

        x_input = torch.mul(torch.sign(x_input), F.relu(torch.abs(x_input) - self.thres))

        pre2 = torch.cat((x_g, pre2), dim=1)
        x = self.atten2(x_input, pre2)
        x = self.shallow_feat2(x)
        x_pred = x + x_input

        return x_pred, x_mid, g


class Net(torch.nn.Module):
    def __init__(self, LayerNo, sensing_rate):
        super(Net, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.patch_size = 32
        self.n_input = int(sensing_rate * 1024)

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.Phiweight = nn.Parameter(
            init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, 32, 3, padding=1, bias=True)
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1, bias=True)

    def forward(self, x):
        PhiTb = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None)
        PhiTb = F.conv_transpose2d(PhiTb, self.Phiweight, stride=self.patch_size)

        x = PhiTb
        z_pre = self.fe(x)
        z_pre2 = self.fe2(x)

        x_record = []
        g_record = []
        x_record.append(x)
        g_record.append(x)

        for i in range(self.LayerNo):
            x_dual, x_mid, g = self.fcs[i](x, z_pre, z_pre2, self.Phiweight, PhiTb, i, x_record, g_record)
            x = x_dual[:, :1, :, :]
            z_pre = x_mid
            z_pre2 = x_dual[:, 1:, :, :]
            x_record.append(x)
            g_record.append(g)

        x_final = x
        return x_final
