## Spectrally Adaptive Channel-aware Unrolling Network for Compressed Sensing
[📄 Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37987)
Deep Unrolling Networks (DUNs) integrate classical optimization recovery problems in Compressed Sensing (CS) with sophisticated deep learning network architectures, leading to substantial breakthroughs. However, prevailing DUNs generally face challenges concerning solidified gradient descent step size strategies, inadequate feature extraction within the iterative stage and limited information interaction between iterative stages. To overcome these obstacles, we propose SCU-Net, a channel-focused unrolling network inspired by the renowned spectral projected gradient optimization algorithm. In particular, we tailore two pivotal components, Barzilai-Borwein-gradient Descent Optimizer (BBDO) and Channel-guided Cross-attention Reconstruction Module (CCRM), to collaboratively undertake the reconstruction task. BBDO leverages a gradient calculation strategy based on BB step size to enhance data fidelity optimization, while CCRM addresses the intricate mapping issue associated with sparse induction, encompassing customized functionalities from Adaptive Channel Interaction Layer (ACIL) and Spatially Augmented Channel-aware Unit (SACU). Among them, ACIL amalgamates convolution operations and channel attention mechanisms to achieve meticulous information screening alongside efficient feature enhancement. SACU introduces dual reinforcement variables to bolster information exchange across different iterative stages, coupled with the optimization of cross-attention to facilitate the modeling of long-distance dependencies. Extensive experiments in both image CS and magnetic resonance imaging exhibit that our SCU-Net manifests superior performance, surpassing state-of-the-art methods.

## Pretrained models
https://drive.google.com/drive/folders/1dFjJN5Pyg8-eWITIU5Hj0p0oWGohOaCR?usp=drive_link

## Citation

If you find this project useful in your research or work, please consider citing:

@inproceedings{wang2026spectrally,
  title={Spectrally Adaptive Channel-aware Unrolling Network for Compressed Sensing},
  author={Wang, Xiaoyang and Gan, Hongping},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={12},
  pages={10190--10198},
  year={2026}
}
