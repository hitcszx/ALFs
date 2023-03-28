# Asymmetric Loss Functions for Learning with Noisy Labels

This repository is the official implementation of [Asymmetric Loss Functions for Learning with Noisy Labels](https://arxiv.org/abs/2106.03110) [ICML 2021] and [Asymmetric Loss Functions for Noise-tolerant Learning: Theory and Applications](https://ieeexplore.ieee.org/document/10039708) [T-PAMI].

## Requirements
```console
Python >= 3.6, PyTorch >= 1.3.1, torchvision >= 0.4.1, numpy>=1.11.2, tqdm >= 4.50.2, seaborn >= 0.11.0, tensorboardX >= 2.5
```

## Learning with Noisy Labels (LNL)
The main running file is [`main.py`](./lnl/main.py) with arguments as follows:
* noise_type: symmetric | asymmetric
* noise_rate: noise rate
* loss: AGCE | AUL | AEL | CE (Cross Entropy) | FL (Focal Loss) | MAE | GCE | SCE | NFL | NCE | ...

The detailed implementation about the proposed asymmetric losses for classification can be found in [`./lnl/losses.py`](./lnl/losses.py)

Example for 0.4 Symmetric noise rate with AUL loss
```console
# CIFAR-10
$  python3  main.py --noise_type      symmetric           \
                    --noise_rate      0.4                 \
                    --loss            AUL                 \
```

## Self-supervised Image Denoising
The main running file is [`main.py`](./denoising/main.py) with arguments as follows:
* exp: n2c | n2n | n2s
* style: gauss | bernoulli | saltpepper | impulse
* loss: heat | poisson | lp | mse ...

The detailed implementation about the proposed asymmetric losses for regression can be found in [`./denoising/losses.py`](./denoising/losses.py)

Example for using the negative heat kernel loss for Gaussian denoising with noise2self
```console
$  python3  main.py --exp       n2s               \
                    --style     gauss15           \
                    --loss      heat0.1           \
```


## Reference
For technical details and full experimental results, please check the paper. If you have used our work in your own, please consider citing:

```bibtex
@InProceedings{zhou2021asymmetric,
  title = 	 {Asymmetric Loss Functions for Learning with Noisy Labels},
  author =       {Zhou, Xiong and Liu, Xianming and Jiang, Junjun and Gao, Xin and Ji, Xiangyang},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12846--12856},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR}
}

@ARTICLE{10039708,
  author={Zhou, Xiong and Liu, Xianming and Zhai, Deming and Jiang, Junjun and Ji, Xiangyang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Asymmetric Loss Functions for Noise-Tolerant Learning: Theory and Applications}, 
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TPAMI.2023.3236459}
}

```
Moreover, we thank the code implemented by [Ma et al.](https://github.com/HanxunH/Active-Passive-Losses) (classification) and [Zhang et al.](https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch) (image denoising).
