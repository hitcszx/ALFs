# Asymmetric Loss Functions for Learning with Noisy Labels

This reposity is the official implementation of [Asymmetric Loss Functions for Learning with Noisy Labels](https://arxiv.org/abs/2106.03110).

## Requirements
```console
Python >= 3.6, PyTorch >= 1.3.1, torchvision >= 0.4.1, numpy>=1.11.2, tqdm >= 4.50.2, seaborn >= 0.11.0
```

## Training
The main running file is [`main.py`](./main.py) with arguments as follows:
* noise_type: symmetric | asymmetric
* noise_rate: noise rate
* loss: AGCE | AUL | AEL | CE (Cross Entropy) | FL (Focal Loss) | MAE | GCE | SCE | NFL | NCE | ...

The detailed implementation about the loss can be found in [`losses.py`](./losses.py)

Example for 0.4 Symmetric noise rate with AUL loss
```console
# CIFAR-10
$  python3  main.py --noise_type      symmetric           \
                    --noise_rate      0.4                 \
                    --loss            AUL                 \
```



## Reference
For technical details and full experimental results, please check the paper. If you have used our work in your own, please consider citing:

```
@inproceedings{zhou2021asymmetric,
  title={asymmetric Loss Functions for Learning with Noisy Labels},
  author={Zhou, Xiong and Liu, Xianming and Jiang, Junjun and Xin, Gao and Xiangyang, Ji},
  booktitle={ICML},
  year={2021}
}
```
Moreover, we thank the code implemented by [Ma et al.](https://github.com/HanxunH/Active-Passive-Losses).
