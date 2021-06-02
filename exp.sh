#!/bin/bash
# symmetric noise
python exp.py --dataset $1 --noise_rate 0.0 --noise_type symmetric --gpus 0 &
python exp.py --dataset $1 --noise_rate 0.2 --noise_type symmetric --gpus 0 &
python exp.py --dataset $1 --noise_rate 0.4 --noise_type symmetric --gpus 0 &
python exp.py --dataset $1 --noise_rate 0.6 --noise_type symmetric --gpus 0 &
python exp.py --dataset $1 --noise_rate 0.8 --noise_type symmetric --gpus 1 &
# asymmetric noise
python exp.py --dataset $1 --noise_rate 0.1 --noise_type asymmetric --gpus 1 &
python exp.py --dataset $1 --noise_rate 0.2 --noise_type asymmetric --gpus 1 &
python exp.py --dataset $1 --noise_rate 0.3 --noise_type asymmetric --gpus 1 &
python exp.py --dataset $1 --noise_rate 0.4 --noise_type asymmetric --gpus 1 &
wait