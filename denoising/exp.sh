#!/bin/bash

python3 gray_main.py --exp n2c --style  gauss20 --loss heat0.2 --gpu 0 &
python3 gray_main.py --exp n2c --style  bernoulli0.3 --loss heat0.2 --gpu 1 &
python3 gray_main.py --exp n2c --style  saltpepper0.3 --loss heat0.2 --gpu 2 &
python3 gray_main.py --exp n2c --style  impulse0.3 --loss heat0.2 --gpu 3 &

python3 gray_main.py --exp n2c --style  gauss20 --loss lp2 --gpu 4 &
python3 gray_main.py --exp n2c --style  bernoulli0.3 --loss lp2 --gpu 5 &
python3 gray_main.py --exp n2c --style  saltpepper0.3 --loss lp2 --gpu 6 &
python3 gray_main.py --exp n2c --style  impulse0.3 --loss lp2 --gpu 7 &
wait