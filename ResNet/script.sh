#!/bin/bash

python3 train.py --train --save --test --epoch 50 > logs/perso_resnet.log

python3 train.py --train --save --test --epoch 50 --torch_resnet > logs/torch_resnet.log