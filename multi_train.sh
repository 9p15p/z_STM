#!/bin/sh
python train_pre_STM.py
python train_main_STM.py --loadepoch pre_STM_latest