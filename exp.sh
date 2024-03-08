#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python3 train_lt_mxp_bacc_bs.py --save_path asan/adamixup/bit_resnext50_1_f1_mp0.2_natural_bs128_bal_soft --model_name bit_resnext50_1 --sampling natural --im_size '100,100' --n_epochs 100 --metric mcc --n_classes 5 --do_mixup 0.2 --batch_size 128 | tee bit_resnext50_1_mixup_mp0.2_natural_bs128_bal_soft

CUDA_VISIBLE_DEVICES=1 python3 train_lt_mxp_bacc_bs.py --save_path asan/adamixup/bit_resnext50_1_f1_mp0.6_natural_bs128_bal_soft --model_name bit_resnext50_1 --sampling natural --im_size '100,100' --n_epochs 100 --metric mcc --n_classes 5 --do_mixup 0.6 --batch_size 128 | tee bit_resnext50_1_mixup_mp0.6_natural_bs128_bal_soft

CUDA_VISIBLE_DEVICES=1 python3 train_lt_mxp_bacc_bs.py --save_path asan/adamixup/bit_resnext50_1_f1_mp0.5_natural_bs128_bal_soft --model_name bit_resnext50_1 --sampling natural --im_size '100,100' --n_epochs 100 --metric mcc --n_classes 5 --do_mixup 0.5 --batch_size 128 | tee bit_resnext50_1_mixup_mp0.5_natural_bs128_bal_soft

CUDA_VISIBLE_DEVICES=1 python3 train_lt_mxp_bacc_bs.py --save_path asan/adamixup/bit_resnext50_1_f1_mp0.8_natural_bs128_bal_soft --model_name bit_resnext50_1 --sampling natural --im_size '100,100' --n_epochs 100 --metric mcc --n_classes 5 --do_mixup 0.8 --batch_size 128 | tee bit_resnext50_1_mixup_mp0.8_natural_bs128_bal_soft

CUDA_VISIBLE_DEVICES=1 python3 train_lt_mxp_bacc_bs.py --save_path asan/adamixup/bit_resnext50_1_f1_mp0.9_natural_bs128_bal_soft --model_name bit_resnext50_1 --sampling natural --im_size '100,100' --n_epochs 100 --metric mcc --n_classes 5 --do_mixup 0.9 --batch_size 128 | tee bit_resnext50_1_mixup_mp0.9_natural_bs128_bal_soft

CUDA_VISIBLE_DEVICES=1 python3 train_lt_mxp_bacc_bs.py --save_path asan/adamixup/bit_resnext50_1_f1_mp1_natural_bs128_bal_soft --model_name bit_resnext50_1 --sampling natural --im_size '100,100' --n_epochs 100 --metric mcc --n_classes 5 --do_mixup 1 --batch_size 128 | tee bit_resnext50_1_mixup_mp1_natural_bs128_bal_soft
