#!/usr/bin/env bash
set -euo pipefail

python src/main.py evaluate --model alexnet --run-batch imagenet_clean --num-workers 8 --batch-size 512
python src/main.py evaluate --model resnet18 --run-batch imagenet_clean --num-workers 8 --batch-size 512
python src/main.py evaluate --model efficientnet_b0 --run-batch imagenet_clean --num-workers 8 --batch-size 256
python src/main.py evaluate --model mobilenet_v3_large --run-batch imagenet_clean --num-workers 8 --batch-size 256

python src/main.py evaluate --model resnet50 --run-batch imagenet_clean --num-workers 8 --batch-size 128
python src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_clean --num-workers 8 --batch-size 128
python src/main.py evaluate --model wide_resnet50_2 --run-batch imagenet_clean --num-workers 8 --batch-size 128
python src/main.py evaluate --model vit_b_16 --run-batch imagenet_clean --num-workers 8 --batch-size 64

python src/main.py evaluate --model resnext101_64x4d --run-batch imagenet_clean --num-workers 8 --batch-size 32
python src/main.py evaluate --model wide_resnet101_2 --run-batch imagenet_clean --num-workers 8 --batch-size 32
python src/main.py evaluate --model swin_v2_b --run-batch imagenet_clean --num-workers 8 --batch-size 32
python src/main.py evaluate --model convnext_base --run-batch imagenet_clean --num-workers 8 --batch-size 32

python src/main.py evaluate --model vit_l_16 --run-batch imagenet_clean --num-workers 8 --batch-size 16
python src/main.py evaluate --model convnext_large --run-batch imagenet_clean --num-workers 8 --batch-size 16
python src/main.py evaluate --model vit_h_14 --run-batch imagenet_clean --num-workers 8 --batch-size 4