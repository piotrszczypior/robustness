#!/usr/bin/env bash

set -euo pipefail

DATA="imagenet_a"
WORKERS=12
SYNC="--sync-drive"
CMD="python src/main.py evaluate --run-batch $DATA --num-workers $WORKERS $SYNC"

echo "======================================================="
echo " Evaluation — $(date)"
echo "======================================================="

$CMD --model alexnet              --batch-size 512
$CMD --model resnet18             --batch-size 512
$CMD --model mobilenet_v3_large   --batch-size 512
$CMD --model densenet121          --batch-size 256
$CMD --model efficientnet_b0      --batch-size 256

$CMD --model resnet50             --batch-size 256
$CMD --model resnet152            --batch-size 128
$CMD --model wide_resnet50_2      --batch-size 128
$CMD --model wide_resnet101_2     --batch-size 128
$CMD --model resnext101_64x4d     --batch-size 128
$CMD --model regnet_y_16gf        --batch-size 128
$CMD --model efficientnet_b4      --batch-size 128
$CMD --model efficientnet_v2_m    --batch-size 128

$CMD --model convnext_base        --batch-size 128
$CMD --model convnext_large       --batch-size 128
$CMD --model swin_b               --batch-size 128
$CMD --model swin_v2_b            --batch-size 128
$CMD --model maxvit_t             --batch-size 64
$CMD --model vit_b_16             --batch-size 256
$CMD --model vit_l_16             --batch-size 128
$CMD --model vit_h_14             --batch-size 32

echo "======================================================="
echo " DONE — $(date)"
echo "======================================================="