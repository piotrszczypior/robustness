#!/usr/bin/env bash
set -euo pipefail

# ./run.sh src/main.py setup --dataset imagenet_c  --archives digital.tar
# ./run.sh src/main.py evaluate --model resnet18 --run-batch imagenet_c_digital --num-workers 8 --batch-size 1024

# # ./run.sh src/main.py setup --dataset imagenet_c  --archives noise.tar
# ./run.sh src/main.py evaluate --model resnet18 --run-batch imagenet_c_noise --num-workers 8 --batch-size 1024

# # ./run.sh src/main.py setup --dataset imagenet_c  --archives weather.tar
# ./run.sh src/main.py evaluate --model resnet18 --run-batch imagenet_c_weather --num-workers 8 --batch-size 1024

# ./run.sh src/main.py evaluate --model resnet50 --run-batch imagenet_c_digital --num-workers 8 --batch-size 512
# ./run.sh src/main.py evaluate --model resnet50 --run-batch imagenet_c_noise --num-workers 8 --batch-size 512

# ./run.sh src/main.py evaluate --model resnet50 --run-batch imagenet_c_weather --num-workers 8 --batch-size 512

# ./run.sh src/main.py evaluate --model resnet152 --run-batch imagenet_c_weather --num-workers 8 --batch-size 512

# # regnet_y_16gf

# ./run.sh src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_c_digital --num-workers 8 --batch-size 256

# ./run.sh src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_c_noise --num-workers 8 --batch-size 256

# ./run.sh src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_c_weather --num-workers 8 --batch-size 256

# # wide_resnet50_2

# ./run.sh src/main.py evaluate --model wide_resnet50_2 --run-batch imagenet_c_weather --num-workers 8 --batch-size 256


# # regnet_y_16gf
# ./run.sh src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_c_digital --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_c_noise --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model regnet_y_16gf --run-batch imagenet_c_weather --num-workers 8 --batch-size 128

# # densenet121
# ./run.sh src/main.py evaluate --model densenet121 --run-batch imagenet_c_weather --num-workers 8 --batch-size 256

# # wide_resnet101_2
# ./run.sh src/main.py evaluate --model wide_resnet101_2 --run-batch imagenet_c_digital --num-workers 8 --batch-size 256
# ./run.sh src/main.py evaluate --model wide_resnet101_2 --run-batch imagenet_c_noise --num-workers 8 --batch-size 256

# # mobilenet_v3_large
# ./run.sh src/main.py evaluate --model mobilenet_v3_large --run-batch imagenet_c_weather --num-workers 8 --batch-size 512

# # wide_resnet101_2
# ./run.sh src/main.py evaluate --model wide_resnet101_2 --run-batch imagenet_c_weather --num-workers 8 --batch-size 256

# # resnext101_64x4d
# ./run.sh src/main.py evaluate --model resnext101_64x4d --run-batch imagenet_c_digital --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model resnext101_64x4d --run-batch imagenet_c_blur --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model resnext101_64x4d --run-batch imagenet_c_noise --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model resnext101_64x4d --run-batch imagenet_c_weather --num-workers 8 --batch-size 128

# # efficientnet_b4

# # efficientnet_v2_m
# ./run.sh src/main.py evaluate --model efficientnet_v2_m --run-batch imagenet_c_digital --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model efficientnet_v2_m --run-batch imagenet_c_noise --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model efficientnet_v2_m --run-batch imagenet_c_weather --num-workers 8 --batch-size 128


# ---
# # convnext_base
# ./run.sh src/main.py evaluate --model convnext_base --run-batch imagenet_c_weather --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model efficientnet_b4 --run-batch imagenet_c_weather --num-workers 8 --batch-size 64

# # efficientnet_b0
# ./run.sh src/main.py evaluate --model efficientnet_b0 --run-batch imagenet_c_weather --num-workers 8 --batch-size 256
# ./run.sh src/main.py evaluate --model efficientnet_b0 --run-batch imagenet_c_noise --num-workers 8 --batch-size 256
# ./run.sh src/main.py evaluate --model efficientnet_b0 --run-batch imagenet_c_digital --num-workers 8 --batch-size 256

# # convnext_large
# ./run.sh src/main.py evaluate --model convnext_large --run-batch imagenet_c_weather --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model convnext_large --run-batch imagenet_c_noise --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model convnext_large --run-batch imagenet_c_digital --num-workers 8 --batch-size 128
# ./run.sh src/main.py evaluate --model convnext_large --run-batch imagenet_c_blur --num-workers 8 --batch-size 128

./run.sh src/main.py evaluate --model maxvit_t --run-batch imagenet_c_noise --batch-size 128 --num-workers 8

./run.sh src/main.py evaluate --model swin_b --run-batch imagenet_c_noise --batch-size 128 --num-workers 8

./run.sh src/main.py evaluate --model swin_v2_b --run-batch imagenet_c_noise --batch-size 128 --num-workers 8