IMAGENET_C_CORRUPTION_GROUPS: dict[str, list[str]] = {
    "blur": [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
    ],
    "digital": [
        "contrast",
        "elastic_transform",
        "jpeg_compression",
        "pixelate",
    ],
    "noise": [
        "gaussian_noise",
        "impulse_noise",
        "shot_noise",
    ],
    "weather": [
        "brightness",
        "fog",
        "frost",
        "snow",
    ],
    "extra": [
        "gaussian_blur",
        "saturate",
        "spatter",
        "speckle_noise",
    ],
}

IMAGENET_C_SEVERITIES = [1, 2, 3, 4, 5]
