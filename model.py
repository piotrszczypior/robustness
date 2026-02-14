import torchvision.models as models


def _build(model_fn, weights):
    return model_fn(weights=weights)


def get_resnet50():
    return _build(
        models.resnet50,
        models.ResNet50_Weights.IMAGENET1K_V1,
    )


def get_vit_b_16():
    return _build(
        models.vit_b_16,
        models.ViT_B_16_Weights.IMAGENET1K_V1,
    )
