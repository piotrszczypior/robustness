import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel, LayerGradCam
import types
from torchcam.methods import GradCAMpp

MODEL_FAMILY: dict[str, str] = {
    "alexnet": "cnn",
    "resnet18": "cnn",
    "resnet50": "cnn",
    "resnet152": "cnn",
    "regnet_y_16gf": "cnn",
    "resnext101_64x4d": "cnn",
    "wide_resnet50_2": "cnn",
    "wide_resnet101_2": "cnn",
    "efficientnet_b0": "cnn",
    "efficientnet_b4": "cnn",
    "efficientnet_v2_m": "cnn",
    "densenet121": "cnn",
    "mobilenet_v3_large": "cnn",
    "vit_b_16": "vit",
    "vit_l_16": "vit",
    "vit_h_14": "vit",
    "swin_b": "swin",
    "swin_v2_b": "swin",
    "maxvit_t": "maxvit",
    "convnext_base": "hybrid",
    "convnext_large": "hybrid",
}


def _normalize(arr: np.ndarray) -> np.ndarray:
    min_, max_ = arr.min(), arr.max()
    if max_ > min_:
        return (arr - min_) / (max_ - min_)
    return arr


def _to_numpy_heatmap(attributions: torch.Tensor) -> np.ndarray:
    attr = attributions.squeeze().detach().cpu()
    if attr.dim() == 3:
        attr = attr.abs().mean(dim=0)
    return _normalize(attr.numpy())


def _run_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    gcpp = LayerGradCam(model, target_layer)
    attrs = gcpp.attribute(input_tensor, target=class_idx)
    attrs = F.interpolate(
        attrs,
        size=input_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return _to_numpy_heatmap(attrs)


def _run_gradcam_pp(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    cam_extractor = GradCAMpp(model, target_layer)
    out = model(input_tensor)
    cams = cam_extractor(class_idx, out)
    cam_extractor.remove_hooks()
    cam = cams[0]
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    else:
        cam = cam.unsqueeze(1)
    resized_cam = F.interpolate(
        cam,
        size=input_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return _normalize(resized_cam.squeeze().detach().cpu().numpy())


def _run_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    steps: int = 50,
) -> np.ndarray:
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)
    attrs = ig.attribute(input_tensor, baseline, target=class_idx, n_steps=steps, internal_batch_size=1)

    return _to_numpy_heatmap(attrs)


def _run_layer_ig(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    steps: int = 50,
) -> np.ndarray:
    """IG attributed to the patch embedding output (ViT-specific)."""
    layer = model.conv_proj
    lig = LayerIntegratedGradients(model, layer)
    baseline = torch.zeros_like(input_tensor)
    attrs = lig.attribute(input_tensor, baseline, target=class_idx, n_steps=steps, internal_batch_size=1)

    # attrs: [1, C, H_patch, W_patch] → collapse channels → resize to input
    attr = attrs.squeeze(0).abs().mean(dim=0, keepdim=True).unsqueeze(0)
    attr = F.interpolate(attr, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return _normalize(attr.squeeze().detach().cpu().numpy())


def _run_smoothgrad_ig(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    steps: int = 25,
    n_samples: int = 10,
    noise_level: float = 0.15,
) -> np.ndarray:
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    baseline = torch.zeros_like(input_tensor)
    attrs = nt.attribute(
        input_tensor,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        nt_samples_batch_size=1,
        stdevs=noise_level,
        baselines=baseline,
        target=class_idx,
        n_steps=steps,
        internal_batch_size=1,
    )

    return _to_numpy_heatmap(attrs)


def _run_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    attention_maps = []
    original_forwards = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            original_forwards[module] = module.forward

            def patched_forward(self, query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False

                out, weights = original_forwards[self](query, key, value, **kwargs)
                attention_maps.append(weights.detach())
                return out, weights

            module.forward = types.MethodType(patched_forward, module)

    with torch.no_grad():
        _ = model(input_tensor)

    for module, orig_fwd in original_forwards.items():
        module.forward = orig_fwd

    if not attention_maps:
        h = w = input_tensor.shape[-1] // 16
        return np.ones((h, w))

    result = torch.eye(attention_maps[0].shape[-1]).to(input_tensor.device)

    for attn in attention_maps:
        if attn.dim() == 4:
            attn_avg = attn.mean(dim=1)
        else:
            attn_avg = attn

        attn_avg = attn_avg + torch.eye(attn_avg.shape[-1]).to(attn_avg.device)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = attn_avg @ result

    seq_len = result.shape[-1]
    grid_size = int((seq_len - 1) ** 0.5)
    mask = result[0, 0, 1:].reshape(grid_size, grid_size)

    return _normalize(mask.cpu().numpy())


def get_all_explanations(
    model: nn.Module,
    model_name: str,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    layer_ig: bool = False,
) -> dict[str, np.ndarray]:
    family = MODEL_FAMILY.get(model_name.lower(), "cnn")

    explanations = {}

    if layer_ig and family == "vit":
        explanations["integrated_gradients"] = _run_layer_ig(model, input_tensor, class_idx)
    else:
        explanations["integrated_gradients"] = _run_integrated_gradients(
            model, input_tensor, class_idx
        )
    explanations["smoothgrad_ig"] = _run_smoothgrad_ig(model, input_tensor, class_idx)

    if family in ("cnn", "hybrid", "swin", "maxvit"):
        explanations["gradcam_pp"] = _run_gradcam_pp(
            model, target_layer, input_tensor, class_idx
        )
    elif family == "vit":
        explanations["attention_rollout"] = _run_attention_rollout(model, input_tensor)

    return explanations
