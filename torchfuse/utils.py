import thop
import torch

from copy import deepcopy


def get_num_params(model):
    """Return the total number of parameters in a model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops_with_torch_profile(model, imgsz=224):
    p = next(model.parameters())
    im = torch.zeros((1, p.shape[1], imgsz, imgsz), device=p.device)
    flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9
    return flops


def model_info(model, imgsz):
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    flops = get_flops_with_torch_profile(model, imgsz)
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    print(f"{model.__class__.__name__} summary: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}")
