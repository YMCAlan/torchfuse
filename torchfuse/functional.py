import torch
import torch.nn as nn

from typing import Union, Tuple, Any

__all__ = ["fuse_conv_and_bn"]


def fuse_nd_conv_and_nd_bn(conv: nn.Module, bn: nn.Module, is_transposed: bool, ndim: int) -> nn.Module:
    """
    Fuse n-dimensional transposed convolution / convolution and batch normalization layers.

    :param conv: Convolution layer
    :param bn: Batch normalization layer
    :param is_transposed: Is the transposed convolution
    :param ndim: Number of dim
    :return: Fused convolution layer.
    """
    fused_conv = getattr(
        nn, f"ConvTranspose{ndim}d" if is_transposed else f"Conv{ndim}d")(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv


def fuse_conv_and_bn(
        conv: nn.Module,
        bn: nn.Module
) -> nn.Module:
    """
    Fuse convolution and batch normalization layers.
    :param conv: Convolution layer
    :param bn: Batch normalization layer
    :return: Fused convolution layer.
    """
    if not isinstance(
            conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        raise ValueError(
            f"Error: Expected 'conv' to be an instance of nn.ConvNd or nn.ConvTransposeNd, but got {type(conv).__name__}."
        )
    if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        raise ValueError(f"Error: Expected 'bn' to be an instance of nn.BatchNormNd, but got {type(bn).__name__}.")

    name = getattr(type(conv), "__name__")
    conv_dim = getattr(type(conv), "__name__")[-2]
    bn_dim = getattr(type(bn), "__name__")[-2]
    if conv_dim is None or bn_dim is None or conv_dim != bn_dim:
        raise ValueError(f"Error: Convolution and Batch Normalization types do not match. "
                         f"Convolution type: {type(conv).__name__}, BN type: {type(bn).__name__}.")

    is_transposed = True if "Transpose" in name else False
    fused_conv = fuse_nd_conv_and_nd_bn(conv, bn, is_transposed, int(conv_dim))
    return fused_conv
