import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfuse.unireplknet import DRepConv, convert_dilated_to_nondilated


def test_equivalency():
    in_channels = 1
    out_channels = 1
    groups = 1
    large_kernel_size = 5
    small_conv_r = 2
    small_conv_k = 3

    equivalent_kernel_size = small_conv_r * (small_conv_k - 1) + 1
    large_conv = nn.Conv2d(in_channels, out_channels, kernel_size=large_kernel_size,
                           padding=large_kernel_size // 2, groups=groups, bias=False)
    dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=small_conv_k,
                             padding=equivalent_kernel_size // 2,
                             dilation=small_conv_r, groups=groups, bias=False)
    H, W = 19, 19
    x = torch.rand(2, in_channels, H, W)
    origin_y = large_conv(x) + dilated_conv(x)
    equivalent_kernel = convert_dilated_to_nondilated(dilated_conv.weight.data, small_conv_r)
    rows_to_pad = large_kernel_size // 2 - equivalent_kernel_size // 2
    merged_kernel = large_conv.weight.data + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    equivalent_y = F.conv2d(x,
                            merged_kernel,
                            bias=None,
                            padding=large_kernel_size // 2,
                            groups=groups)

    print("\n ======== kernel vis ============ \n")
    print(dilated_conv.weight.data.squeeze())
    print(equivalent_kernel.squeeze())

    print("\n ======== Error ============ \n")
    relative_error = (equivalent_y - origin_y).abs().sum() / origin_y.abs().sum()
    print('Relative error:', relative_error.item())
    absolute_error = (equivalent_y - origin_y).abs().mean()
    print('Mean Absolute error:', absolute_error.item())


def test_drb():
    model = DRepConv(3, 52, k=17).cuda().eval()
    input = torch.randn((1, 3, 512, 512)).cuda()

    origin_output = model(input)
    model.fuse_convs()
    model.forward = model.forward_fuse
    equivalent_output = model(input)

    print("\n ======== Error ============ \n")
    relative_error = (equivalent_output - origin_output).abs().sum() / origin_output.abs().sum()
    print('Relative error:', relative_error.item())
    absolute_error = (equivalent_output - origin_output).abs().mean()
    print('Mean Absolute error:', absolute_error.item())
