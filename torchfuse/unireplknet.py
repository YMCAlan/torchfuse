from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils.torch_utils import fuse_conv_and_bn
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

settings_dict = {
    # kernel dilates/ stride in de_conv
    17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
    15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
    13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
    11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
    9: ([5, 5, 3, 3], [1, 2, 3, 4]),
    7: ([5, 3, 3], [1, 2, 3]),
    5: ([3, 3], [1, 2])
}


def _fuse_bn(conv, bn):
    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    return torch.mm(w_bn, w_conv).view(conv.weight.shape), torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DRB(nn.Module):
    """
       Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
       We assume the inputs to this block are (N, C, H, W)
       """

    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, k, g=math.gcd(c1, c2))
        self.kernel_sizes, self.dilates = settings_dict[k][0], settings_dict[k][1]

        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__setattr__(
                f'cv_k{k}_r{r}', Conv(c1, c2, k=k, s=1, p=(r * (k - 1) + 1) // 2, g=math.gcd(c1, c2), d=r)
            )

    def forward(self, x):
        out = self.cv1(x)
        for k, r in zip(self.kernel_sizes, self.dilates):
            cv_k_r = self.__getattr__(f'cv_k{k}_r{r}')
            out = out + cv_k_r(x)
        return out

    def forward_fuse(self, x):
        return self.cv1(x)

    def fuse_convs(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cv1 = nn.Conv2d(
            in_channels=self.cv1.conv.in_channels,
            out_channels=self.cv1.conv.out_channels,
            kernel_size=self.cv1.conv.kernel_size,
            stride=self.cv1.conv.stride,
            padding=self.cv1.conv.padding,
            dilation=self.cv1.conv.dilation,
            groups=self.cv1.conv.groups,
            bias=True,
        ).requires_grad_(False)

        self.cv1.weight.data = kernel
        self.cv1.bias.data = bias

    def get_equivalent_kernel_bias(self):
        cv1_w, cv1_b = _fuse_bn(self.cv1.conv, self.cv1.bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            cv_k_r = self.__getattr__(f'cv_k{k}_r{r}')
            cv_k_r_w, cv_k_r_b = _fuse_bn(cv_k_r.conv, cv_k_r.bn)
            cv1_w = merge_dilated_into_large_kernel(cv1_w, cv_k_r_w, r)
            cv1_b += cv_k_r_b
            self.__delattr__(f'cv_k{k}_r{r}')
        return cv1_w, cv1_b
