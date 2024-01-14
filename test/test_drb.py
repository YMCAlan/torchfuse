import pytest
import torch
from torchfuse.unireplknet import DRB


def test_drb():
    model = DRB(3, 52, k=17).cuda().eval()
    input = torch.randn((1, 3, 512, 512)).cuda()

    output = model(input)
    print()
    print(output.size())

    model.fuse_convs()
    model.forward = model.forward_fuse

    fused_output = model(input)
    print(output.size())
    error = (output - fused_output).abs().mean().item()
    print(f"\nError (): {error}")