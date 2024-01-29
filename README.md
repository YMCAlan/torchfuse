# Torchfuse
The torchfuser provides a fuse functional, reducing both model parameters and FLOPs.

# Usage
```Python
import torch
from torchvision.models.resnet import resnet50
from torchfuse.resnet_fuser import ResNetFuser

if __name__ == '__main__':
    model = resnet50().eval()
    fuser = ResNetFuser(model, 224)
    fused_model = fuser()
    x = torch.randn(1, 3, 224, 224)

    output = model(x)
    fused_output = fused_model(x)

    error = (fused_output - output).abs().mean().item()
    print("Error = ", error)
    print("Output Predict :", output.argmax().item())
    print("Fused Output Predict :", fused_output.argmax().item())
```
# Reference
RepVGG: Making VGG-style ConvNets Great Again '
[RepVGG](https://arxiv.org/pdf/2101.03697.pdf)
