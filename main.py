import torch
from torchvision.models.resnet import resnet50, resnet18
from torchfuse.resnet_fuser import ResNetFuser

if __name__ == '__main__':
    model = resnet18().eval()
    fuser = ResNetFuser(model, 224)
    fused_model = fuser()
    x = torch.randn(1, 3, 224, 224)

    output = model(x)
    fused_output = fused_model(x)

    for m in fused_model.modules():
        print(m)

    error = (fused_output - output).abs().mean().item()
    print("Error = ", error)
    print("Output Predict :", output.argmax().item())
    print("Fused Output Predict :", fused_output.argmax().item())
