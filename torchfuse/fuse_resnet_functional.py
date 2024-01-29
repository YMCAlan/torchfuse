import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchfuse.functional import fuse_conv_and_bn


def fuse_downsample(downsample):
    fused_conv = fuse_conv_and_bn(downsample[0], downsample[1])
    downsample = nn.Sequential(fused_conv)
    return downsample

def fuse_basic_block(m):
    fused_conv1 = fuse_conv_and_bn(m.conv1, m.bn1)
    fused_conv2 = fuse_conv_and_bn(m.conv2, m.bn2)

    m.__delattr__("bn1")
    m.__delattr__("bn2")
    m.__setattr__("conv1", fused_conv1)
    m.__setattr__("conv2", fused_conv2)

    if hasattr(m, 'downsample'):
        fuse_downsample(m.downsample)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None and m.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    m.forward = forward.__get__(m, BasicBlock)


def fuse_bottle_neck(m):
    fused_conv1 = fuse_conv_and_bn(m.conv1, m.bn1)
    fused_conv2 = fuse_conv_and_bn(m.conv2, m.bn2)
    fused_conv3 = fuse_conv_and_bn(m.conv3, m.bn3)

    m.__delattr__("bn1")
    m.__delattr__("bn2")
    m.__delattr__("bn3")

    m.__setattr__("conv1", fused_conv1)
    m.__setattr__("conv2", fused_conv2)
    m.__setattr__("conv3", fused_conv3)

    if hasattr(m, 'downsample') and m.downsample is not None:
        m.downsample = fuse_downsample(m.downsample)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    m.forward = forward.__get__(m, Bottleneck)


def fuse_resnet(m):
    fused_conv1 = fuse_conv_and_bn(m.conv1, m.bn1)
    m.__delattr__("bn1")
    m.__setattr__("conv1", fused_conv1)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    m._forward_impl = _forward_impl.__get__(m, ResNet)

    return m
