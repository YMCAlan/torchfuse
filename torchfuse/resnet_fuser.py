import torch
from torchfuse.base_fuser import BasedFuser
from torchfuse.fuse_resnet_functional import fuse_resnet, fuse_bottle_neck, fuse_basic_block
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchfuse.utils import model_info
from copy import deepcopy


class ResNetFuser(BasedFuser):
    def __init__(self, model, imgsz):
        super().__init__(model, imgsz)

    def __call__(self, *args, **kwargs):
        model_info(self.model, self.imgsz)
        self.fused()
        model_info(self.fused_model, self.imgsz)

        return self.fused_model

    def fused(self):
        model = deepcopy(self.model)
        for m in model.modules():
            if isinstance(m, Bottleneck):
                fuse_bottle_neck(m)
            elif isinstance(m, BasicBlock):
                fuse_basic_block(m)
            elif isinstance(m, ResNet):
                fuse_resnet(m)
            else:
                pass
        self.fused_model = model
