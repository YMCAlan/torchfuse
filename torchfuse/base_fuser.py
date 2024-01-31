from copy import deepcopy


class BasedFuser(object):
    def __init__(self, model, imgsz):
        if model.training:
            model = model.eval()
        self.model = deepcopy(model)
        self.fused_model = None
        self.imgsz = imgsz

    def fused(self):
        raise NotImplementedError
