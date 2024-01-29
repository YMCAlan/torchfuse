from copy import deepcopy


class BasedFuser(object):
    def __init__(self, model, imgsz):
        self.model = deepcopy(model)
        self.fused_model = None
        self.imgsz = imgsz
