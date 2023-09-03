from .model import *

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'Cvt':
        m = getCvt(config=config)
    elif model_type == 'LAGNet':
        m = getLAGNet(config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return m
