import torch
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

def generate_yolox(num_classes: int, depth: float, width: float, act: str):
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)

    return model

def yolox_nano(num_classes: int):
    depth = 0.33
    width = 0.25

    in_channels = [256, 512, 1024]
    # NANO model use depthwise = True, which is main difference.
    backbone = YOLOPAFPN(
        depth, width, in_channels=in_channels,
        act="silu", depthwise=True,
    )
    head = YOLOXHead(
        num_classes, width, in_channels=in_channels,
        act="silu", depthwise=True
    )
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)

    return model

def yolox_tiny(num_classes: int):
    return generate_yolox(num_classes=num_classes,
                          depth=0.33,
                          width=0.375,
                          act="silu")

def yolox_medium(num_classes: int):
    return generate_yolox(num_classes=num_classes,
                          depth=0.67,
                          width=0.75,
                          act="silu")