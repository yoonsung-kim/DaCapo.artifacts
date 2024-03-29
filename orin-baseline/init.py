import timm
import torch
from torchvision import models

if __name__ == "__main__":
    for model_name in ["resnet18", "resnet34", "wide_resnet50_2", "wide_resnet101_2"]:
        models.__dict__[model_name](pretrained=True)
    
    timm.create_model('vit_base_patch32_224.augreg_in1k', pretrained=True)
    timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True)