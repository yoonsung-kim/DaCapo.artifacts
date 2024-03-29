import os
import timm
import torch
import torch.nn as nn
from typing import List, Tuple
from torchvision import models
from torch.utils.data import Dataset
from bfp.bfp_config import BfpConfig
from bfp.bfp_model_converter import BfpModelConverter
from bfp.bfp_model_precision_changer import BfpModelPrecisionChanger

from emulator.config import Config
from util.accuracy import ClassificationAccuracyTracker, LossTracker


MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "vit_b_16",
    "vit_b_32",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

VIT_NAMES = {
    "vit_b_16": "vit_base_patch16_224.augreg_in1k",
    "vit_b_32": "vit_base_patch32_224.augreg_in1k",
}


class ModelPrecision:
    MX9 = 7
    MX6 = 4
    MX4 = 2


class Model:
    def __init__(self,
                 config: Config,
                 name: str,
                 precision: ModelPrecision,
                 batch_size: int,
                 num_classes: int,
                 device: torch.device,
                 freeze: bool,
                 weight_path: str = None):
        self.config = config
        self.name = name
        if self.name not in MODEL_NAMES:
            raise ValueError(f"invalid model name: {self.name}")
        
        if self.name == "vit_b_16" or self.name == "vit_b_32":
            if weight_path is not None:
                self.module = timm.create_model(VIT_NAMES[self.name], num_classes=num_classes)
                self.module.load_state_dict(torch.load(weight_path, map_location=device)["state_dict"])
            else:
                self.module = timm.create_model(VIT_NAMES[self.name], pretrained=True)
                last_linear = self.module.head
                self.module.head = nn.Linear(last_linear.in_features,
                                             num_classes,
                                             bias=False if last_linear.bias is None else True)
        else:
            if weight_path is not None:
                self.module = models.get_model(self.name, num_classes=num_classes)
                print(f"load weights... {os.path.basename(weight_path)}")
                self.module.load_state_dict(torch.load(weight_path, map_location=device)["state_dict"])
            else:
                self.module = models.get_model(self.name, pretrained=True)
                last_linear = self.module.fc
                self.module.fc = nn.Linear(last_linear.in_features,
                                           num_classes,
                                           bias=False if last_linear.bias is None else True)

        self.precision = precision
        self.row_fraction = None
        self.batch_size = batch_size
        self.iter_time = None
        self.device = device

        self.bfp_converter = BfpModelConverter()
        self.bfp_precision_changer = BfpModelPrecisionChanger()

        BfpConfig.use_bfp = True
        
        BfpConfig.bfp_M_Bit = self.precision
        BfpConfig.group_size = 16

        BfpConfig.use_mx = True

        BfpConfig.apply_single_bfp_tensor = True
        BfpConfig.prec_activation = True
        BfpConfig.prec_weight = True
        BfpConfig.prec_gradient = True
        
        self.bfp_converter.convert(self.module, ratio=1.0)

        # move to device
        self.module = self.module.to(self.device)

    def update_config(self, precision: ModelPrecision, row_fraction: float):
        self.precision = precision
        self.row_fraction = row_fraction
        self.iter_time = None
        self.bfp_precision_changer(self.module, m_bit=precision)

    def change_precision(self, precision: ModelPrecision):
        self.bfp_precision_changer.change_precision(self.module, m_bit=precision)

    def train(self,
              train_dataset: Dataset,
              valid_dataset: Dataset,
              epochs: int) -> float:
        raise ValueError(f"not implemented")
    
    def valid(self,
              valid_dataset: Dataset) -> float:
        raise ValueError(f"not implemented")
    
    def infer(self,
              phase: int,
              dataset: Dataset,
              index_slice: Tuple[int, int],
              phase_accuracy_tracker: ClassificationAccuracyTracker,
              phase_loss_tracker: LossTracker,
              window_accuracy_tracker: ClassificationAccuracyTracker):
        raise ValueError(f"not implemented")

    def train_iter(self, tensors: List[torch.Tensor], anns: List[dict]):
        raise ValueError(f"not implemented")

    def infer_iter(self, tensors: List[torch.Tensor] , anns: List[dict]):
        raise ValueError(f"not implemented")
    
    def label(self, dataset: Dataset) -> Dataset:
        raise ValueError(f"not implemented")
    
    def calculate_metric(self, dataset_list: List[Dataset]) -> float:
        raise ValueError(f"not implemented")


if __name__ == "__main__":
    pass