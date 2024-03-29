import math
import numpy as np
from typing import Tuple
from emulator.config import Config
from emulator.model import ModelPrecision
from emulator.dataset import TrainSampleDataset
from emulator.profiler import Profiler, TRAIN, INFERENCE, LABEL


ACC_DIFF_THRESHOLD = -3. # as percentile

# SR_PAIRS = [
#     {
#         "train_sr": 216, # 108
#         "label_sr": 540,
#     },
#     {
#         "train_sr": 324, # 216
#         "label_sr": 432,
#     },
#     {
#         "train_sr": 432, # 324
#         "label_sr": 324,
#     },
#     {
#         "train_sr": 540, # 432
#         "label_sr": 216,
#     },
# ]
# SR_PAIRS = [
#     {
#         "train_sr": 216, # 108
#         "label_sr": 540,
#     },
#     # {
#     #     "train_sr": 324, # 216
#     #     "label_sr": 432,
#     # },
#     # {
#     #     "train_sr": 432, # 324
#     #     "label_sr": 324,
#     # },
#     {
#         "train_sr": 540, # 432
#         "label_sr": 216,
#     },
# ]

# for resnet18-wide_resnet_50_2
# SR_PAIRS = [
#     {
#         "train_sr": 540, # 108
#         "label_sr": 540,
#     },
# ]

# SR_PAIRS = [
#     {
#         "train_sr": 432, # 108
#         "label_sr": 432,
#     },
# ]

# SR_PAIRS = [
#     {
#         "train_sr": 324, # 108
#         "label_sr": 324,
#     },
# ]

SR_PAIRS = [
    {
        "train_sr": 432, # 108
        "label_sr": 432,
    },
]


# SR_PAIRS = [
#     {
#         "train_sr": 540, # 108
#         "label_sr": 540,
#     },
# ]

# for vit_b_32-vit_b_16
# SR_PAIRS = [
#     {
#         "train_sr": 171 + 108, # 108
#         "label_sr": 171 + 108,
#     },
# ]

LEN_SR_PAIRS = len(SR_PAIRS)
START_PAIR_INDEX = 0
START_TRAIN_SR = SR_PAIRS[START_PAIR_INDEX]["train_sr"]
START_LABEL_SR = SR_PAIRS[START_PAIR_INDEX]["label_sr"]

class DacapoProfiler(Profiler):
    def __init__(self, config: Config):
        super().__init__(config=config)

        self.M_I_p1 = ModelPrecision.MX6
        self.M_I_p2 = ModelPrecision.MX6

        self.is_first = True
        self.sr_pair_index = START_PAIR_INDEX

        self.F_I = np.arange(self.total_row - 1) + 1

    def get_best_train_config(self,
                              train_dataset: TrainSampleDataset) -> Tuple[int, int, float]:
        configs = []

        num_train_dataset = len(train_dataset)

        for f in self.F_I:
            iter_I = self.calculate_iter(m=ModelPrecision.MX6,
                                         f=f,
                                         batch_size=1,
                                         type=INFERENCE)
            
            drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I))])

            if drop_ratio_p1 != 0:
                continue
            
            iter_T = self.calculate_iter(m=ModelPrecision.MX9,
                                         f=self.total_row - f,
                                         batch_size=self.batch_size,
                                         type=TRAIN)
            
            train_time = iter_T * math.ceil(num_train_dataset / self.batch_size)

            configs.append({
                "I_f": f,
                "T_f": self.total_row - f,
                "train_time": train_time
            })

        configs = sorted(configs, key=lambda x: x["train_time"])

        config = configs[0]

        return config["I_f"], config["T_f"], config["train_time"]
    

    def get_best_valid_config(self,
                              valid_dataset: TrainSampleDataset) -> Tuple[int, int, float]:
        configs = []

        num_train_dataset = len(valid_dataset)

        for f in self.F_I:
            iter_I = self.calculate_iter(m=ModelPrecision.MX6,
                                         f=f,
                                         batch_size=1,
                                         type=INFERENCE)
            
            drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I))])

            if drop_ratio_p1 != 0:
                continue
            
            iter_V = self.calculate_iter(m=ModelPrecision.MX6,
                                         f=self.total_row - f,
                                         batch_size=self.batch_size,
                                         type=INFERENCE)
            
            train_time = iter_V * math.ceil(num_train_dataset / self.batch_size)

            configs.append({
                "I_f": f,
                "V_f": self.total_row - f,
                "valid_time": train_time
            })

        configs = sorted(configs, key=lambda x: x["valid_time"])

        config = configs[0]

        return config["I_f"], config["V_f"], config["valid_time"]

    def get_best_label_config(self,
                              num_data_to_sample: int) -> Tuple[int, int, float]:
        configs = []

        for f in self.F_I:
            iter_I = self.calculate_iter(m=ModelPrecision.MX6,
                                         f=f,
                                         batch_size=1,
                                         type=INFERENCE)
            
            drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I))])

            if drop_ratio_p1 != 0:
                continue
            
            iter_L = self.calculate_iter(m=ModelPrecision.MX6,
                                         f=self.total_row - f,
                                         batch_size=1,
                                         type=LABEL)
            
            label_time = iter_L * num_data_to_sample

            configs.append({
                "I_f": f,
                "L_f": self.total_row - f,
                "label_time": label_time
            })

        configs = sorted(configs, key=lambda x: x["label_time"])

        config = configs[0]

        return config["I_f"], config["L_f"], config["label_time"]

    def schedule(self,
                 prev_data_metric: float,
                 curr_data_metric: float) -> Tuple[int]:
        if prev_data_metric is None:
            print(f"previous data metric is None, return current sampling rate pair")
            
            sr_pair = SR_PAIRS[self.sr_pair_index]
            
            return sr_pair["train_sr"], sr_pair["label_sr"]

        diff = curr_data_metric - prev_data_metric

        if diff >= ACC_DIFF_THRESHOLD:
            if self.sr_pair_index + 1 < LEN_SR_PAIRS:
                self.sr_pair_index += 1

            sr_pair = SR_PAIRS[self.sr_pair_index]
            print(f"DECREASE SAMPLE IMAGES {sr_pair['label_sr']}")
        else:
            self.sr_pair_index = 0
            sr_pair = SR_PAIRS[self.sr_pair_index]
            print(f"INCREASE SAMPLE IMAGES {sr_pair['label_sr']}")

        return sr_pair["train_sr"], sr_pair["label_sr"]


if __name__ == "__main__":
    pass