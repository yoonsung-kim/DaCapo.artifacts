import csv
import numpy as np
from pathlib import Path
from emulator.config import Config
from emulator.statistic import WindowStatistic
from emulator.profile_result import ProfileResult
from emulator.model import ModelPrecision


TRAIN = "T"
INFERENCE = "I"
LABEL = "L"


class Profiler:
    def __init__(self, config: Config):
        self.config = config

        self.is_first = True

        # constants
        self.fps = self.config.fps # fps of streaming frame 
        self.m_T_p1 = ModelPrecision.MX9 # mantissa bit for inference at phase 1
        self.m_L_p2 = ModelPrecision.MX6 # mantissa bit for labeling at phase 2
        self.batch_size = self.config.train_batch_size # batch size for training
        self.total_row = self.config.total_row

        # variables
        self.M_I_p1 = [ModelPrecision.MX9, ModelPrecision.MX6, ModelPrecision.MX4] # mantissa bit for inference at phase 1
        self.M_I_p2 = [ModelPrecision.MX9, ModelPrecision.MX6, ModelPrecision.MX4] # mantissa bit for inference at phase 2
        self.F_I_p1 = np.arange(self.total_row - 1) + 1 # fraction of rows on systolic array for inference at phase 1
        self.F_I_p2 = np.arange(self.total_row - 1) + 1 # fraction of rows on systolic array for inference at phase 2

        self.device = self.config.profile_device


        self.simulation_table = {}

        student = self.config.student_model
        teacher = self.config.teacher_model
        train_batch_size = self.config.train_batch_size
        table_path = Path(self.config.table_path)
        freq = self.config.freq

        student_train_batched_path= table_path / "exec_time_result" / student / f"{student}-train-b{train_batch_size}-freq_{freq}_mhz.csv"
        student_infer_batched_path= table_path / "exec_time_result" / student / f"{student}-infer-b{train_batch_size}-freq_{freq}_mhz.csv"
        student_infer_path= table_path / "exec_time_result" / student / f"{student}-infer-b1-freq_{freq}_mhz.csv"
        teacher_infer_path= table_path / "exec_time_result" / teacher / f"{teacher}-label-b1-freq_{freq}_mhz.csv"

        csv_readers = [
            csv.reader(open(student_train_batched_path)),
            csv.reader(open(student_infer_path)),
            csv.reader(open(student_infer_batched_path)),
            csv.reader(open(teacher_infer_path))
        ]

        for csv_reader in csv_readers:
            next(csv_reader)

            for row in csv_reader:
                name, exec_time = str(row[0]), float(row[1])

                if name in self.simulation_table.keys():
                    raise ValueError(f"duplicated: {name}")
                
                self.simulation_table[name] = exec_time

        # with open(self.config.table_path, "r") as f:
        #     reader = csv.reader(f)
        #     next(reader)

        #     for row in reader:
        #         name = row[0]
        #         iter_time = float(row[1])
        #         self.simulation_table[name] = iter_time

        for f in self.F_I_p1:
            name = f"{ModelPrecision.MX9}_{f}_{self.batch_size}_T"
            if name not in self.simulation_table:
                    raise ValueError(f"no information: {name}")
            
            name = f"{ModelPrecision.MX6}_{f}_1_L"
            if name not in self.simulation_table:
                    raise ValueError(f"no information: {name}")

            for m in self.M_I_p1:
                name = f"{m}_{f}_1_I"
                if name not in self.simulation_table:
                    raise ValueError(f"no information: {name}")
                
                name = f"{m}_{f}_{self.batch_size}_I"
                if name not in self.simulation_table:
                    raise ValueError(f"no information: {name}")
                
    def calculate_iter(self, m: int, f: int, batch_size: int, type: str):
        return self.simulation_table[f"{m}_{f}_{batch_size}_{type}"]

    def profile(self,
                prev_statistic: WindowStatistic,
                max_train_sampling_rate: float,
                remain_time: float) -> ProfileResult:
        """
        prev_pred_loss_p1: float,
        prev_measured_loss_p2: float,
        prev_S_t: float,
        dataset: Dataset,
        student_parameter
        """
        raise ValueError(f"not implemented: use derived class from Profiler class")
    
    def generate_default_profile_result(self):
        half_row = int(self.total_row / 2.)
        half_window_time = int(self.config.window_time / 2.)

        return ProfileResult(p1_time=half_window_time,
                             m_T_p1=ModelPrecision.MX9,
                             m_I_p1=self.config.initial_m_I_p1,
                             f_I_p1=self.total_row,
                             iter_I_p1=self.calculate_iter(m=ModelPrecision.MX6,
                                                           f=self.total_row,
                                                           batch_size=1,
                                                           type=INFERENCE),
                             iter_T_p1=None,
                             p2_time=half_window_time,
                             m_L_p2=ModelPrecision.MX6,
                             m_I_p2=self.config.initial_m_I_p2,
                             f_I_p2=half_row,
                             iter_I_p2=self.calculate_iter(m=ModelPrecision.MX6,
                                                           f=half_row,
                                                           batch_size=1,
                                                           type=INFERENCE),
                             iter_L_p2=self.calculate_iter(m=ModelPrecision.MX6,
                                                           f=half_row,
                                                           batch_size=1,
                                                           type=LABEL),
                             epochs=0,
                             desired_sampling_rate=self.config.initial_sampling_rate,
                             train_sampling_rate=self.config.initial_sampling_rate,
                             fixed_valid_sampling_rate=0.)


if __name__ == "__main__":
    pass