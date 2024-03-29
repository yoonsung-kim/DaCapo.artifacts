from emulator.model import ModelPrecision
from emulator.config import Config
from emulator.profile_result import ProfileResult


class WindowStatistic:
    def __init__(self, window_index: int, config: Config, profile_result: ProfileResult):
        self.window_index = window_index
        self.config = config

        self.total_row = self.config.total_row
        #TODO: can be all None?

        # phase #1
        # inference
        self.p1_time: float = None
        self.p1_num_inference: int = None
        self.p1_num_imgs: int = None

        # target: ground truth
        # self.p1_net_avg_acc1: float = None
        self.p1_acc: float = None

        # phase #2
        # inference
        self.p2_time: float = self.config.window_time
        self.p2_num_inference: int = None
        self.p2_num_imgs: int = None

        # target: ground truth
        # self.p2_net_avg_acc1: float = None
        self.p2_acc: float = None

        # profiled results
        # used for next window
        self.pred_acc: float = 0.
        
        # target: teacher label
        self.p2_measured_metric: float = 0.
        self.prev_metric: float = 0.
        self.desired_sampling_rate: float = self.config.initial_sampling_rate
        self.train_sampling_rate: float = 0.
        self.fixed_valid_sampling_rate: float = 0.

        self.m_T_p1 = ModelPrecision.MX9
        self.m_I_p1 = self.config.initial_m_I_p1
        self.f_I_p1 = self.config.total_row
        self.iter_I_p1 = None

        self.m_L_p2 = ModelPrecision.MX6
        self.m_I_p2 = self.config.initial_m_I_p2
        self.f_I_p2 = self.config.initial_f_I_p2
        self.iter_I_p2 = None

        self.epochs = None

        self.window_acc = None

        # set feactures
        self.profile_result = profile_result
        
        if self.profile_result is not None:
            self.p1_time = self.profile_result.p1_time
            self.m_T_p1 = self.profile_result.m_T_p1
            self.m_I_p1 = self.profile_result.m_I_p1
            self.f_I_p1 = self.profile_result.f_I_p1
            self.iter_I_p1 = self.profile_result.iter_I_p1

            self.p2_time = self.profile_result.p2_time
            self.m_L_p2 = self.profile_result.m_L_p2
            self.m_I_p2 = self.profile_result.m_I_p2
            self.f_I_p2 = self.profile_result.f_I_p2
            self.iter_I_p2 = self.profile_result.iter_I_p2

            self.epochs = self.profile_result.epochs
            self.desired_sampling_rate = profile_result.desired_sampling_rate
            self.train_sampling_rate = profile_result.train_sampling_rate
            self.fixed_valid_sampling_rate = profile_result.fixed_valid_sampling_rate


    def summary(self):
        print(f"[>>>>> summary of {self.window_index}th window statistic >>>>>]")
        print(f"window time: {self.config.window_time} seconds")
        print(f"p1: {self.p1_time:.2f} seconds")
        print(f"T -> mantissa: {self.m_T_p1} bit, fraction: {self.total_row - self.f_I_p1}/{self.total_row}")
        print(f"I -> mantissa: {self.m_I_p1} bit, fraction: {self.f_I_p1}/{self.total_row}")
        print(f"sampling rate")
        print(f"desired: {self.desired_sampling_rate*100.:.1f}%")
        print(f"train:   {self.train_sampling_rate*100.:.1f}%")
        print(f"valid:   {self.fixed_valid_sampling_rate*100.:.1f}%")
        print(f"training epochs: {self.epochs}")
        print(f"p2: {self.p2_time:.2f} seconds")
        print(f"L -> mantissa: {self.m_L_p2} bit, fraction: {self.total_row - self.f_I_p2}/{self.total_row}")
        print(f"I -> mantissa: {self.m_I_p2} bit, fraction: {self.f_I_p2}/{self.total_row}")
        print(f"accuracy (p1 & p2): {self.window_acc:.1f}%")
        print(f"[<<<<< summary of {self.window_index}th window statistic <<<<<]")