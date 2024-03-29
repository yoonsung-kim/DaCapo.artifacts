import math
import numpy as np
from emulator.model import ModelPrecision
from emulator.profiler import Profiler, ProfileResult, TRAIN, INFERENCE, LABEL


class StaticProfiler(Profiler):
    def __init__(self, config):
        super().__init__(config)

        self.is_first = True

        self.static_epochs = self.config.initial_epoch
        self.static_m_T_p1 = ModelPrecision.MX9
        self.static_m_I_p1 = self.config.initial_m_I_p1
        self.static_m_L_p2 = ModelPrecision.MX6
        self.static_m_I_p2 = self.config.initial_m_I_p2
        self.static_f_I_p1 = self.config.initial_f_I_p1
        self.static_f_I_p2 = self.config.initial_f_I_p2
        self.static_sampling_rate = self.config.initial_sampling_rate

    def profile(self, **kwargs) -> ProfileResult:
        if self.is_first:
            self.is_first = False
            return self.generate_default_profile_result()
        
        D_t = int(math.floor((self.config.window_time * self.config.fps) * self.static_sampling_rate))

        iter_I_p1 = self.calculate_iter(m=self.static_m_I_p1,
                                        f=self.static_f_I_p1,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I_p1))])
        if drop_ratio_p1 != 0:
            # print("invalid config, inference drop at phase 1")
            return None

        iter_I_p2 = self.calculate_iter(m=self.static_m_I_p2,
                                        f=self.static_f_I_p2,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p2 = np.max([0., 1. - (1. / (self.fps * iter_I_p2))])
        if drop_ratio_p2 != 0:
            # print("invalid config, inference drop at phase 2")
            return None

        iter_T_p1 = self.calculate_iter(m=self.m_T_p1,
                                        f=self.total_row - self.static_f_I_p1,
                                        batch_size=self.batch_size,
                                        type=TRAIN)
        
        iter_L_p2 = self.calculate_iter(m=self.m_L_p2,
                                        f=self.total_row - self.static_f_I_p2,
                                        batch_size=1,
                                        type=LABEL)
    

        p2 = iter_L_p2 * D_t

        p1 = self.config.window_time - p2
        # p1 = iter_T_p1 * math.ceil(D_t / self.batch_size) * self.static_epochs
        # p2 = self.config.window_time - p1

        # if p1 + p2 > self.config.window_time or iter_L_p2 * D_t > p2:
        if p1 + p2 > self.config.window_time or (iter_T_p1 * (math.ceil(D_t / self.batch_size) * self.static_epochs)) > p1:
            # print(f"epoch: {self.static_epochs}, "
            #       f"sampling rate: {self.static_sampling_rate*100.:.1f}, "
            #       f"p1_time: {p1:.5f}, " 
            #       f"m_I_p1: {self.static_m_I_p1}, "
            #       f"f_I_p1: {self.static_f_I_p1}, "
            #       f"p2_time: {p2:.5f}, "
            #       f"m_I_p2: {self.static_m_I_p2}, "
            #       f"f_I_p2: {self.static_f_I_p2}, "
            #       f"label time: {iter_L_p2 * D_t}")
            # print("invalid config, cannot finish jobs within window time")
            return None
        
        return ProfileResult(p1_time=p1,
                             m_T_p1=self.static_m_T_p1,
                             m_I_p1=self.static_m_I_p1,
                             f_I_p1=self.static_f_I_p1,
                             iter_I_p1=iter_I_p1,
                             iter_T_p1=iter_T_p1,
                             p2_time=p2,
                             m_L_p2=self.static_m_L_p2,
                             m_I_p2=self.static_m_I_p2,
                             f_I_p2=self.static_f_I_p2,
                             iter_I_p2=iter_I_p2,
                             iter_L_p2=iter_L_p2,
                             epochs=self.static_epochs,
                             desired_sampling_rate=self.static_sampling_rate,
                             train_sampling_rate=self.static_sampling_rate,
                             fixed_valid_sampling_rate=0.)

    def generate_default_profile_result(self):
        # find case of no training cost and labeling maximum desired sampling rate
        results = []

        for m_I_p2 in self.M_I_p2:
            for f_I_p2 in self.F_I_p2:
                for m_I_p1 in self.M_I_p1:
                    value = self.__obj_func_without_training(m_I_p1=m_I_p1,
                                                             m_I_p2=m_I_p2,
                                                             f_I_p1=self.total_row,
                                                             f_I_p2=f_I_p2,
                                                             remain_time=self.config.window_time)
            
                    if value is not None:
                        results.append(value)

        results = sorted(results, key=lambda x: x.p1_time)

        if len(results) > 0:
            return results[0]
        
        raise ValueError(f"cannot start static profiler")
    
    def __obj_func_without_training(self,
                                    m_I_p1: int,
                                    m_I_p2: int,
                                    f_I_p1: int,
                                    f_I_p2: int,
                                    remain_time: int):
        iter_I_p1 = self.calculate_iter(m=m_I_p1,
                                        f=f_I_p1,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p1 = np.max([0., 1. - (1. / (self.fps * iter_I_p1))])
        
        iter_I_p2 = self.calculate_iter(m=m_I_p2,
                                        f=f_I_p2,
                                        batch_size=1,
                                        type=INFERENCE)
        drop_ratio_p2 = np.max([0., 1. - (1. / (self.fps * iter_I_p2))])

        if drop_ratio_p1 != 0 or drop_ratio_p2 != 0:
            return None

        num_total_imgs = self.config.fps * self.config.window_time
        sr_to_sample = self.static_sampling_rate
        num_imgs_to_sample = int(math.floor(num_total_imgs * sr_to_sample))

        iter_L_p2 = self.calculate_iter(m=self.m_L_p2,
                                        f=self.total_row - f_I_p2,
                                        batch_size=1,
                                        type=LABEL)
        
        p2 = iter_L_p2 * num_imgs_to_sample
        p1 = remain_time - p2

        if p1 <= 0 or p1 + p2 > remain_time:
            return None
        
        return ProfileResult(p1_time=p1,
                             m_T_p1=ModelPrecision.MX9,
                             m_I_p1=m_I_p1,
                             f_I_p1=f_I_p1,
                             iter_I_p1=iter_I_p1,
                             iter_T_p1=0.,
                             p2_time=p2,
                             m_L_p2=ModelPrecision.MX6,
                             m_I_p2=m_I_p2,
                             f_I_p2=f_I_p2,
                             iter_I_p2=iter_I_p2,
                             iter_L_p2=iter_L_p2,
                             epochs=0,
                             desired_sampling_rate=sr_to_sample,
                             train_sampling_rate=0.,
                             fixed_valid_sampling_rate=0.)


if __name__ == "__main__":
    pass