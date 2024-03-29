from emulator.config import Config
from emulator.model import ModelPrecision
from emulator.profiler import Profiler, ProfileResult, INFERENCE


class EmptyProfiler(Profiler):
    def __init__(self, config: Config):
        super().__init__(config=config)
    
    def profile(self, **kwargs) -> ProfileResult:
        half_window_time = int(self.config.window_time / 2.)

        return ProfileResult(p1_time=half_window_time,
                             m_T_p1=ModelPrecision.MX9,
                             m_I_p1=self.config.initial_m_I_p1,
                             f_I_p1=self.total_row,
                             iter_I_p1=self.calculate_iter(self.config.initial_m_I_p1,
                                                           self.total_row,
                                                           1,
                                                           INFERENCE),
                             iter_T_p1=-1,
                             p2_time=half_window_time,
                             m_L_p2=ModelPrecision.MX6,
                             m_I_p2=self.config.initial_m_I_p2,
                             f_I_p2=self.total_row,
                             epochs=0,
                             desired_sampling_rate=0.,
                             train_sampling_rate=0.,
                             fixed_valid_sampling_rate=0.,
                             iter_I_p2=self.calculate_iter(self.config.initial_m_I_p1,
                                                           self.total_row,
                                                           1,
                                                           INFERENCE),
                             iter_L_p2=None)


if __name__ == "__main__":
    pass