class ProfileResult:
    def __init__(self,
                 p1_time,
                 m_T_p1,
                 m_I_p1,
                 f_I_p1,
                 iter_I_p1,
                 iter_T_p1,
                 p2_time,
                 m_L_p2,
                 m_I_p2,
                 f_I_p2,
                 iter_I_p2,
                 iter_L_p2,
                 epochs,
                 desired_sampling_rate,
                 train_sampling_rate,
                 fixed_valid_sampling_rate):
        self.p1_time = p1_time
        self.m_T_p1 = m_T_p1
        self.m_I_p1 = m_I_p1
        self.f_I_p1 = f_I_p1
        self.iter_I_p1 = iter_I_p1
        self.iter_T_p1 = iter_T_p1

        self.p2_time = p2_time
        self.m_L_p2 = m_L_p2
        self.m_I_p2 = m_I_p2
        self.f_I_p2 = f_I_p2
        self.iter_I_p2 = iter_I_p2
        self.iter_L_p2 = iter_L_p2

        self.epochs = epochs
        self.desired_sampling_rate = desired_sampling_rate
        self.train_sampling_rate = train_sampling_rate
        self.fixed_valid_sampling_rate = fixed_valid_sampling_rate

    def __str__(self) -> str:
        ROWS = 32

        result = ""
        result += f"p1: {self.p1_time:>5.1f} seconds, # of rows: T/I {ROWS - self.f_I_p1}/{self.f_I_p1} ({ROWS}), infer MX: {self.m_I_p1} bit\n"
        result += f"p2: {self.p2_time:>5.1f} seconds, # of rows: L/I {ROWS - self.f_I_p2}/{self.f_I_p2} ({ROWS}), infer MX: {self.m_I_p2} bit\n"
        result += f"training epoch: {self.epochs}\n"
        result += f"desired sampling rate: {self.desired_sampling_rate * 100:.1f}%\n"
        result += f"training sampling rate: {self.train_sampling_rate * 100:.1f}%\n"
        result += f"fixed valid sampling rate: {self.fixed_valid_sampling_rate * 100:.1f}%"

        return result