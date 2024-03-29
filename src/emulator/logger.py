import csv
from datetime import datetime
from emulator.config import Config
from emulator.statistic import WindowStatistic


class Logger:
    def __init__(self, config: Config, name: str):
        self.config = config

        current_time = datetime.now()
        year = current_time.year
        month = current_time.month
        day = current_time.day
        hour = current_time.hour
        min = current_time.minute
        seconds = current_time.second
        current_time = f"{year}-{month:02d}-{day:02d}-{hour:02d}-{min:02d}-{seconds:02d}"


        seed = self.config.seed
        num_classes = self.config.num_classes
        window_time = self.config.window_time
        fps = self.config.fps
        lr = self.config.lr

        scenario = ""

        # self.file_name = f"{name.lower()}-{config.student_model}-with-{config.teacher_model}.csv"
        self.file_name = f"result.csv"

        self.output_root = self.config.output_root
        self.path = f"{self.output_root}/{self.file_name}"

        self.header = [
            "window",
            "accuracy",
            "epoch",
            "label sampling rate",
            "train sampling rate",
            "valid sampling rate",

            "previous metric",
            "measured metric",

            "phase 1 time",
            "phase 1 accuracy",
            "phase 1 # of imgs",
            "m_T_p1",
            "m_I_p1",
            "f_T_p1",
            "f_I_p1",

            "phase 2 time",
            "phase 2 accuracy",
            "phase 2 # of imgs",
            "m_L_p2",
            "m_I_p2",
            "f_L_p2",
            "f_I_p2",
            ]
        
        # self.window_index = 0
        # self.corrects_path = f"{self.output_root}/{name.lower()}-{config.student_model}-with-{config.teacher_model}.json"
        self.corrects_path = f"{self.output_root}/result.json"
        self.corrects = {}

        with open(self.path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def write_window_info(self, stat: WindowStatistic):
        with open(self.path, "a") as f:
            writer = csv.writer(f)
            
            writer.writerow([
                # self.window_index,
                stat.window_index,
                stat.window_acc,
                stat.epochs,
                int(stat.desired_sampling_rate * 100.),
                int(stat.train_sampling_rate * 100.),
                int(stat.fixed_valid_sampling_rate * 100.),

                stat.prev_metric,
                stat.p2_measured_metric,

                # stat.p1_last_infer_time,
                stat.p1_time,
                stat.p1_acc,
                stat.p1_num_imgs,
                stat.m_T_p1,
                stat.m_I_p1,
                self.config.total_row - stat.f_I_p1,
                stat.f_I_p1,

                # stat.p2_last_infer_time,
                stat.p2_time,
                stat.p2_acc,
                stat.p2_num_imgs,
                stat.m_L_p2,
                stat.m_I_p2,
                self.config.total_row - stat.f_I_p2,
                stat.f_I_p2
            ])

            # self.window_index += 1
