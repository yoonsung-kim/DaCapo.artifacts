import math
from typing import List
from torch.utils.data import Subset
from src.config.config import Config
from torch.utils.data import ConcatDataset

'''
import torch
from torch.utils.data import random_split, Subset

# Assuming you have a dataset named 'dataset' that you want to create a subset from
dataset = ...  # Your dataset

# Define the size of the subset you want to create
subset_size = 100

# Get the total length of the dataset
dataset_length = len(dataset)

# Create a random permutation of indices
indices = torch.randperm(dataset_length)

# Select a random subset based on the defined size
subset_indices = indices[:subset_size]

# Create the random subset using Subset
subset = Subset(dataset, subset_indices)


'''


class Task:
    def __init__(self, task_type: str, reset_iterator: bool) -> None:
        self.task_type = task_type
        self.reset_iterator = reset_iterator

class BasicWindowScheduler:
    @classmethod
    def generate_schedule(cls, **kwargs) -> List[Task]:
        # aware
        epochs = Config.generate_epochs()
        
        results: List[Task] = []

        total_train_batchs = math.ceil(Config.num_img_per_window / Config.train_batch_size)

        for _ in range(epochs):
            for b_idx in range(total_train_batchs):
                results.append(Task(
                    task_type="train",
                    reset_iterator=True if (b_idx) % total_train_batchs == 0 else False
                ))

        for _ in range(Config.num_img_per_window):
            results.append(Task(
                task_type="inference",
                reset_iterator=False
            ))

        return results

    @classmethod
    def generate_epoch(cls) -> int:
        window_time = Config.window_time
        num_images = Config.num_img_per_window

        train_batch_size = Config.train_batch_size
        num_batch_for_train = math.ceil(num_images / train_batch_size)

        eta_train_per_batch = Config.eta_train_per_batch
        eta_infer_per_image = Config.eta_infer_per_image

        # TODO: aware teacher
        remain_window_time_for_train = window_time - (eta_infer_per_image * num_images)

        eta_per_epoch = eta_train_per_batch * num_batch_for_train

        return int(math.floor(remain_window_time_for_train / eta_per_epoch))
    
if __name__ == "__main__":
    schedules = BasicWindowScheduler.generate_schedule(train_window=None, infer_window=None)
    print(f"schedule length: {len(schedules)}")
    # for s in schedules:
    #     print(s)