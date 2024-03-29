import os
from src.config.config import Config
from typing import Tuple
from torch.utils.data import Subset
from yolox.data import COCODataset, TrainTransform, ValTransform


def generate_dataset(img_size: Tuple[int], data_path: str, annotation_path: str):
    dataset = COCODataset(data_dir=data_path,
                          json_file=annotation_path,
                          name="train",
                          img_size=img_size,
                          preproc=ValTransform(legacy=False))
    
    return dataset


def generate_window(img_size: Tuple[int],
                    data_path: str,
                    annotation_path: str):
    ann_path = f"{annotation_path}"

    num_windows = len(os.listdir(ann_path))

    windows = []

    for idx in range(num_windows):
        dataset = generate_dataset(img_size=img_size,
                                   data_path=data_path,
                                   annotation_path=f"{ann_path}/window_{idx}.json")
        
        windows.append(dataset)

    return windows


if __name__ == "__main__":
    config = Config("/home/yskim/projects/bfp-continual-learning.code/emulator/config/example.json")

    windows = generate_window(img_size=config.student_image_size,
                              data_path=config.data_path,
                              window_time=config.window_time,
                              fps=config.fps)
    
    print(f"# of windows: {len(windows)}")
    print(windows[0])
