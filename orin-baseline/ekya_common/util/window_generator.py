import os
import json
import math
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from yolox.data import COCODataset, TrainTransform, ValTransform
from torchvision.utils import save_image
from typing import List


TASK_TYPE = ["classification", "detection"]

# COCO2BOREAS = {
#     3:  0, # car
#     1:  1, # person
#     10: 2, # traffic light
#     8:  3, # truck
#     2:  4, # bicycle
# }

# def generate_dataset(task_type: str,
#                      img_size: Tuple[int],
#                      data_path: str,
#                      annotation_path: str):
#     dataset = COCODataset(data_dir=data_path,
#                           json_file=annotation_path,
#                           name="train",
#                           img_size=img_size,
#                           preproc=ValTransform(legacy=False))
    
#     if task_type == "classification":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(img_size),
#             # transforms.RandomCrop(img_size, padding=4),
#             # transforms.RandomHorizontalFlip(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])

#         dataset = ClassificationDataset(detection_dataset=dataset,
#                                         transform=transform,
#                                         img_size=img_size,
#                                         data_path=data_path,
#                                         annotation_path=annotation_path)
    
#     return dataset
LABEL_MAPPING = {
    "label_0": 0,
    "label_1": 1,
    "label_2": 2,
    "label_3": 3,
}


class WindowGenerator:
    @classmethod
    def generate_window(cls,
                        task_type: str,
                        img_size: Tuple[int],
                        data_path: str,
                        annotation_path: str,
                        img_dir_list: List[str]):
        if task_type not in TASK_TYPE:
            raise ValueError(f"invalid task type: {task_type}")

        windows = []

        if task_type == "classification":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            window_img_dir_names = os.listdir(data_path)
            num_windows = len(window_img_dir_names)

            for img_dir_path in img_dir_list:
                dataset = ImageFolder(root=img_dir_path,
                                      transform=transform,
                                      target_transform=lambda label: LABEL_MAPPING.get(label, label))
                windows.append(dataset)
        else:
            num_windows = len(os.listdir(annotation_path))

            for idx in range(num_windows):
                dataset = COCODataset(data_dir=data_path,
                          json_file=f"{annotation_path}/window_{idx}.json",
                          name="train",
                          img_size=img_size,
                          preproc=ValTransform(legacy=False))
                
                windows.append(dataset)

        return windows

import torch
import numpy as np
from torch.utils.data import TensorDataset
def static_sampling(img_tensor_per_label, sampling_rate):
    inputs = []
    targets = []

    total_num = np.sum([len(x) for x in img_tensor_per_label.values()])
    num_labels = len(img_tensor_per_label.keys())
    num_imgs_per_label = int(math.floor(total_num * sampling_rate) / num_labels)

    print(f"info -> total # of imgs: {total_num}, # of labels: {num_labels}, sampling rate: {sampling_rate * 100:.1f}%, # of sampled images per label: {num_imgs_per_label}")   

    for label in img_tensor_per_label.keys():
        imgs = img_tensor_per_label[label]

        num_imgs = len(imgs)
        # print(f"{label}: {num_imgs}")

        if num_imgs < num_imgs_per_label:
            raise ValueError(f"invalid # of imgs: {num_imgs}/{num_imgs_per_label}")
        
        step = int(math.floor(num_imgs / num_imgs_per_label))

        cnt = 0
        for offset in range(0, num_imgs, step):
            if cnt == num_imgs_per_label:
                break

            inputs.append(imgs[offset]["input"])
            targets.append(imgs[offset]["target"])
            cnt += 1

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return TensorDataset(inputs, targets)


if __name__ == "__main__":
    # windows = WindowGenerator.generate_window(task_type="classification",
    #                                           img_size=(224, 224),
    #                                           data_path=None,
    #                                           annotation_path=None,
    #                                           img_dir_list=json.load(open("/mnt/hdd/data/continual-learning-dataset/boreas/scenario/rainy-sunny-cloudy-snowy.json"))["window_path"])
    
    # dataset = windows[0]

    # >>> 
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
    
    dataset = ImageFolder(root="/mnt/hdd/data/continual-learning-dataset/boreas/cropped/4_label-human/fps_30-window_time_120/human/images/window_0",
                          transform=transform,
                          target_transform=lambda label: LABEL_MAPPING.get(label, label))
    # <<<  

    label_cnt = {}
    img_tensor_per_label = {}

    from torch.utils.data import DataLoader
    batch_size = 32
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        pin_memory=True,
                        num_workers=8,
                        drop_last=False)
    
    for inputs, targets in tqdm(loader):
        for b in range(targets.shape[0]):
            label = targets[b].item()
            if label in label_cnt.keys():
                label_cnt[label] += 1
            else:
                label_cnt[label] = 0

            if label not in img_tensor_per_label.keys():
                img_tensor_per_label[label] = [{
                    "input": inputs[b],
                    "target": targets[b]
                }]
            else:
                img_tensor_per_label[label].append({
                    "input": inputs[b],
                    "target": targets[b]
                })
    
    for label in label_cnt.keys():
        print(f"{label}: {label_cnt[label]}")

    print(f"sampled dataset: {len(dataset)}")
    label_cnt = {}
    dataset = static_sampling(img_tensor_per_label, 0.05)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        pin_memory=True,
                        num_workers=8,
                        drop_last=False)
    for inputs, targets in tqdm(loader):
        for b in range(targets.shape[0]):
            label = targets[b].item()
            if label in label_cnt.keys():
                label_cnt[label] += 1
            else:
                label_cnt[label] = 1

    for label in label_cnt.keys():
        print(f"{label}: {label_cnt[label]}")