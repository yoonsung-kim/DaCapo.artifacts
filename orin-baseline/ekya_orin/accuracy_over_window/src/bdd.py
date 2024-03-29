import json
import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import torchvision.transforms as transforms

from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader

class BDD_Dataset(Dataset):
    def __init__(self, img_paths, targets, img_names):
        self.img_paths = img_paths
        self.targets = targets
        self.img_names = img_names
        self.transform = dict()
        
        self.state = "train"
        self.transform["train"] = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        self.transform["val"] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
    def train(self):
        self.state = "train"
    
    def val(self):
        self.state = "val"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        target = self.targets[idx]
                
        img = Image.open(img_path).convert("RGB")        
        img = self.transform[self.state](img)
                
        return img, target

class CustomSampler(Sampler):
    def __init__(self, index_slice):
        self.indices = list(range(index_slice[0], index_slice[1] + 1))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class MystiqueDataset(Dataset):
    def __init__(self, scenario_dir: Path, *args, **kargs):
        super().__init__(*args, **kargs)

        self.scenario_dir = scenario_dir
        self.json_path = scenario_dir / "ann.json"
        self.image_dir = scenario_dir / "images"

        imgs = json.load(open(self.json_path))

        self.samples = []

        for img in imgs:
            self.samples.append((
                str(self.image_dir / img["image"]),
                img["category"],
                img["image"].split(".jpg")[0]
            ))

    def __getitem__(self, index):
        img_path = self.samples[index][0]
        label = self.samples[index][1]
        img_name = self.samples[index][2]

        return img_path, label, img_name

    def __len__(self):
        return len(self.samples)
    
    def generate_frame_latency_table(self, fps: int):
        latency = 1. / fps

        return [latency * (i) for i in range(len(self.samples))]
    
def split_dataset_indices_by_time(frame_latency_table: List[float],
                                  start_index: int,
                                  start_time: float,
                                  duration_time: float) -> Tuple[Tuple[int], int, float]:
    end_time = start_time + duration_time
    for index, value in enumerate(frame_latency_table):
        if value < end_time:
            last_index = index

    index_slice = (start_index, last_index)

    if last_index + 1 >= len(frame_latency_table):
        time = frame_latency_table[last_index] - frame_latency_table[start_index]
        return index_slice, -1, -1, time
    else:
        time = frame_latency_table[last_index + 1] - frame_latency_table[start_index]
        return index_slice, last_index + 1, frame_latency_table[last_index + 1], time

def generate_bdd_windows(scenario_dir, window_time):    
    dataset = MystiqueDataset(Path(scenario_dir))
    
    fps = 30
    start_idx = 0
    start_time = 0
    duration_time = window_time
    
    latency_table = dataset.generate_frame_latency_table(fps=fps)

    windows = list()
    
    while start_idx != -1:
        index_slice, \
        next_start_idx, \
        next_start_time, \
        time = split_dataset_indices_by_time(frame_latency_table=latency_table,
                                             start_index=start_idx,
                                             start_time=start_time,
                                             duration_time=duration_time)

        data_loader = DataLoader(dataset=dataset,
                                 sampler=CustomSampler(index_slice=index_slice),
                                 shuffle=False,
                                 batch_size=fps * duration_time,
                                 pin_memory=True)
        
        for inputs, labels, names in data_loader:
            window = BDD_Dataset(inputs, labels.tolist(), names)
            windows.append(window)
        
        start_idx = next_start_idx
        start_time = next_start_time
        
    return windows

if __name__ == "__main__":
    directory_path = "/mnt/hdd/data/dataset/continual-learning-dataset/bdd100k/scenario_0-type_1-clear"
    
    windows = generate_bdd_windows(directory_path)
    
    print(len(windows))
    print(len(windows[0]))