import json
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Sampler
from torch.utils.data import Dataset


def generate_windows(scenario_path: Path, fps: int, window_time: int) -> List[Dataset]:
    num_total_imgs = len(json.load(open(scenario_path / "ann.json")))
    num_per_window = fps * window_time
    num_window = int(num_total_imgs / num_per_window)

    result: List[Dataset] = []
    for w in range(num_window):
        start_idx = w * num_per_window

        result.append(DatasetOfWindow(scenario_dir=scenario_path,
                                      transform=None,
                                      start_idx=start_idx,
                                      num_imgs=num_per_window))

    return result


class CustomSampler(Sampler):
    def __init__(self, indices, shuffle: bool = False):
        self.indices = indices
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DatasetOfWindow(Dataset):
    def __init__(self, scenario_dir: Path, transform, start_idx: int, num_imgs: int, *args, **kargs):
        super().__init__(*args, **kargs)

        self.scenario_dir = scenario_dir
        self.json_path = scenario_dir / "ann.json"
        self.image_dir = scenario_dir / "images"

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        imgs = json.load(open(self.json_path))

        self.samples = []

        for i in range(start_idx, start_idx+num_imgs):
            self.samples.append((
                str(self.image_dir / imgs[i]["image"]),
                imgs[i]["category"],
                imgs[i]["image"].split(".jpg")[0]
            ))

        # for img in imgs:
        #     self.samples.append((
        #         str(self.image_dir / img["image"]),
        #         img["category"],
        #         img["image"].split(".jpg")[0]
        #     ))

    def __getitem__(self, index):
        img_path = self.samples[index][0]
        label = self.samples[index][1]
        img_name = self.samples[index][2]
        
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            print("NONE!")
            img = img

        # return img, label, img_name
        return img, label
    
    def __len__(self):
        return len(self.samples)
    

class EntireDataset(Dataset):
    def __init__(self, scenario_dir: Path, transform, *args, **kargs):
        super().__init__(*args, **kargs)

        self.scenario_dir = scenario_dir
        self.json_path = scenario_dir / "ann.json"
        self.image_dir = scenario_dir / "images"

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

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
        
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = img

        return img, label, img_name

    def __len__(self):
        return len(self.samples)