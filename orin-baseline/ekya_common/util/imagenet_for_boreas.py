import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
COCO2BOREAS = {
    3: 1, # car
    1: 2, # person
    10: 3, # traffic light
    8: 4, # truck
    2: 5, # bicycle
}
"""

# IMAGENET2BOREAS = {
#     # car
#     "n02814533": 0,
#     "n04037443": 0,
#     "n04285008": 0,

#     # traffic light
#     "n06874185": 1,

#     # bicycle
#     "n02835271": 2,

#     # truck
#     "n03345487": 3,
#     "n03417042": 3,
#     "n03930630": 3,
#     "n04461696": 3,
#     "n04467665": 3,
# }

# https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
IMAGENET2BOREAS = {
    # car
    "n02814533": 0,
    "n04037443": 0,
    "n04285008": 0,

    # traffic light
    "n06874185": 1,

    # truck
    "n03345487": 2,
    "n03417042": 2,
    "n03930630": 2,
    "n04461696": 2,
    "n04467665": 2,
}

class ImaeNetDatasetForBoreas(Dataset):
    def __init__(self,
                 imagenet_path: str,
                 mode: str,
                 transform: transforms,
                 even_dist: bool = False):
        super().__init__()

        self.imagenet_path = imagenet_path
        self.mode = mode
        self.transform = transform

        self.image_dir_path = Path(imagenet_path) / mode

        sample_dict = {
            0: [],
            1: [],
            2: [],
        }
    
        for dir in IMAGENET2BOREAS.keys():
            label = IMAGENET2BOREAS[dir]

            img_path = self.image_dir_path / dir

            img_names = sorted(os.listdir(img_path))

            for img_name in img_names:
                sample_dict[label].append((str(img_path / img_name), label))
                # self.samples.append((str(img_path / img_name), label))
        
        self.samples = []

        if even_dist:
            # for key in sample_dict.keys():
            raise ValueError(f"not implemented")
        else:
            for key in sample_dict.keys():
                self.samples += sample_dict[key]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        pil_image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = self.transform(pil_image)
        else:
            image = pil_image

        return image, label
            
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImaeNetDatasetForBoreas(imagenet_path="/mnt/hdd/data/imagenet",
                                      mode="train",
                                      transform=transform)

    from torch.utils.data.dataloader import DataLoader    
    data_loader = DataLoader(dataset=dataset,
                             num_workers=4,
                             pin_memory=False,
                             shuffle=False)
    
    print(len(data_loader))

    for data, label in data_loader:
        print(data.shape)
        print(label)
        break