from typing import List
from torchvision import transforms
from ekya.datasets.CityscapesClassification import CityscapesClassification
from torch.utils.data import Dataset


class ScaledDataset(Dataset):
    def __init__(self, base_dataset, scale):
        self.base_dataset = base_dataset
        self.scale = scale

    def __len__(self):
        return self.scale * len(self.base_dataset)

    def __getitem__(self, index):
        print(index)
        base_index = index // self.scale
        return self.base_dataset[base_index]


TRAIN_CITIES = [
    "aachen",
    "bremen",
    "darmstadt", "erfurt", "hanover", "krefeld",
    "strasbourg", "tubingen", "weimar", "bochum", "cologne",
    "dusseldorf", "hamburg", "jena", "monchengladbach", "stuttgart",
    "ulm", "zurich"
]

VALID_CITIES = ["frankfurt", "lindau", "munster"]


def generate_dataset(mode: str, city_list: List[str]):
    transforms_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
    ])

    dataset = CityscapesClassification(
        root="/external-volume/dataset/cityscapes/",
        sample_list_name=city_list,
        sample_list_root="/external-volume/dataset/cityscapes/sample_lists/citywise",
        subsample_idxs=None,
        transform=transforms_data,
        target_transform=None,
        split=mode,
        mode="fine",
        use_cache=False,
        resize_res=224,
        label_type="human")

    return dataset


if __name__ == "__main__":
    dataset = generate_dataset(mode="train", city_list=TRAIN_CITIES)
    print(f"train dataset image cnt: {len(dataset)}")

    # scale = 3
    # dataset = ScaledDataset(dataset, scale)
    # print(f"x{scale} scaled train dataset image cnt: {len(dataset)}")
    # for i in range(6):
    #     print(f"data idx: {i}, value at [0,0,0]: {dataset[i][0][0,0,0]}")