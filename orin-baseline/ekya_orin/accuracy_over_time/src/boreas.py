from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode

import json
import torch

def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms

class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)

class Dataset(ImageFolder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def find_classes(self, directory: str):
        TRANSFORMS = {
                "label_0": 0,
                "label_1": 1,
                "label_2": 2,
                "label_3": 3,
        }
        labels = list(x.name for x in Path(directory).iterdir() if x.is_dir())
        return (labels, {k:TRANSFORMS[k] for k in labels})
    
def load_data(data_dir):
    data_crop_size = 224
    interpolation = "bilinear"
    
    auto_augment_policy = None
    random_erase_prob = 0.0
    ra_magnitude = None
    augmix_severity = None
    
    dataset = Dataset(
        data_dir,
        ClassificationPresetTrain(
            crop_size=data_crop_size,
            interpolation=InterpolationMode(interpolation),
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend="PIL",
            use_v2=False,
        )
    )
    
    data_sampler = torch.utils.data.RandomSampler(dataset)
    
    return dataset, data_sampler

def generate_boreas_windows(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    windows = []
    labels = []
    
    for _, scene in enumerate(data["window_path"]):
        data_dir = scene
        dataset, data_sampler = load_data(data_dir)
        windows.append(dataset)
        labels.append(dataset.classes)
    
    return windows, labels
        
if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    
    windows, labels = generate_boreas_windows("/external-volume/dataset/boreas/boreas/scenario-for-emulator/version-0/cloudy-rainy-sunny-snowy-cloudy-rainy.json")
    
    data_loader = DataLoader(
            windows[0],
            batch_size=1,
            drop_last=False,
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )
    
    a = torch.unique(torch.tensor(windows[0].targets), return_counts=True)
    
    print(min(a[1]))
        