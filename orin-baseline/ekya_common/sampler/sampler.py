import torch
from src.config.config import Config
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset


class SamplerWithoutRehearsal:
    @classmethod
    def sample_from_window(cls, prev_window: Subset, curr_window: Subset) -> Subset:
        return curr_window
    
    @classmethod
    def get_name(cls):
        return "SamplerWithoutRehearsal"
    

class SamplerWithUniformSelection:
    @classmethod
    def sample_from_window(cls, prev_window: Subset, curr_window: Subset) -> Subset:
        num_images = Config.num_img_per_window

        subsets = [prev_window, curr_window]
        new_subsets = []

        for subset in subsets:
            indices = torch.randperm(num_images)
            new_subsets.append(Subset(subset, indices[0:int(num_images / 2)]))

        new_window = ConcatDataset(new_subsets)
        print(len(new_window))

        return new_window
    
    @classmethod
    def get_name(cls):
        return "SamplerWithUniformSelection"


if __name__ == "__main__":
    from src.dataset.cityscapes import generate_dataset, TRAIN_CITIES
    from src.util.window_generator import generate_windows

    dataset = generate_dataset(mode="train", city_list=TRAIN_CITIES)
    windows = generate_windows(num_image_per_window=Config.num_img_per_window, scale=1)
    print(len(windows))

    new_window = SamplerWithUniformSelection.sample_from_window(
        prev_window=windows[0],
        curr_window=windows[1])
    print(len(new_window))