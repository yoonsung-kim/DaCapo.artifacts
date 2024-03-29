import numpy as np
from typing import List, Union, Iterable
from torch.utils.data import Subset
from itertools import combinations, permutations
from src.dataset.cityscapes import generate_dataset, TRAIN_CITIES, VALID_CITIES


def generate_scenario_by_group():
    PRETRAINED_CITY = [
        "aachen",
        "erfurt",
        "krefeld",
        "monchengladbach",
        "weimar",
    ]

    CITY_GROUP = {
        0: [
            "bochum",
            "dusseldorf"
        ],
        1: [
            "hamburg",
            "strasbourg"
        ],
        2: [
            "bremen",
            "cologne"
        ],
        3: [
            "hanover",
            "tubingen"
        ],
        4: [
            "ulm",
            "zurich"
        ],
        5: [
            "stuttgart",
            "darmstadt"
        ]
    }

    pretrained_weight_name = ""
    for i in range(len(PRETRAINED_CITY)):
        pretrained_weight_name += PRETRAINED_CITY[i][0:3]

        if i != len(PRETRAINED_CITY) - 1:
            pretrained_weight_name += "-"
    
    scenarios = []

    group_perms = list(permutations([0, 1, 2, 3, 4, 5]))
    np.random.RandomState(seed=128).shuffle(group_perms)
    # for group in group_perms:
    #     print(group)
    # exit()

    for groups in group_perms:
        scenario = []
        # print(groups)
        for g in groups:
            scenario += CITY_GROUP[g]
            # print(CITY_GROUP[g])
        # print(scenario)
        scenarios.append(scenario)
    #     exit()
    # exit()

    return scenarios, pretrained_weight_name


def generate_all_scenario():
    CUSTOM_LIST_FOR_SCENARIO = [
        "bochum",
        "dusseldorf",
        "hamburg",
        "strasbourg",
        "bremen",
        "cologne",
        # "frankfurt",
        # "munster"

        # # "bochum",
        # "dusseldorf",
        # "hamburg",
        # "strasbourg",
        # # "bremen",
        # "cologne",
        # # "frankfurt",
        # # "munster"
    ]

    results = []

    for city_for_pretraining in CUSTOM_LIST_FOR_SCENARIO:
        city_for_cl = []
        for city in CUSTOM_LIST_FOR_SCENARIO:
            if city_for_pretraining != city:
                city_for_cl.append(city)
        for perm in permutations(city_for_cl):
            # print(perm)
            row = [city_for_pretraining] + list(perm)
            results.append(row)

    return results

def generate_windows(num_image_per_window, city_list, scale) -> Union[List[Subset], List[int], str]:
    dataset = generate_dataset(mode="train", city_list=city_list, scale=scale)
    print(f"total length of data: {len(dataset)}")

    total_images = len(dataset)

    windows = []

    # with open(f"/home/yskim/projects/bfp-based-cl/dummy/labels.npy", "rb") as f:
    #     labels_list = np.load(f, allow_pickle=False)
    #     labels_list = labels_list.tolist()

    labels = []

    for ds_start_idx in range(0,
                              total_images,
                              num_image_per_window):
        if ds_start_idx + num_image_per_window > total_images:
            break

        sub_dataset = Subset(dataset,
                             list(range(ds_start_idx,
                                        ds_start_idx+num_image_per_window)))
        
        windows.append(sub_dataset)
        # labels.append(labels_list[ds_start_idx:ds_start_idx+num_image_per_window])
        labels.append([])

    return windows, labels, city_list[0]


if __name__ == "__main__":
    # generate_all_scenario()
    scenarios, pretrained_weight_path = generate_scenario_by_group()
    # for s in scenarios:
    #     print(s)
    # print(len(scenarios))
    # print(pretrained_weight_path)
    # print(scenarios[0])
    # exit()

    windows, labels = generate_windows(600, scale=1)
    print(f"window length: {len(windows)}")
    for i in range(len(windows)):
        print(f"{i}th -> length of window: {len(windows[i])}, labels: {len(labels[i])}")