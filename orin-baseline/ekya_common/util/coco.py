import io
import os
import json
import torch
import shutil
import tempfile
import contextlib
import torchvision
import torch.nn as nn
from typing import Tuple
from yolox.utils import xyxy2xywh
from yolox.data import COCODataset
from collections import defaultdict
from src.config.config import Config
from pycocotools.cocoeval import COCOeval
from torch.utils.data.dataloader import DataLoader

def postprocess(prediction, num_classes, conf_thre=0.01, nms_thre=0.65, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def convert_to_coco_format(outputs,
                           info_imgs,
                           ids,
                           img_size: Tuple[int],
                           coco_dataset: COCODataset,
                           return_outputs=False):
    data_list = []
    image_wise_data = defaultdict(dict)
    for (output, img_h, img_w, img_id) in zip(
        outputs, info_imgs[0], info_imgs[1], ids
    ):
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            img_size[0] / float(img_h), img_size[1] / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # >>> filter labels
        # VALID_COCO_LABELS = [3, 1, 10, 8, 2]
        VALID_COCO_LABELS = [2, 0, 9, 7, 1]
        # print("check length")
        assert(bboxes.shape[0] == scores.shape[0])
        
        result_bbox = []
        result_score = []
        result_category = []

        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx]
            score = scores[idx]

            class_id = int(cls[idx])
            # print(class_id)

            if class_id in VALID_COCO_LABELS:
                result_bbox.append(bbox.numpy().tolist())
                result_score.append(score.numpy().item())
                result_category.append(coco_dataset.class_ids[class_id])

        image_wise_data.update({
            int(img_id): {
                "bboxes": result_bbox,
                "scores": result_score,
                "categories": result_category,
            }
        })
        # <<<

        # image_wise_data.update({
        #     int(img_id): {
        #         "bboxes": [box.numpy().tolist() for box in bboxes],
        #         "scores": [score.numpy().item() for score in scores],
        #         "categories": [
        #             coco_dataset.class_ids[int(cls[ind])]
        #             for ind in range(bboxes.shape[0])
        #         ],
        #     }
        # })

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            class_id = int(cls[ind])
            if class_id in VALID_COCO_LABELS:
                label = coco_dataset.class_ids[class_id]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        # for ind in range(bboxes.shape[0]):
        #     label = coco_dataset.class_ids[int(cls[ind])]
        #     pred_data = {
        #         "image_id": int(img_id),
        #         "category_id": label,
        #         "bbox": bboxes[ind].numpy().tolist(),
        #         "score": scores[ind].numpy().item(),
        #         "segmentation": [],
        #     }  # COCO json format
        #     data_list.append(pred_data)

    if return_outputs:
        return data_list, image_wise_data
    return data_list


def evaluate_prediction(data_dict, data_loader: DataLoader):
    annType = ["segm", "bbox", "keypoints"]

    info = ""

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_dict) > 0:
        cocoGt = data_loader.dataset.coco

        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)

        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info += redirect_string.getvalue()
        # cat_ids = list(cocoGt.cats.keys())
        # cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
        # AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
        # info += "per class AP:\n" + AP_table + "\n"
        # AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
        # info += "per class AR:\n" + AR_table + "\n"
        return cocoEval.stats[0], cocoEval.stats[1], info
    else:
        return 0, 0, info


def generate_annotation_per_window(config: Config):
    data_root = config.data_path
    ann_path = f"{data_root}/annotations"
    ann_file = f"{ann_path}/instances_train.json"

    with open(ann_file, "r") as f:
        ann_dict = json.load(f)

    # prepare annotation info
    ann_of_image = {}
    for ann in ann_dict["annotations"]:
        img_id = ann["image_id"]

        if img_id not in ann_of_image.keys():
            ann_of_image[img_id] = [ann]
        else:
            ann_of_image[img_id].append(ann)

    # print(ann_dict["categories"])
    # print(ann_dict["type"])
    # print(len(ann_dict["images"]))

    # generate annotation files
    ann_per_window_root = f"{ann_path}/window_time{config.window_time}-fps{config.fps}"
    if os.path.isdir(ann_per_window_root):
        shutil.rmtree(ann_per_window_root)
    os.mkdir(ann_per_window_root)

    total_length = len(ann_dict["images"])
    num_imgs_per_window = config.num_images_per_window
    window_idx = 0
    for start_idx in range(0, total_length, num_imgs_per_window):
        if start_idx + num_imgs_per_window > total_length:
            break

        images_per_window = ann_dict["images"][start_idx:start_idx+num_imgs_per_window]
        annotations = []

        for img in images_per_window:
            img_id = img["id"]
            annotations += ann_of_image[img_id]

        # print(len(annotations))

        json_data = {
            "images": images_per_window,
            "type": "instances",
            "annotations": annotations,
            "categories": ann_dict["categories"]
        }

        ann_per_window_root
        with open(f"{ann_per_window_root}/window_{window_idx}.json", "w") as f:
            json.dump(json_data, f, indent=4)
        window_idx += 1


def preprocess(inputs, targets, tsize):
        scale_y = tsize[0] / tsize[0]
        scale_x = tsize[1] / tsize[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets


if __name__ == "__main__":
    config = Config("/home/yskim/projects/bfp-continual-learning.code/emulator/config/example.json")
    generate_annotation_per_window(config)