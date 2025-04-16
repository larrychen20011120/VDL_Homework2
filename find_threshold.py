import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
import cv2
import numpy as np
import csv
from collections import defaultdict
import torchvision.ops as ops


def get_labels_from_annotations(annotations):
    
    grouped_preds = defaultdict(list)
    for pred in annotations:
        grouped_preds[pred["image_id"]].append(pred)

    image2label = {}

    all_image_ids = sorted(set(p["image_id"] for p in annotations))
    for image_id in all_image_ids:
        boxes = grouped_preds.get(image_id, [])
        if not boxes:
            image2label[image_id] = -1
            continue
        
        try:
            boxes.sort(key=lambda x: x["bbox"][0])
        except:
            print(f"Error sorting boxes for image_id {image_id}: {boxes}")
            continue

        digits = [str(b["category_id"]-1) for b in boxes]
        full_number = int("".join(digits))
        image2label[image_id] = full_number

    return image2label


TEST_DIR = "nycu-hw2-data/valid"
CHECKPOINT_PATH = "Grod_shuffle/model_epoch_29.pth"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":

    with open(os.path.join("nycu-hw2-data", "valid.json"), 'r') as f:
        json_data = json.load(f)

    image_paths = os.listdir(TEST_DIR)
    image_paths = list(
        sorted(
            [ os.path.join(TEST_DIR, path) for path in image_paths ]
        )
    )

    # 載入模型與權重
    model = get_model(num_classes=11)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    results = []
    for path in tqdm(image_paths):

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        outputs = model(image)

        for output in outputs:
            image_id = path.split("/")[-1].split(".")[0]
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()

            ### NMS ###
            keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), 0.5)
            
            for idx in keep:
                box = boxes[idx]
                score = scores[idx]
                label = labels[idx]
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                result = {
                    "image_id": int(image_id),
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "score": float(score),
                    "category_id": int(label)
                }
                results.append(result)
            
            if len(keep) == 0:
                result = {
                    "image_id": int(image_id),
                    "bbox": [],
                    "score": 0.0,
                    "category_id": -1
                }
                results.append(result)

            ### NMS ###

            # for box, score, label in zip(boxes, scores, labels):
            #     x_min, y_min, x_max, y_max = box
            #     width = x_max - x_min
            #     height = y_max - y_min
            #     result = {
            #         "image_id": int(image_id),
            #         "bbox": [float(x_min), float(y_min), float(width), float(height)],
            #         "score": float(score),
            #         "category_id": int(label)
            #     }
            #     results.append(result)

            # if len(boxes) == 0:
            #     result = {
            #         "image_id": int(image_id),
            #         "bbox": [],
            #         "score": 0.0,
            #         "category_id": -1
            #     }
            #     results.append(result)

    image2labels = get_labels_from_annotations(json_data["annotations"])


    for threshold in np.arange(0.1, 1.0, 0.05):
        grouped_preds = defaultdict(list)
        for pred in results:
            if pred["score"] >= threshold:
                grouped_preds[pred["image_id"]].append(pred)

        image2preds = {}
        all_image_ids = sorted(set(p["image_id"] for p in results))
        for image_id in all_image_ids:
            boxes = grouped_preds.get(image_id, [])
            if not boxes:
                image2preds[image_id] = -1
                continue

            # 依照 bbox 的 x 座標排序
            boxes.sort(key=lambda x: x["bbox"][0])  # x_min 作為排序依據

            # 拼接數字
            digits = [str(b["category_id"]-1) for b in boxes]
            full_number = int("".join(digits))
            image2preds[image_id] = full_number

        # calculate accuracy
        

        correct = 0
        total = 0
        for image_id, pred in image2preds.items():
            if image_id in image2labels:
                total += 1
                if pred == image2labels[image_id]:
                    correct += 1
        accuracy = correct / total if total > 0 else 0
        print(f"Threhold: {threshold} | Accuracy: {accuracy:.4f}")
    