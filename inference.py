import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DigitDetectionDataset
from model import get_model
from utils import collate_fn
import cv2
import numpy as np
import csv
from collections import defaultdict
import torchvision.ops as ops

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 11  # 0-9 digits + 1 background
BATCH_SIZE = 1 
NUM_WORKERS = 4

TEST_DIR = "nycu-hw2-data/test"
CHECKPOINT_PATH = "Grod_shuffle/model_epoch_29.pth"
OUTPUT_JSON = "pred.json"

if __name__ == "__main__":
    
    image_paths = os.listdir(TEST_DIR)
    image_paths = list(
        sorted(
            [ os.path.join(TEST_DIR, path) for path in image_paths ]
        )
    )

    # 載入模型與權重
    model = get_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    results = []
    for path in tqdm(image_paths):

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (256, 256))
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

    # 儲存 JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f)


    # 參數設定
    PRED_JSON = "pred.json"
    OUTPUT_CSV = "pred.csv"
    SCORE_THRESHOLD = 0.45

    # 讀入 pred.json
    with open(PRED_JSON, "r") as f:
        preds = json.load(f)

    # 將 bbox 結果依照 image_id 分組
    grouped_preds = defaultdict(list)
    for pred in preds:
        if pred["score"] >= SCORE_THRESHOLD:
            grouped_preds[pred["image_id"]].append(pred)

    # 建立 csv 結果
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])

        all_image_ids = sorted(set(p["image_id"] for p in preds))
        for image_id in all_image_ids:
            boxes = grouped_preds.get(image_id, [])
            if not boxes:
                writer.writerow([image_id, -1])
                continue

            # 依照 bbox 的 x 座標排序
            boxes.sort(key=lambda x: x["bbox"][0])  # x_min 作為排序依據

            # 拼接數字
            digits = [str(b["category_id"]-1) for b in boxes]
            full_number = int("".join(digits))
            writer.writerow([image_id, full_number])
