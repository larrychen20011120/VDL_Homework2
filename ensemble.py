import os
import json
import torch
import torchvision.ops as ops  # 引入 NMS
from tqdm import tqdm
import cv2
import numpy as np
import csv
from collections import defaultdict
from model import get_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 11  # 0-9 digits + 1 background
BATCH_SIZE = 1 
NUM_WORKERS = 4

TEST_DIR = "nycu-hw2-data/test"
MODEL_CHECKPOINTS = ["Grod_shuffle/model_epoch_29.pth", "erasing_hierarchical/model_epoch_29.pth"]
OUTPUT_JSON = "pred.json"

SCORE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.5  # NMS IOU 閾值

def load_models():
    models = []
    for checkpoint in MODEL_CHECKPOINTS:
        model = get_model(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    return models

def run_inference(models, image_paths):
    all_results = []
    
    for path in tqdm(image_paths):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 存儲所有模型的預測結果
        all_boxes = []
        all_scores = []
        all_labels = []

        for model in models:
            outputs = model(image)

            for output in outputs:
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()

                all_boxes.extend(boxes)
                all_scores.extend(scores)
                all_labels.extend(labels)

        if len(all_boxes) == 0:
            continue

        # 使用 NMS 過濾重疊的框
        keep = ops.nms(torch.tensor(all_boxes), torch.tensor(all_scores), NMS_IOU_THRESHOLD)

        # 儲存過濾後的結果
        image_id = path.split("/")[-1].split(".")[0]
        for idx in keep:
            box = all_boxes[idx]
            score = all_scores[idx]
            label = all_labels[idx]
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            result = {
                "image_id": int(image_id),
                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                "score": float(score),
                "category_id": int(label)
            }
            all_results.append(result)

        # 如果沒有預測到任何框
        if len(keep) == 0:
            result = {
                "image_id": int(image_id),
                "bbox": [],
                "score": 0.0,
                "category_id": -1
            }
            all_results.append(result)

    return all_results

def save_results(results, output_json, output_csv):
    # 儲存 JSON 結果
    with open(output_json, "w") as f:
        json.dump(results, f)

    # 讀入 JSON 結果並生成 CSV
    grouped_preds = defaultdict(list)
    for pred in results:
        if pred["score"] >= SCORE_THRESHOLD:
            grouped_preds[pred["image_id"]].append(pred)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])

        all_image_ids = sorted(set(p["image_id"] for p in results))
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

if __name__ == "__main__":
    image_paths = os.listdir(TEST_DIR)
    image_paths = list(sorted([os.path.join(TEST_DIR, path) for path in image_paths]))

    models = load_models()
    results = run_inference(models, image_paths)
    save_results(results, OUTPUT_JSON, "pred.csv")
