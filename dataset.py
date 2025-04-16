import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

train_transforms = A.Compose(
    [
        A.Rotate(limit=(-30, 30), p=0.5),
        A.ChannelShuffle(p=0.5),
        A.PlanckianJitter(p=0.5),
        A.RandomSunFlare(p=0.5, num_flare_circles_range=(1, 4)),
        A.RandomShadow(p=0.5),
        A.Erasing(p=0.5, scale=(0.02, 0.06), fill="random"),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams(
        format='pascal_voc', # Specify input format
        label_fields=['categories'] # Specify label argument name(s)
    ),
)
test_transforms = A.Compose(
    [
        # A.Resize(height=256, width=256),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams(
        format='pascal_voc', # Specify input format
        label_fields=['categories'] # Specify label argument name(s)
    ),
)

def pascal2coco(box):
    # Convert PASCAL VOC format to COCO format
    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
    w = x_max - x_min
    h = y_max - y_min
    return [x_min, y_min, w, h]
    
def coco2pascal(box):
    # Convert COCO format to PASCAL VOC format
    x, y, w, h = box[0], box[1], box[2], box[3]
    return [x, y, x + w, y + h]


#########################################################################################
# Dataset format:                                                                       #
#                                                                                       #
#   - "images":         {'id': 2, 'file_name': '2.png', 'height': 49, 'width': 115}     #
#   - "categories":     {'id': 1, 'name': '0'}                                          #
#   - "annotations":    {'id': 1, 'image_id': 1, 'bbox': [60.0, 14.0, 28.0, 52.0],      #
#                        'category_id': 7, 'area': 1456.0, 'iscrowd': 0}                #
#                                                                                       #
#########################################################################################

class DigitDetectionDataset(Dataset):

    def __init__(self, json_data, split):
        
        entry = os.path.join("nycu-hw2-data", split)
        self.split = split
        self.transforms = train_transforms if split == "train" else test_transforms  

        self.image_dirs = [
            os.path.join(entry, item["file_name"])
            for item in json_data["images"]
        ]

        self.categories2name = {
            item["id"]: item["name"]
            for item in json_data["categories"]
        }
        self.categories2name['0'] = 'ignored'

        # Initialize annotations with the same length as image_dirs
        # Each entry in annotations will be a list of dictionaries
        self.annotations = [ [] for _ in range(len(self.image_dirs)) ]
        
        for item in json_data["annotations"]:
            image_index = item["image_id"] - 1
            self.annotations[image_index].append({
                "bbox": item["bbox"],
                "category_id": item["category_id"],
                "iscrowd": item["iscrowd"]
            })

    def __getitem__(self, idx):
        
        image_path = self.image_dirs[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0

        
        target = { 
            "image_id": idx+1
        }
        iscrowds = [ annotation["iscrowd"] for annotation in self.annotations[idx] ]
        labels = [ annotation["category_id"] for annotation in self.annotations[idx] ]
        bboxes = [ 
            coco2pascal(annotation["bbox"])
            for annotation in self.annotations[idx] 
        ]

        target["labels"] = labels

        sample = self.transforms(image=image, bboxes=bboxes, categories=labels)
        
        while len(sample['bboxes']) == 0:
            sample = self.transforms(image=image, bboxes=bboxes, categories=labels)

        image = sample["image"]

        target['boxes'] = torch.tensor(
            sample['bboxes'], 
            dtype=torch.float32
        )

        target['labels'] = torch.tensor(
            list(map(int, sample['categories'])), 
            dtype=torch.int64
        )
        target["area"] = torch.tensor(
            [ (box[2]-box[0]) * (box[3]-box[1]) for box in target['boxes'] ],
            dtype=torch.float32
        )
        target["iscrowd"] = torch.tensor(iscrowds, dtype=torch.int64)

        return image, target
        

    def __len__(self):
        return len(self.image_dirs)
    

def visualize_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height  = pascal2coco(box)
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    fig.savefig('bbox.png')


if __name__ == "__main__":

    # Load the JSON data
    with open(os.path.join("nycu-hw2-data", "train.json"), 'r') as f:
        json_data = json.load(f)
    # Create the dataset
    dataset = DigitDetectionDataset(json_data, split="train")
    

    ### test whether dataset functions well
    print('length of dataset = ', len(dataset), '\n')
    img, target = dataset[78]
    print(img.dtype, '\n',target)

    ### test whether dataloader works
    from utils import collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    print('length of dataloader = ', len(dataloader), '\n')
    for i, data in enumerate(dataloader):
        print(data)
        break


    ### test whether visualization works
    img, target = dataset[13]
    visualize_bbox(img.permute(1, 2, 0).numpy(), target)
    print('visualization done')