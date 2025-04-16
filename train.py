import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DigitDetectionDataset
from model import get_model
from utils import collate_fn
from engine import train_one_epoch, evaluate, LOG


###### Hyper parameters ######
BATCH_SIZE = 16
NUM_CLASSES = 11
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 4
WEIGHT_DECAY = 0.0001
EPOCHS = 30
##############################

if __name__ == "__main__":

    if not os.path.exists(LOG):
        os.makedirs(LOG)

    # Set the random seed for reproducibility
    torch.manual_seed(313551058)

    # Load the JSON data
    with open(os.path.join("nycu-hw2-data", "train.json"), 'r') as f:
        train_json_data = json.load(f)
    with open(os.path.join("nycu-hw2-data", "valid.json"), 'r') as f:
        val_json_data = json.load(f)
    # Create the dataset
    train_dataset = DigitDetectionDataset(train_json_data, split="train")
    val_dataset = DigitDetectionDataset(val_json_data, split="valid")
    
    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Create the model
    model = get_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    ### assign different learning rates to different parts of the model
    backbone_params = []
    rpn_params = []
    roi_heads_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            elif "rpn" in name:
                rpn_params.append(param)
            elif "roi_heads" in name:
                roi_heads_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': rpn_params, 'lr': 1e-3},
        {'params': roi_heads_params, 'lr': 1e-3}
    ], weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    for epoch in range(EPOCHS):
        # # training for one epoch
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=100)

        # update the learning rate
        lr_scheduler.step()

        # # evaluate on the test dataset
        evaluate(model, val_loader, device=DEVICE)
        # Save the model
        torch.save(model.state_dict(), f"{LOG}/model_epoch_{epoch}.pth")

