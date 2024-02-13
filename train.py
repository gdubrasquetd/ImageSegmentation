import os
from glob import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from UNET import UNET
from SegNet import SegNet

from loss import DiceLoss, MultiClassDiceLoss, JaccardLoss, WeightedDiceLoss
from utils import seeding, create_dir, epoch_time, rgb_to_class, class_to_rgb
from data import DriveDataset
import parameters as p


def train(model, loader, optimizer, loss_function, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss/len(loader)

def evaluate(model, loader, loss_function, device):
    epoch_loss = 0.0
    model.eval()
    pred_show = []
    truth_show = []
    with torch.no_grad():
        for x,y in loader:
            truth_show = x
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
                            
            y_pred = model(x)
            pred_show = y_pred
            loss = loss_function(y_pred, y)
            epoch_loss += loss.item()
            
     # Save a preview of the model prediction 
    line = np.ones((p.size[1], 5, 3)) * 0
    pred_show = class_to_rgb(pred_show)
    # truth_show = class_to_rgb(truth_show)
    pred_show = pred_show.permute(0, 2, 3, 1).detach().cpu().numpy()
    pred_show = cv2.cvtColor((pred_show[0]), cv2.COLOR_BGR2RGB)
    truth_show = truth_show.permute(0, 2, 3, 1).detach().cpu().numpy()
    show = np.concatenate([pred_show, line, truth_show[0]*255], axis=1)
    cv2.imwrite("preview.png", show)
    
    return epoch_loss/len(loader)
        

if __name__ == "__main__":
    
    seeding(p.seed)
    
    create_dir(p.checkpoint_path)
    
    train_x = sorted(glob(os.path.join(".", p.train_processed_path, "images", "*")))
    train_y = sorted(glob(os.path.join(".", p.train_processed_path, "masks", "*")))
      
    split_index = int(len(train_x) * p.train_validation_split)
    
    train_x, validation_x = np.split(train_x, [split_index])
    train_y, validation_y = np.split(train_y, [split_index])
            
    print("Taining data :", len(train_x), len(train_y), "Validation data :", len(validation_x), len(validation_y))
        
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(validation_x, validation_y)
    
    train_losses = []
    val_losses = []
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=p.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    device = torch.device('cuda')
    model = UNET()
    if p.load_pretrained_model and os.path.exists(p.pretrained_path):
        model.load_state_dict(torch.load(p.pretrained_path, map_location=device))
    model = model.to(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, verbose=True)
    loss_function = WeightedDiceLoss()
    
    best_valid_loss = float("inf")
    
    times = []

    for epoch in range(p.num_epochs):
        start_time = time.time()
        
        training_loss = train(model, train_loader, optimizer, loss_function, device)
        validation_loss = evaluate(model, valid_loader, loss_function, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        train_losses.append(training_loss)
        val_losses.append(validation_loss)
        
        bash_width = os.get_terminal_size().columns
        
        times.append(epoch_mins * 60 + epoch_secs)
        estimated_time = (int)((np.mean(times) * (p.num_epochs - epoch)) / 60)
        
        text1 =f"Epoch : {epoch+1}"
        text2 = f"Remaining: {estimated_time} min"
        text3 = f"Epoch Time : {epoch_mins} min {epoch_secs} sec"
        
        print("=" * bash_width)
        print("{:<{}}{:^{}}{:>{}}".format(text1, bash_width // 3, text2, bash_width // 3, text3, bash_width // 3))
        print("Train Loss: ", round(training_loss, 5))
        print("Valid loss: ", round(validation_loss, 5))
        
        if validation_loss < best_valid_loss:
            print("Update Best Checkpoint.")

            best_valid_loss = validation_loss
            torch.save(model.state_dict(), p.checkpoint_path + "checkpoint.pth")
            
        if epoch == p.num_epochs:
            print("Final Checkpoint.")
            torch.save(model.state_dict(), p.checkpoint_path + "final_checkpoint.pth")

        if epoch > 0:
            epochs = np.arange(1, epoch+2)
            plt.plot(epochs, train_losses, label='Train Loss')
            plt.plot(epochs, val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss.png')
            plt.close()
