import os 
import numpy as np 
import cv2 
import torch
from torch.utils.data import Dataset
from utils import rgb_to_class
import matplotlib.pyplot as plt


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_paths):
        
        self.images_path = images_path
        self.masks_path = masks_paths
        self.n_samples = len(images_path)
        
    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0 # (512, 512, 3)
        image = np.transpose(image, (2, 0, 1)) # (3, 512, 512)
        image = image.astype(np.float32) 
        image = torch.from_numpy(image)
        
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_COLOR)
        mask = np.transpose(mask, (2, 0, 1)) # (3, 512, 512)
        mask = torch.from_numpy(mask)
        mask = rgb_to_class(mask)

        return image, mask
    
    def __len__(self):
        return self.n_samples