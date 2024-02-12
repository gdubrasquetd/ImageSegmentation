import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import torch.nn.functional as F

from UNET import UNET
from SegNet import SegNet

from utils import create_dir, seeding, rgb_to_class, class_to_rgb
import parameters as p


def calculate_metrics(pred, truth):
    
    np.set_printoptions(threshold=np.inf)
    
    truth = truth.cpu().numpy()
    truth = truth.reshape(-1)
    
    max_indices = torch.argmax(pred, dim=1)
    pred = F.one_hot(max_indices, p.nb_class)
    pred = pred.permute(0, 3, 1, 2)
    pred = pred.cpu().numpy()
    pred = pred.reshape(-1)

    score_jaccard = jaccard_score(truth, pred)
    score_f1 = f1_score(truth, pred)
    score_recall = recall_score(truth, pred)
    score_precision = precision_score(truth, pred)
    score_accuracy = accuracy_score(truth, pred)
        
    return [score_jaccard, score_f1, score_recall, score_precision, score_accuracy]


if __name__ == "__main__":
    
    seeding(p.seed)
    
    create_dir("results")
    
    test_x = sorted(glob(os.path.join(".", p.test_processed_path, "images", "*")))
    test_y = sorted(glob(os.path.join(".", p.test_processed_path, "masks", "*")))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET()
    model = model.to(device=device)
    model.load_state_dict(torch.load(p.checkpoint_path + "checkpoint.pth", map_location=device))
    model.eval()
    
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split("/")[-1]
        name = name.split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image / 255.0 # (512, 512, 3)
        x = np.transpose(x, (2, 0, 1)) # (3, 512, 512)
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        mask = cv2.imread(y, cv2.IMREAD_COLOR)
        y = np.transpose(mask, (2, 0, 1)) # (3, 512, 512)
        y = torch.from_numpy(y)
        y = rgb_to_class(y)
        y = y.to(device)

        with torch.no_grad():
            start = time.time()
            pred_y = model(x)
            total_time = time.time() - start
            time_taken.append(total_time)
            
            score = calculate_metrics(pred_y, y)
            metrics_score = list(map(add, metrics_score, score))
            
        pred_y = class_to_rgb(pred_y)
        pred_y = pred_y.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_y = cv2.cvtColor((pred_y[0]), cv2.COLOR_BGR2RGB)

        line = np.ones((p.size[1], 5, 3)) * 50
        
        cv2.putText(image, 'Image :', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 1)
        cv2.putText(mask, 'Ground Truth :', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 1)
        cv2.putText(pred_y , 'Prediction :', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 1)

        cat_images = np.concatenate([image, line, mask, line, pred_y], axis=1)
        cv2.imwrite(p.results_path + name + ".png", cat_images)
        
    jaccard = round(metrics_score[0]/len(test_x),3)
    f1 = round(metrics_score[1]/len(test_x),3)
    recall = round(metrics_score[2]/len(test_x),3)
    precision = round(metrics_score[3]/len(test_x),3)
    acc = round(metrics_score[4]/len(test_x),3)
    print("Jaccard:", jaccard, "- F1:", f1, "- Recall:", recall, "- Precision:", precision, "- Acc: ", acc)
    
    time_taken.pop(0)
    mean_time = np.mean(time_taken)
    print("Mean Time: ",round( mean_time, 5))
    print("Mean FPS: ", round(1 / mean_time))