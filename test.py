import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from UNET import UNET
from SegNet import SegNet

from utils import create_dir, seeding
import parameters as p


def calculate_metrics(pred, truth):
    
    print("y_pred =", pred)
    print("y =",truth)
    
    truth = truth.cpu().numpy()
    # truth = truth > 0.5
    # truth = truth.astype(np.uint8)
    truth = truth.reshape(-1)
     
    pred = pred.cpu().numpy()
    # pred = pred > 0.5
    # pred = pred.astype(np.uint8)
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
    
    test_x = sorted(glob("./" + p.test_processed_path + "images/*"))
    test_y = sorted(glob("./" + p.test_processed_path + "masks/*"))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegNet()
    model = model.to(device=device)
    model.load_state_dict(torch.load(p.checkpoint_path + "checkpoint.pth", map_location=device))
    model.eval()
    
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split("/")[-1]
        name = name.split(".")[0]
        
        image = cv2.imread(x, cv2.IMREAD_COLOR) # (512, 512, 3)
        x = np.transpose(image, (2, 0, 1)) # (3, 512, 512)
        x = x /255
        x = np.expand_dims(x, axis=0) # (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        
        mask = cv2.imread(y, cv2.IMREAD_COLOR) # (512, 512, 3)
        y = np.transpose(mask, (2, 0, 1))  # (3, 512, 512)
        y = y /255
        y = np.expand_dims(y, axis=0) # (1, 3, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
     
        with torch.no_grad():
            start = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start
            time_taken.append(total_time)
            
            # score = calculate_metrics(pred_y, y)
            # metrics_score = list(map(add, metrics_score, score))
            
            pred_y = pred_y[0].cpu().numpy()
                
            # pred_y = pred_y > 0.5
            # pred_y = np.array(pred_y, dtype=np.uint8)
        

        pred_y = np.transpose(pred_y, (1, 2, 0))  # (512, 512, 3)

        line = np.ones((p.size[1], 10, 3)) * 128
        
        pred_y = cv2.cvtColor(pred_y * 255, cv2.COLOR_BGR2RGB)
        print(pred_y)
        cv2.imwrite(p.results_path + name + "test.png", pred_y)
        cv2.putText(image, 'Image :', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(mask, 'Ground Truth :', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(pred_y , 'Prediction :', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cat_images = np.concatenate(
            [image, line, mask, line, pred_y], axis=1
        )
        cv2.imwrite(p.results_path + name + ".png", cat_images)
        
    jaccard = round(metrics_score[0]/len(test_x),3)
    f1 = round(metrics_score[1]/len(test_x),3)
    recall = round(metrics_score[2]/len(test_x),3)
    precision = round(metrics_score[3]/len(test_x),3)
    acc = round(metrics_score[4]/len(test_x),3)
    print("Jaccard:", jaccard, "- F1:", f1, "- Recall:", recall, "- Precision:", precision, "- Acc: ", acc)

    mean_time = np.mean(time_taken)
    print("Mean Time: ",round( mean_time, 5))
    print("Mean FPS: ", round(1 / mean_time))