import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F

from UNET import UNET
from SegNet import SegNet

from utils import create_dir, seeding, rgb_to_class, class_to_rgb, metrics_printer, plot_confusion_matrix
from metrics import precision_recall_f1_score, accuracy_score, weighted_dice_score, jaccard_score, confusion_matrix
import parameters as p

confusion_matrix_list = []
global_metrics = []
precision_metrics = []
recall_metrics = []
f1_score_metrics = []
true_positive_metrics = []
true_negative_metrics = []
false_positive_metrics = []
false_negative_metrics = []

def calculate_metrics(pred, truth):
    
    np.set_printoptions(threshold=np.inf)
    
    truth = truth.cpu().numpy()
    
    max_indices = torch.argmax(pred, dim=1)
    pred = F.one_hot(max_indices, p.nb_class)

    pred = pred.permute(0, 3, 1, 2)
    pred = pred.squeeze(dim=0)
    pred = pred.cpu().numpy()
    
    accuracy = accuracy_score(truth, pred)
    jaccard= jaccard_score(truth, pred)
    weighted_dice = weighted_dice_score(truth, pred)
    # matrix = confusion_matrix(truth, pred)

    precision, recall, f1_score, true_positive, true_negative, false_positive, false_negative = precision_recall_f1_score(truth, pred)
    
    # confusion_matrix_list.append(matrix)
    global_metrics.append([accuracy, jaccard, weighted_dice])
    precision_metrics.append(precision)
    recall_metrics.append(recall)
    f1_score_metrics.append(f1_score)
    true_positive_metrics.append(true_positive)
    true_negative_metrics.append(true_negative)
    false_positive_metrics.append(false_positive)
    false_negative_metrics.append(false_negative) 
    
    return


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
    
    metrics2_score = []
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
            
            calculate_metrics(pred_y, y)
            
        pred_y = class_to_rgb(pred_y)
        pred_y = pred_y.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_y = cv2.cvtColor((pred_y[0]), cv2.COLOR_BGR2RGB)

        line = np.ones((p.size[1], 5, 3)) * 100
        
        cv2.putText(image, 'Image :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)
        cv2.putText(mask, 'Ground Truth :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)
        cv2.putText(pred_y , 'Prediction :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)

        cat_images = np.concatenate([image, line, mask, line, pred_y], axis=1)
        cv2.imwrite(p.results_path + name + ".png", cat_images)

    # matrix = np.mean(confusion_matrix_list, axis=0)
    # plot_confusion_matrix(matrix)

    accuracy = round(np.mean(global_metrics, axis=0)[0], 3)
    jaccard = round(np.mean(global_metrics, axis=0)[1], 3)
    weighted_dice = round(np.mean(global_metrics, axis=0)[2], 3)
    print("Jaccard:", jaccard, "- Accuracy:", accuracy, "- Weighted Dice:", weighted_dice, "\n")
    
    if p.precision : print(metrics_printer(precision_metrics, "Precision"))
    if p.recall : print(metrics_printer(recall_metrics, "Recall"))
    if p.f1_score : print(metrics_printer(f1_score_metrics, "F1 Score"))
    if p.true_positive : print(metrics_printer(true_positive_metrics, "True Positive"))
    if p.true_negative : print(metrics_printer(true_negative_metrics, "True Negative"))
    if p.false_positive : print(metrics_printer(false_positive_metrics, "False Positive"))
    if p.false_negative : print(metrics_printer(false_negative_metrics, "False Negative"))
    
    time_taken.pop(0)
    mean_time = np.mean(time_taken)
    print("Mean Time: ",round( mean_time, 5))
    print("Mean FPS: ", round(1 / mean_time))