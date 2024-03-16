import shutil
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

from utils import create_dir, seeding, rgb_to_class, class_to_rgb, rgb_to_index, metrics_printer, plot_confusion_matrix, create_logfile, print_save
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
    
    truth_class = rgb_to_class(truth)
    truth_index = rgb_to_index(truth)
    
    pred_index = torch.argmax(pred, dim=1)
    pred_class = F.one_hot(pred_index, p.nb_class)

    pred_class = pred_class.permute(0, 3, 1, 2)
    pred_class = pred_class.squeeze(dim=0)
    
    truth_index = truth_index.cpu().numpy()
    truth_class = truth_class.cpu().numpy()
    pred_index = pred_index.cpu().numpy()
    pred_class = pred_class.cpu().numpy()
    
    accuracy = accuracy_score(truth_class, pred_class)
    jaccard = jaccard_score(truth_class, pred_class)
    weighted_dice = weighted_dice_score(truth_class, pred_class)
    matrix = confusion_matrix(truth_index, pred_index)

    precision,\
    recall,\
    f1_score,\
    true_positive,\
    true_negative,\
    false_positive,\
    false_negative = precision_recall_f1_score(truth_class, pred_class)
    
    confusion_matrix_list.append(matrix)
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
 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = create_dir("Results_" + timestamp)
    
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
        
        cv2.putText(image, 'Image :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(mask, 'Ground Truth :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(pred_y , 'Prediction :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, 'Image :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(mask, 'Ground Truth :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pred_y , 'Prediction :', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        
        cat_images = np.concatenate([image, line, mask, line, pred_y], axis=1)
        cv2.imwrite(os.path.join(results_path, name + ".png"), cat_images)

    # Printing and saving metrics
    
    if os.path.exists('loss_preview.png'):
        shutil.copy('loss_preview.png', os.path.join(results_path, 'loss_curve.png'))
        
    logfile = create_logfile(results_path)

    matrix = np.mean(confusion_matrix_list, axis=0)
    plot_confusion_matrix(matrix, results_path)

    accuracy = round(np.mean(global_metrics, axis=0)[0], 3)
    jaccard = round(np.mean(global_metrics, axis=0)[1], 3)
    weighted_dice = round(np.mean(global_metrics, axis=0)[2], 3)
    print_save(("Jaccard:" + str(jaccard) + " - Accuracy:" + str(accuracy) + " - Weighted Dice:" + str(weighted_dice) + "\n"), logfile)
    
    if p.precision : print_save(metrics_printer(precision_metrics, "Precision"), logfile)
    if p.recall : print_save(metrics_printer(recall_metrics, "Recall"), logfile)
    if p.f1_score : print_save(metrics_printer(f1_score_metrics, "F1 Score"), logfile)
    if p.true_positive : print_save(metrics_printer(true_positive_metrics, "True Positive"), logfile)
    if p.true_negative : print_save(metrics_printer(true_negative_metrics, "True Negative"), logfile)
    if p.false_positive : print_save(metrics_printer(false_positive_metrics, "False Positive"), logfile)
    if p.false_negative : print_save(metrics_printer(false_negative_metrics, "False Negative"), logfile)
    
    time_taken.pop(0)
    mean_time = np.mean(time_taken)
    print_save("Mean Time: " + str(round( mean_time, 5)), logfile)
    print_save("Mean FPS: " + str(round(1 / mean_time)), logfile)