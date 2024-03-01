import os
import time
import random
import numpy as np
import parameters as p
import torch
import matplotlib.pyplot as plt

from UNET import UNET
from SegNet import SegNet
from metrics import DiceLoss, MultiClassDiceLoss, JaccardLoss, WeightedDiceLoss

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    time_mins = int(elapsed_time / 60)
    time_secs = int(elapsed_time - (time_mins * 60))
    
    return time_mins, time_secs

# def class_to_rgb(y_pred):
#     y_pred_class = torch.argmax(y_pred, dim=1, keepdim=True) # (2, 1, 512, 512)
        
#     y_pred_color = torch.zeros((y_pred.shape[0], 3, y_pred.shape[2], y_pred.shape[3]), dtype=torch.uint8, device=y_pred.device)
#     for n in range(y_pred_class.shape[0]):
#         for i in range(y_pred_class.shape[2]):
#             for j in range(y_pred_class.shape[3]):
#                 color_index = int(y_pred_class[n, 0, i, j])
#                 color = p.class_colors[color_index]
#                 y_pred_color[n, :, i, j] = torch.tensor(color) # (2, 3, 512, 512)
#     return y_pred_color

class_colors_np = np.array([p.class_colors[i] for i in range(p.nb_class)], dtype=np.uint8)

def class_to_rgb(y_pred):

    y_pred_class = torch.argmax(y_pred, dim=1) # (2, 1, 512, 512)
    
    y_pred_class_index_np = y_pred_class.cpu().numpy()

    # Create tensor to color from indexes
    y_pred_color = np.take(class_colors_np, y_pred_class_index_np, axis=0)

    # Convert to torch
    y_pred_color = torch.from_numpy(y_pred_color).permute(0, 3, 1, 2)

    return y_pred_color


def rgb_to_class(y_true):
    
    y_true_class = torch.zeros((p.nb_class, y_true.shape[1], y_true.shape[2]), dtype=torch.uint8)
    test = []
    for i in range(y_true.shape[1]):
        for j in range(y_true.shape[2]):
            color_tensor = y_true[:, i, j]
            color = tuple(color_tensor.tolist())
            index = 0
            for key, value in p.class_colors.items():
                if abs(value[0]-color[2]) < 5 and abs(value[1]-color[1]) < 5 and abs(value[2]-color[0]) < 5:
                    index = key
                    test.append(index)

            y_true_class[index, i, j] = 1 # (nb_class, 512, 512)
    return y_true_class

def metrics_printer(list, name):
    result = name + " per classes :\n"
    for i in range(p.nb_class):
        result += str(i) + ": "
        result += str(round(np.mean(list, axis=0)[i], 2))
        result += " | "
    result += "\n"
    return result

def plot_confusion_matrix(matrix):
    
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')  # Sauvegarde de l'image en PNG
    plt.show()

def model_selector(model_name):
    match model_name.lower():
        case "unet":
            return UNET()
        case "segnet":
            return SegNet()
        
def loss_selector(loss_name):
    match loss_name.lower():
        case "diceloss":
            return WeightedDiceLoss()
        case "jaccard":
            return JaccardLoss()