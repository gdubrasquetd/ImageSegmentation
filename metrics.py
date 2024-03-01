import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters as p
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice


class MultiClassDiceLoss(torch.nn.Module):
    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, y_pred, y_true):

        assert y_pred.size() == y_true.size(), "Predicted and true tensors must have the same shape"
        
        smooth = 1e-7
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        intersection = (y_pred * y_true).sum()
        cardinality = (y_pred.sum() + y_true.sum())  
              
        dice_coefficient = (2. * intersection + smooth) / (cardinality + smooth)
        
        dice_loss = 1. - dice_coefficient
        
        return dice_loss


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth = 1e-7):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
      
        assert y_pred.size() == y_true.size(), ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)

        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)

        union = cardinality - intersection
        jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(smooth)

        return 1. - jaccard_score
    
class WeightedDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth = 1e-7):

        assert y_pred.size() == y_true.size(), ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)
        
        total_weights = sum(p.class_weights.values())
        dice_sum = 0
        for n in range(p.nb_class):
            weight = p.class_weights[n] / total_weights
            n_y_pred = y_pred[:, n, :, :].reshape(-1)
            n_y_true = y_true[:, n, :, :].reshape(-1)
            intersection = (n_y_pred * n_y_true).sum()
            cardinality = (n_y_pred.sum() + n_y_true.sum())  

            dice_sum += weight * (2. * intersection + smooth) / (cardinality + smooth)
            
        return 1. - dice_sum
    
    
def accuracy_score(y_true, y_pred):
    
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    assert y_true.shape == y_pred.shape, ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)

    correct_pixels = np.sum(y_true == y_pred)

    accuracy = correct_pixels / np.prod(y_true.shape)

    return accuracy


def jaccard_score(y_true, y_pred, smooth = 1e-7):
        
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    assert y_true.shape == y_pred.shape, ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)

    intersection = sum(y_pred * y_true)
    cardinality = sum(y_pred + y_true)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth)

    return jaccard_score
    
def weighted_dice_score(y_true, y_pred, smooth = 1e-7):
   
    assert y_true.shape == y_pred.shape, ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)
    
    total_weights = sum(p.class_weights.values())
    dice_sum = 0
    for n in range(p.nb_class):
        weight = p.class_weights[n] / total_weights
        n_y_pred = y_pred[n, :, :].reshape(-1)
        n_y_true = y_true[n, :, :].reshape(-1)
        intersection = (n_y_pred * n_y_true).sum()
        cardinality = (n_y_pred.sum() + n_y_true.sum())  

        dice_sum += weight * (2. * intersection + smooth) / (cardinality + smooth)
    
    return dice_sum


def confusion_matrix(y_true, y_pred):
    
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
        
    assert y_true.shape == y_pred.shape, ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)

    cm = np.zeros((p.nb_class, p.nb_class), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    return cm


def precision_recall_f1_score(y_true, y_pred, smooth = 1e-7):
    
    assert y_true.shape == y_pred.shape, ("Shape error : Predicted =", y_pred.shape, "Truth =", y_true.shape)

    precision_list = []
    recall_list = []
    f1_score_list = []
    true_positive_list = []
    true_negative_list = []
    false_positive_list = []
    false_negative_list = []
    
    y_true = torch.from_numpy(y_true)
    y_pred = torch.from_numpy(y_pred)
    
    y_true = torch.argmax(y_true, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)

    for class_id in range(p.nb_class):
        true_positive = torch.sum((y_true == class_id) & (y_pred == class_id)).item()
        true_negative = torch.sum((y_true == class_id) & (y_pred != class_id)).item()
        false_positive = torch.sum((y_true != class_id) & (y_pred == class_id)).item()
        false_negative = torch.sum((y_true == class_id) & (y_pred != class_id)).item()
        
        precision = true_positive / (true_positive + false_positive + smooth)
        recall = true_positive / (true_positive + false_negative + smooth)
        f1_score = 2 * (precision * recall) / (precision + recall + smooth)
                
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        true_positive_list.append(true_positive)
        true_negative_list.append(true_negative)
        false_positive_list.append(false_positive)
        false_negative_list.append(false_negative)
    
    return precision_list, recall_list, f1_score_list, true_positive_list, true_negative_list, false_positive_list, false_negative_list