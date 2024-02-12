import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters as p

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


# class JaccardLoss(nn.Module):
#     def __init__(self):
#         super(JaccardLoss, self).__init__()

#     def forward(self, y_pred, y_true):

#         smooth = 1e-7
        
#         assert y_pred.size() == y_true.size(), "Predicted and true tensors must have the same shape" \
#                                             + ("Predicted =", y_pred.shape, "Truth =", y_true.shape)

#         intersection = torch.sum(y_pred * y_true, dim=(2, 3))
#         union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true, dim=(2, 3)) - intersection

#         jaccard_score = (intersection + smooth) / (union + smooth)

#         jaccard_loss_per_class = 1 - jaccard_score # must process a loss for each classes

#         mean_jaccard_loss = jaccard_loss_per_class.mean()

#         return mean_jaccard_loss
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