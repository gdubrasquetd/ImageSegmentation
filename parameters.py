import torch
## PARAMETERS VALUES

# Original data
train_images_path = "original_data/train/images/*"
train_masks_path = "original_data/train/masks/*"
test_images_path = "original_data/test/images/*"
test_masks_path = "original_data/test/masks/*"

# Processed Data

train_processed_path = "processed_data/train/"
test_processed_path = "processed_data/test/"

# Seeding

seed = 42

# Data augmentation

Global_Flip = False

Horizontal_Flip = False

Vertical_Flip = False

Rotation = False
Rotation_Angle = [5, 30]   	# Random value between 10 and 45 or -10 and -45

Brightness = False
Bright_Range = [10, 40]     	# Random value between 30 and 80 or -30 and -80

Blur = False

Sharpening = False

Zoom = False
Zoom_Range = [0.5, 0.95]     	# Random value between 0 and 1 

# Hyperparamters

size = (512, 512)
batch_size = 2
num_epochs = 5
lr = 0.0001
train_validation_split = 0.8
dropout_prob = 0

# Class

nb_class = 6

class_colors = {
    (246, 41, 132): 0,
    (228, 193, 110): 1,
    (152, 16, 60): 2,
    (58, 221, 254): 3,  
    (155, 155, 155): 4,
    (41, 169, 41): 5
}

# Graph

graph_path = ""

# Results

results_path = "results/"

# Saved weights

checkpoint_path = "checkpoint/"

load_pretrained_model = False
pretrained_path = "checkpoint/checkpoint.pth"
