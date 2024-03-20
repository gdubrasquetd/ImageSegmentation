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

seed = 40

# Data augmentation

Global_Flip = False

Horizontal_Flip = False

Vertical_Flip = True

Rotation = False
Rotation_Angle = [20, 50]   	# Random value between 10 and 45 or -10 and -45

Brightness = True
Bright_Range = [25, 50]     	# Random value between 30 and 80 or -30 and -80

Blur = True

Sharpening = True

Zoom = False
Zoom_Range = [0.5, 0.95]     	# Random value between 0 and 1 

# Hyperparamters

model = "segnet"

loss = "diceloss"

size = (512, 512)

batch_size = 2
num_epochs = 50
lr = 0.0001
train_validation_split = 0.8
dropout_prob = 0

# Class

#region Eyes
# dataset_name = "Eyes"

# nb_class = 2

# class_colors = {
#     0 : (0, 0, 0),        #Background
#     1 : (255, 255, 255)   #Vessels
# }

# class_weights = {
#     0 : 0,
#     1 : 1
# }
#endregion

#region Face
dataset_name = "Face"

nb_class = 6

class_colors = {
    0 : (255, 0, 0),      #Background
    1 : (127, 0, 0),      #Hair
    2 : (255, 255, 0),    #Skin
    3 : (0, 0, 255),      #Eyes
    4 : (0, 255, 255),    #Nose  
    5 : (0, 255, 0)       #Mouth
}

class_weights = {
    0 : 1,
    1 : 1,
    2 : 1,
    3 : 1,
    4 : 1,
    5 : 1
}
#endregion

#region Satellite
# dataset_name = "Satellite"

# nb_class = 6

# class_colors = {
#     0 : (155, 155, 155),  #Unlabeled
#     1 : (226, 169, 41),   #Water   
#     2 : (254, 221, 58),   #Vegetation
#     3 : (132, 41, 246),   #Land
#     4 : (110, 193, 228),  #Road
#     5 : (60, 16, 152)     #Building
# }

# class_weights = {
#     0 : 1,
#     1 : 1,
#     2 : 1,
#     3 : 1,
#     4 : 1,
#     5 : 1
# }
#endregion

#region Drone
# dataset_name = "Drone"

# nb_class = 5

# class_colors = {
#     0 : (169, 169, 169),    # Ground
#     1 : (14, 135, 207),     # Water
#     2 : (124, 252, 0),      # Tree
#     3 : (155, 38, 182),     # Object
#     4 : (255, 20, 147)      # Human
# }

# class_weights = {
#     0 : 1,
#     1 : 1,
#     2 : 1,
#     3 : 1,
#     4 : 1
# }
#endregion

#region Car
# dataset_name = "Car"

# nb_class = 32

# class_colors = {
#     0 : (64, 128, 64),      # Animal 
#     1 : (192, 0, 128),      # Archway
#     2 : (0, 128, 192),      # Bicyclist
#     3 : (0, 128, 64),       # Bridge
#     4 : (128, 0, 0),        # Building
#     5 : (64, 0, 128),       # Car
#     6 : (64, 0, 192),       # CartLuggagePram
#     7 : (192, 128, 64),     # Child
#     8 : (192, 192, 128),    # Column_Pole
#     9 : (64, 64, 128),      # Fence
#     10 : (128, 0, 192),     # LaneMkgsDriv
#     11 : (192, 0, 64),      # LaneMkgsNonDriv
#     12 : (128, 128, 64),    # Misc_Text
#     13 : (192, 0, 192),     # MotorcycleScooter
#     14 : (128, 64, 64),     # OtherMoving
#     15 : (64, 192, 128),    # ParkingBlock
#     16 : (64, 64, 0),       # Pedestrian
#     17 : (128, 64, 128),    # Road
#     18 : (128, 128, 192),   # RoadShoulder
#     19 : (0, 0, 192),       # Sidewalk
#     20 : (192, 128, 128),   # SignSymbol
#     21 : (128, 128, 128),   # Sky
#     22 : (64, 128, 192),    # SUVPickupTruck
#     23 : (0, 0, 64),        # TrafficCone
#     24 : (0, 64, 64),       # TrafficLight
#     25 : (192, 64, 128),    # Train
#     26 : (128, 128, 0),     # Tree
#     27 : (192, 128, 192),   # Truck_Bus
#     28 : (64, 0, 64),       # Tunnel
#     29 : (192, 192, 0),     # VegetationMisc
#     30 : (0, 0, 0),         # Void
#     31 : (64, 192, 0)       # Wall
# }


# class_weights = {
#     0 : 1,  
#     1 : 1, 
#     2 : 1, 
#     3 : 1, 
#     4 : 1, 
#     5 : 1, 
#     6 : 1, 
#     7 : 1, 
#     8 : 1, 
#     9 : 1, 
#     10 : 1, 
#     11 : 1, 
#     12 : 1, 
#     13 : 1, 
#     14 : 1, 
#     15 : 1, 
#     16 : 1, 
#     17 : 1, 
#     18 : 1, 
#     19 : 1, 
#     20 : 1, 
#     21 : 1,     
#     22 : 1, 
#     23 : 1, 
#     24 : 1, 
#     25 : 1, 
#     26 : 1, 
#     27 : 1, 
#     28 : 1, 
#     29 : 1, 
#     30 : 1, 
#     31 : 1
# }
#endregion

#region Moto
# dataset_name = "Moto"

# nb_class = 6

# class_colors = {
#     0 : (245, 166, 35),     # Unlabelled
#     1 : (155, 155, 155),    # Road
#     2 : (248, 231, 28),     # Landmark
#     3 : (57, 234, 92),      # Other User
#     4 : (74, 144, 226),     # User Moto
#     5 : (65, 117, 6)        # User hands
# }

# class_weights = {
#     0 : 1,
#     1 : 1,
#     2 : 1,
#     3 : 3,
#     4 : 2,
#     5 : 2
# }
#endregion

#region Underwater
# dataset_name = "Underwater"

# nb_class = 8

# class_colors = {
#     0 : (0, 0, 0),          # Background
#     1 : (0, 0, 255),        # Humans
#     2 : (0, 255, 0),        # Sea-grass
#     3 : (0, 255, 255),      # Wrecks / Ruins
#     4 : (255, 0, 0),        # Robots
#     5 : (255, 0, 255),      # Reefs
#     6 : (255, 255, 0),      # Fish
#     7 : (255, 255, 255)     # Sea-floor / Rocks
# }

# class_weights = {
#     0 : 1,
#     1 : 1,
#     2 : 1,
#     3 : 1,
#     4 : 1,
#     5 : 1,
#     6 : 1,
#     7 : 1
# }
#endregion

# Metrics

precision = True
recall = True
f1_score = True

true_positive = False
true_negative = False
false_positive = False
false_negative = False

save_report = True

# Saved weights

checkpoint_path = "checkpoint/"

load_pretrained_model = False
pretrained_path = "checkpoint/checkpoint.pth"
