import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import random
import parameters as p

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    

def augment_data(images, masks, save_path, augmented = True):
        
    for _, (x_path, y_path) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = x_path.split("/")[-1]
        name = name.split(".")[0]

        x = cv2.imread(x_path, cv2.IMREAD_COLOR)
        y = cv2.imread(y_path, cv2.IMREAD_COLOR)
        
        if x_path is None:
            print("Can't open : ", x_path)

        if y_path is None:
            print("Can't open : ", y_path)

        X = [x]
        Y = [y]
        
        if(augmented):
            if(p.Global_Flip):
                x_augment = cv2.flip(x, -1)
                y_augment = cv2.flip(y, -1)
                X.append(x_augment)
                Y.append(y_augment)
                
            if(p.Horizontal_Flip):
                x_augment = cv2.flip(x, 0)
                y_augment = cv2.flip(y, 0)
                X.append(x_augment)
                Y.append(y_augment)

            if(p.Vertical_Flip):
                x_augment = cv2.flip(x, 1)
                y_augment = cv2.flip(y, 1)
                X.append(x_augment)
                Y.append(y_augment)
                
            if(p.Rotation):
                height, width = x.shape[:2]
                angle = random.randrange(p.Rotation_Angle[0], p.Rotation_Angle[1])
                rotation_matrix = cv2.getRotationMatrix2D((height/2, width/2), random.choice([-angle, angle]), 1)
                x_augment = cv2.warpAffine(x, rotation_matrix, (width, height))
                y_augment = cv2.warpAffine(y, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)
                X.append(x_augment)
                Y.append(y_augment)
                
            if(p.Brightness):
                value = random.randrange(p.Bright_Range[0], p.Bright_Range[1])
                bright = np.ones(x.shape, dtype="uint8") * value
                x_augment = cv2.add(x, bright)
                X.append(x_augment)
                x_augment = cv2.subtract(x, bright)
                X.append(x_augment)
                Y.append(y)
                Y.append(y)
                            
            if(p.Blur):
                x_augment = cv2.GaussianBlur(x, (5,5), cv2.BORDER_DEFAULT)
                X.append(x_augment)
                Y.append(y)
                
            if(p.Sharpening):
                sharpening = np.array([[0, -1, 0,],
                                    [-1, 5, -1,],
                                    [0, -1, 0,]])
                x_augment = cv2.filter2D(x, -1, sharpening)
                X.append(x_augment)
                Y.append(y)

            if(p.Zoom):
                zoom_percent = np.random.uniform(p.Zoom_Range[0], p.Zoom_Range[1])
                height, width = x.shape[:2]
                top = np.random.randint(0, int(height * (1 - zoom_percent)))
                left = np.random.randint(0, int(width * (1 - zoom_percent)))
                bottom = top + int(height * zoom_percent)
                right = left + int(width * zoom_percent)
                x_augment = x[top:bottom, left:right]
                y_augment = y[top:bottom, left:right] 
                X.append(x_augment)
                Y.append(y_augment)
            
        index = 0
        for img, msk in zip(X, Y):
            img = cv2.resize(img, p.size)
            msk = cv2.resize(msk, p.size, interpolation=cv2.INTER_NEAREST)
            
            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"
            
            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)
            
            cv2.imwrite(image_path, img)
            cv2.imwrite(mask_path, msk)
            index += 1        
        

if __name__ == "__main__":
    #Seeding
    np.random.seed(p.seed)
    
    #Load data
    train_x = sorted(glob(p.train_images_path))
    train_y = sorted(glob(p.train_masks_path))
 
    test_x = sorted(glob(p.test_images_path))
    test_y = sorted(glob(p.test_masks_path))
    
    #Create newdirs
    create_dir(p.train_processed_path + "images")
    create_dir(p.train_processed_path + "masks")
    create_dir(p.test_processed_path + "images")
    create_dir(p.test_processed_path + "masks")
    
    #Data Augmentation
    augment_data(train_x, train_y, p.train_processed_path, True)
    augment_data(test_x, test_y, p.test_processed_path, False)