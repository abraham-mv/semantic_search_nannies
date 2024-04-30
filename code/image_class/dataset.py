import os
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset



def class_labels_reassign(age):
    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6



def build_profile_pics_csv(directory: str, output_csv: str):
    """
    This function writes a csv file containing path to images and user ages extraced with Information Retrieval techniques.
    """
    # Get list of images
    file_list = os.listdir(directory)
    # Get user ages
    user_ages = pd.read_csv("data/output/python_tests/user_ages.csv")

    with open(output_csv, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'age', 'age_class'])
        for img_file in file_list:
          img_path = os.path.join(directory, img_file)
          img_id = int(img_file.split('.')[0])
          ages_df = user_ages[user_ages.id == img_id]["age"]
          if ages_df.any():
            age = ages_df.values[0]
            writer.writerow([img_file, img_path, age, class_labels_reassign(age)])
        print("CSV file file ready")
    pass




class ImagesDataset(Dataset): # inheritin from Dataset class
    def __init__(self, csv_file, root_dir="", augmentation=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.augmentation = augmentation

        initial_transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        ]

        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(45)
        ]

        if self.augmentation:
            self.transform = transforms.Compose(initial_transformations + augmentations)
        else:
            self.transform = transforms.Compose(initial_transformations)

    def __len__(self):
        return len(self.data) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.data.iloc[idx, 1]) #use image path column (index = 1) in csv file
        image = cv2.imread(image_path) # read image by cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB for matplotlib
        age = self.data.loc[idx, "age"] # use class name column (index = 2) in csv file
        age_class = self.data.loc[idx, "age_class"] # use class index column (index = 3) in csv file
        if self.transform:
            image = self.transform(image)
        return image, age, age_class
    
    def visualize(self):
        plt.figure(figsize=(12,6))
        for i in range(10):
            idx = np.random.randint(0, self.__len__())
            image, class_name, class_index = self.__getitem__(idx)
            ax=plt.subplot(2,5,i+1) # create an axis
            #ax.title.set_text(str(image.shape) + '-' + str(class_index)) # create a name of the axis based on the img name
            plt.imshow(image.permute(1, 2, 0)) # show the img

if __name__ == "__main__":
    build_profile_pics_csv("data/input/images/", "data/input/profiles_pics.csv")
