import numpy as np
import os, cv2, csv
# from DAVE2 import DAVE2Model
# from DAVE2pytorch import DAVE2PytorchModel
import kornia

from PIL import Image
import copy
from scipy import stats
import torch.utils.data as data
from pathlib import Path
import skimage.io as sio
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import random
from torchvision.transforms import Compose, ToPILImage, ToTensor

from torchvision.transforms import Compose, ToTensor, PILToTensor, functional as transforms
# from io import BytesIO
# import skimage



TRANSFORM = Compose([
    ToTensor(),
    # PILToTensor(),
    # transforms.Resize((100,100)),
    # transforms.Grayscale(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
])

class MultiDirectoryDataSequence(data.Dataset):
    # this class is used to load data from multiple directories. 
    # each image is split into two images, and the corresponding steering angle is duplicated
    def __init__(self, root, image_size=(336,188), transform=TRANSFORM, robustification=False, noise_level=10):


        # store all input variables
        self.root = root
        self.transform = transform
        self.image_size = image_size
        
        self.robustification = robustification
        self.noise_level = noise_level

        # get all image paths and directories
        all_image_paths = []
        self.dirs = []

        for dir in os.listdir(root):
            if os.path.isdir(os.path.join(root, dir)):
                self.dirs.append(dir)
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith(".jpg"):
                        all_image_paths.append(os.path.join(root, dir, file))
                        all_image_paths.append(os.path.join(root, dir, file))
        
        print("Found {} images in {} directories".format(len(all_image_paths)//2, len(self.dirs)))
        print("Directories: {}".format(self.dirs))
        self.all_image_paths = all_image_paths


        # get corresponding csv file under root directory based on self.dirs
        csv_files = []
        for file in os.listdir(root):
            stem_name = "_".join(file.split("_")[:-1])
            if file.endswith(".csv") and stem_name in self.dirs:
                csv_files.append(os.path.join(root, file))
        
        print("Found {} csv files".format(len(csv_files)))
        print("csv files: {}".format(csv_files))

        assert len(csv_files) == len(self.dirs), "csv files do not match directories"
        self.csv_files = csv_files

        # get all csv data
        self.csv_data = []
        for file in self.csv_files:
            with open(file, newline='') as csvfile:
                data = list(csv.reader(csvfile))
                duplicate_data = [item for item in data for i in range(2)]
                self.csv_data.extend(duplicate_data)

        assert len(self.csv_data) == len(self.all_image_paths), "csv data does not match image paths"
        # print("first ten lines of csv data: {}".format(self.csv_data[-10:]))


    def __len__(self):
        return len(self.all_image_paths)
    
    def __getitem__(self, index):
        # get image path and csv data
        image_path = self.all_image_paths[index]
        csv_data = self.csv_data[index]
        # print("image path: {}".format(image_path))
        # print("csv data: {}".format(csv_data))

        # get image
        image = Image.open(image_path)
        #downsample image to self.image_size
        image = image.resize(self.image_size, Image.ANTIALIAS)
        image = self.transform(image)
        # print("transformed image shape: {}".format(transformed_image.shape))

        speed = float(csv_data[3])
        steering_angle = float(csv_data[-1])

        if self.robustification:
            if random.random() < 0.5:
                # flip image
                image = torch.flip(image, [2])
                # flip steering angle
                steering_angle = -steering_angle
            if random.random() > 0.5:
                gauss = kornia.filters.GaussianBlur2d((5, 5), (5.5, 5.5))
                image = gauss(image[None])[0]

            image = torch.clamp(image + (torch.randn(*image.shape) / self.noise_level), 0, 1)

        data = {
            "image": image,
            "speed": torch.FloatTensor([speed]),
            "steering_angle": torch.FloatTensor([steering_angle]),
            "all": torch.FloatTensor([speed, steering_angle])
        }


        return data
    def get_total_samples(self):
        return len(self)

    def get_directories(self):
        return self.dirs

    def get_outputs_distribution(self):
        all_steering_angles = []
        for i in range(len(self)):
            all_steering_angles.append(float(self.csv_data[i][-1]))
        moments = self.get_distribution_moments(all_steering_angles)
        return moments

    def get_distribution_moments(self, arr):
        moments = {}
        moments['shape'] = np.asarray(arr).shape
        moments['mean'] = np.mean(arr)
        moments['median'] = np.median(arr)
        moments['var'] = np.var(arr)
        moments['skew'] = stats.skew(arr)
        moments['kurtosis'] = stats.kurtosis(arr)
        moments['max'] = max(arr)
        moments['min'] = min(arr)
        return moments

