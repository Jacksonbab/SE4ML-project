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
    def __init__(self, root, image_size=(672,188), transform=TRANSFORM, robustification=True, noise_level=10, num_duplicates=3, duplicate_threshold=0.1):


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
        
        print("Found {} images in {} directories".format(len(all_image_paths), len(self.dirs)))
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
                duplicate_data = [item for item in data for i in range(1)]
                self.csv_data.extend(duplicate_data)

        assert len(self.csv_data) == len(self.all_image_paths), "csv data does not match image paths"
        # print("first ten lines of csv data: {}".format(self.csv_data[-10:]))

        
        # balance dataset
        self.balance_dataset(interval=0.05)


        # duplicate steering angle greater than 0.1
        self.duplicate_steering_angle_greater_than(duplicate_threshold, num_duplicates=num_duplicates)
        print(f"duplicate steering angle greater than {duplicate_threshold}, time of duplicates: {num_duplicates}")
        print("total samples: {}".format(len(self)))

        # double the size of dataset
        # self.all_image_paths.extend(self.all_image_paths)
        # self.csv_data.extend(self.csv_data)






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
            rand_num = random.random()
            if rand_num > 0.8:
                gauss = kornia.filters.GaussianBlur2d((3, 3), (3.3, 3.3))
                image = gauss(image[None])[0]

                image = torch.clamp(image + (torch.randn(*image.shape) / self.noise_level), 0, 1)

        data = {
            "image": image,
            "speed": torch.FloatTensor([speed]),
            "steering_angle": torch.FloatTensor([steering_angle]),
            "all": torch.FloatTensor([speed, steering_angle])
        }


        return data
    

    def get_steering_angle(self, csv_data):
        return float(csv_data[-1])
    
    def get_speed(self, csv_data):
        return float(csv_data[3])

    def get_steering_angles(self):
        steering_angles = []
        for i in range(len(self)):
            steering_angles.append(self.get_steering_angle(self.csv_data[i]))
        return steering_angles

    def get_total_samples(self):
        return len(self)

    def get_directories(self):
        return self.dirs
    
    def get_image_by_steering_angle(self, steering_angle):
        # get image by steering angle
        for i in range(len(self)):
            data_point = self[i]
            if abs(data_point['steering_angle'] - steering_angle) < 0.1:
                img_tensor = data_point['image']
                img = ToPILImage()(img_tensor)
                return img
        return None

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
        moments['proportion of turning'] = len([i for i in arr if abs(i) > 0.2])/len(arr)
        return moments

    def duplicate_steering_angle_greater_than(self, threshold, num_duplicates=1):
        # expand dataset by duplicating images with steering angle greater than threshold
        # this is used to balance the dataset
        all_images = self.all_image_paths
        all_csv_data = self.csv_data
        new_images = []
        new_csv_data = []
        for i in range(len(all_images)):
            if abs(self.get_steering_angle(all_csv_data[i])) > threshold:
                new_images.append(all_images[i])
                new_csv_data.append(all_csv_data[i])

        self.all_image_paths.extend(new_images*num_duplicates)
        self.csv_data.extend(new_csv_data*num_duplicates)


    def balance_dataset(self, interval=0.1):
        # the miminum steering angle is -0.8 and the maximum steering angle is 0.8
        # we want to split the steering angle into intervals of size interval and balance the dataset by duplicating images in each interval
        # this is used to balance the dataset
        all_images = self.all_image_paths
        all_csv_data = self.csv_data

        num_intervals = int((0.8 - (-0.8))//interval)
        minimum_data_points = len(self)//num_intervals * 2
        maximum_data_points = len(self)//num_intervals * 2
        print("minimum data points per interval: {}".format(minimum_data_points))

        balanced_images = []
        balanced_csv_data = []
        for i in range(num_intervals):
            new_images = []
            new_csv_data = []
            if i == 0:
                interval_start = -0.8 + i*interval-1e-3
            else:
                interval_start = -0.8 + i*interval+1e-3
            interval_end = -0.8 + (i+1)*interval+1e-3
            for j in range(len(all_images)):
                if self.get_steering_angle(all_csv_data[j]) >= interval_start and self.get_steering_angle(all_csv_data[j]) < interval_end:
                    new_images.append(all_images[j])
                    new_csv_data.append(all_csv_data[j])
            if len(new_images) < minimum_data_points:
                balanced_images.extend(new_images*(minimum_data_points//len(new_images)))
                balanced_csv_data.extend(new_csv_data*(minimum_data_points//len(new_csv_data)))
            else:
                balanced_images.extend(new_images)
                balanced_csv_data.extend(new_csv_data)
            # elif len(new_images) > maximum_data_points:
            #     # randomly sample maximum_data_points from new_images
            #     new_images = random.sample(new_images, maximum_data_points)
            #     new_csv_data = random.sample(new_csv_data, maximum_data_points)
            #     balanced_images.extend(new_images)
            #     balanced_csv_data.extend(new_csv_data)
            # show only 1 digit after decimal point for interval_start and interval_end
            print("interval: {:.1f} -- {:.1f}, number of images: {}".format(interval_start, interval_end, len(new_images)))
        self.all_image_paths = balanced_images
        self.csv_data = balanced_csv_data
        print("total samples: {}".format(len(self)))





