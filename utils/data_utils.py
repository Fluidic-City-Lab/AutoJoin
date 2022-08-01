import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import random
import torchvision.transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.generate_augs import generate_RGB_image, generate_HSV_image, generate_distort_image, generate_blur_image, generate_noise_image

class TrainDriveDataset(Dataset):
    def __init__(self, args, x, y):
        self.args = args
        
        self.x = x # names of images
        self.y = y # steering angles of images

        if self.args.img_dim:
            img_dim = int(args.img_dim)
            self.transform = T.Compose(
                [T.Resize((img_dim,img_dim)) ,
                T.ToTensor()]
            )
        else:
            self.transform = T.Compose(
                [T.ToTensor()]
            )

        self.curriculum_max = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        img_path = f'{self.args.data_dir}/{self.args.dataset}/train/{self.x[key]}' + ".jpg"
        label = self.y[key]

        if not os.path.isfile(img_path):
            print(img_path, " not exists")

        img = Image.open(img_path)
        img = img.convert("RGB")
        # img = np.asarray(img)
        img = self.transform(img)

        # Correct datatype here
        return [img, label.astype(np.float32)]
    
    def increase_curr_max(self):
        self.curriculum_max += 0.1
    
    def get_curr_max(self):
        return self.curriculum_max
    
    def set_curr_max(self, cv):
        self.curriculum_max = cv


class TestDriveDataset(Dataset):
    def __init__(self, args, x, y, test_perturb, test_num):
        self.args = args
        self.x = x # name of clean image
        self.y = y # steering angle label

        self.test_perturb = test_perturb
        self.test_num = test_num

        self.transform = T.Compose(
                [T.ToTensor()]
            )

        assert (len(self.x) == len(self.y))
    
    def perturb(self, x):
        def get_aug_method(test_perturb):
            word_array = test_perturb.split('_')
            aug_method = ""

            for x in range(len(word_array)):
                if x != len(word_array) - 1:
                    aug_method += word_array[x] + " "
                else:
                    aug_method += word_array[x]
            
            aug_method_test = aug_method[:-2]
            aug_level_test = aug_method[-1]

            return aug_method_test, aug_level_test
        
        test_perturb_name, test_level = get_aug_method(self.test_perturb)
        
        rgb_dark_light = {
            0: [2, 4],
            1: [2, 5],
            2: [1, 4],
            3: [1, 5],
            4: [0, 4],
            5: [0, 5]
        }

        dark_light = {
            0: [0, 4],
            1: [0, 5],
            2: [1, 4],
            3: [1, 5],
            4: [2, 4],
            5: [2, 5]
        }

        rgb_hsv_levels = {
            "1": 0.02,
            "2": 0.2,
            "3": 0.5,
            "4": 0.65,
            "5": 1.0
        }

        if test_perturb_name == "R lighter":
            values = rgb_dark_light[5]
            noise_image = generate_RGB_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])
    
        if test_perturb_name == "R darker":
            values = rgb_dark_light[4]
            noise_image = generate_RGB_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "G lighter":
            values = rgb_dark_light[3]
            noise_image = generate_RGB_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "G darker":
            values = rgb_dark_light[2]
            noise_image = generate_RGB_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "B lighter":
            values = rgb_dark_light[1]
            noise_image = generate_RGB_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "B darker":
            values = rgb_dark_light[0]
            noise_image = generate_RGB_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "H darker":
            values = dark_light[0]
            noise_image = generate_HSV_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])
        
        if test_perturb_name == "H lighter":
            values = dark_light[1]
            noise_image = generate_HSV_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])
        
        if test_perturb_name == "S darker":
            values = dark_light[2]
            noise_image = generate_HSV_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])
        
        if test_perturb_name == "S lighter":
            values = dark_light[3]
            noise_image = generate_HSV_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "V darker":
            values = dark_light[4]
            noise_image = generate_HSV_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        if test_perturb_name == "V lighter":
            values = dark_light[5]
            noise_image = generate_HSV_image(x, values[0], values[1], dist_ratio=rgb_hsv_levels[test_level])

        blur_levels = {
            "1": 7,
            "2": 17,
            "3": 37,
            "4": 67,
            "5": 107
        }

        noise_levels = {
            "1": 20,
            "2": 50,
            "3": 100,
            "4": 150,
            "5": 200 
        }

        distort_levels = {
            "1": 1,
            "2": 10,
            "3": 50,
            "4": 200,
            "5": 500
        }

        if test_perturb_name == "blur":
            noise_image = generate_blur_image(x, blur_levels[test_level])

        if test_perturb_name == "noise":
            noise_image = generate_noise_image(x, noise_levels[test_level])
        
        if test_perturb_name == "distort":
            noise_image = generate_distort_image(x, distort_levels[test_level])

        return noise_image

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        label = self.y[key]

        if self.test_num < 1: # Clean 
            img_path = f'{self.args.data_dir}/{self.args.dataset}/test/clean/{self.x[key]}' + ".jpg"
        
            if not os.path.isfile(img_path):
                print(img_path, " not exists")

            img = Image.open(img_path)
            # img = img.convert("RGB")
            # img = np.moveaxis(np.asarray(img), -1, 0)

        elif self.test_num < 76: # Single Perturbation
            img_path = f'{self.args.data_dir}/{self.args.dataset}/test/clean/{self.x[key]}' + ".jpg"
        
            if not os.path.isfile(img_path):
                print(img_path, " not exists")

            img = Image.open(img_path)
            img = np.asarray(img).copy()
            img = self.perturb(img)
            img = np.moveaxis(img, 0, -1) 
            img = Image.fromarray(np.uint8(img), "RGB")
            
        else: # Combined and Unseen
            img_path = f'{self.args.data_dir}/{self.args.dataset}/test/{self.test_perturb}/{self.x[key]}' + ".jpg"

            img = Image.open(img_path)
            # img = np.moveaxis(np.asarray(img), -1, 0)
        
        if self.args.img_dim:
            img = img.resize((self.args.img_dim, self.args.img_dim))
        
        img = self.transform(img)

        return [img, label.astype(np.float32)]