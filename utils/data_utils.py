from cv2 import blur
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import random
import torchvision.transforms as transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.generate_augs import generate_RGB_image, generate_HSV_image, generate_distort_image, generate_blur_image, generate_noise_image, generate_random_image

class TrainDriveDataset(Dataset):
    def __init__(self, args, x, y):
        self.args = args
        
        self.x = x # names of images
        self.y = y # steering angles of images

        if self.args.img_dim:
            img_dim = int(args.img_dim)
            self.transform = transforms.Compose(
                [transforms.Resize((img_dim,img_dim)) ,
                transforms.ToTensor()]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor()]
            )

        self.curriculum_max = 0.

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

class ClassifyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.curriculum_max = 0.
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        return self.dataset[key]

    def increase_curr_max(self):
        self.curriculum_max += 0.1
    
    def get_curr_max(self):
        return self.curriculum_max
    
    def set_curr_max(self, cv):
        self.curriculum_max = cv

class TrainDriveDatasetNP(Dataset):
    def __init__(self, args, x, y):
        self.args = args
        
        self.x = x # numpy array of images
        self.y = y # steering angles of images

        if self.args.img_dim:
            img_dim = int(args.img_dim)
            self.transform = transforms.Compose(
                [transforms.Resize((img_dim,img_dim)) ,
                transforms.ToTensor()]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor()]
            )

        self.curriculum_max = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        img = self.x[key]
        label = self.y[key]

        img = Image.fromarray(img).convert("RGB")
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

# Just going to assume that this class uses npz files. If using images, then see first class for example
class TrainDriveDatasetPerturb(Dataset):
    def __init__(self, args, x, y):
        self.args = args
        
        self.x = x # numpy array of images
        self.y = y # steering angles of images

        if self.args.img_dim:
            img_dim = int(args.img_dim)
            self.transform = transforms.Compose(
                [transforms.Resize((img_dim,img_dim)) ,
                transforms.ToTensor()]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor()]
            )

        self.curriculum_max = 0
        self.i = 0
        self.methods = [self.perturb_brightness, self.perturb_contrast, self.perturb_saturation,
                        self.perturb_hue, self.perturb_noise, self.perturb_blur, self.perturb_distort]
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, key):
        img = self.x[key]
        label = self.y[key]

        clean_img = img.copy()
        clean_img = self.transform(clean_img)

        if self.i % len(self.methods) == 0:
            random.shuffle(self.methods)
    
        perturbation = self.methods[self.i%len(self.methods)]
        noise_img = perturbation(img)

        self.increase_i()

        return [clean_img, noise_img, label.astype(np.float32)]
    
    def increase_curr_max(self):
        self.curriculum_max += 0.1
    
    def get_curr_max(self):
        return self.curriculum_max
    
    def set_curr_max(self, cv):
        self.curriculum_max = cv
    
    def increase_i(self):
        self.i += 1

    def perturb_noise(self, img):
        intensity = np.random.uniform(high=self.curriculum_max)
        noise_level = int(intensity * (200 - 20) + 20)

        def add_noise(image, sigma):
            row,col,ch= image.shape
            mean = 0
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            noisy = np.float32(noisy)
            return noisy

        img = np.uint8(add_noise(img, noise_level))
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        return img

    def perturb_blur(self, img):
        intensity = np.random.uniform(high=self.curriculum_max)
        blur_level = int(intensity * (107 - 7) + 7)
        if blur_level % 2 == 0: # blur has to be an odd number
            blur_level += 1
        
        img = np.uint8(cv2.GaussianBlur(img, (blur_level, blur_level), 0))
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        return img

    def perturb_distort(self, img):
        intensity = np.random.uniform(high=self.curriculum_max)
        distort_level = int(intensity * (500 - 1) + 1)

        K = np.eye(3)*1000
        K[0,2] = img.shape[1]/2
        K[1,2] = img.shape[0]/2
        K[2,2] = 1

        img = np.uint8(cv2.undistort(img, K, np.array([distort_level,distort_level,0,0])))
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        return img

    def perturb_brightness(self, img):
        img = Image.fromarray(img).convert("RGB")

        brightness = transforms.ColorJitter(brightness=(0,1),
                                    contrast=(0,0),
                                    saturation=(0,0),
                                    hue=(0,0))

        img = brightness(img)
        img = self.transform(img)

        return img

    def perturb_contrast(self, img):
        img = Image.fromarray(img).convert("RGB")

        contrast = transforms.ColorJitter(brightness=(0,0),
                                    contrast=(0,1),
                                    saturation=(0,0),
                                    hue=(0,0))

        img = contrast(img)
        img = self.transform(img)

        return img

    def perturb_saturation(self, img):
        img = Image.fromarray(img).convert("RGB")

        saturation = transforms.ColorJitter(brightness=(0,0),
                                    contrast=(0,0),
                                    saturation=(0,1),
                                    hue=(0,0))

        img = saturation(img)
        img = self.transform(img)

        return img

    def perturb_hue(self, img):
        img = Image.fromarray(img).convert("RGB")

        hue = transforms.ColorJitter(brightness=(0,0),
                            contrast=(0,0),
                            saturation=(0,0),
                            hue=(-0.5, 0.5))

        img = hue(img)
        img = self.transform(img)

        return img

class TestDriveDataset(Dataset):
    def __init__(self, args, x, y, test_perturb, test_num):
        self.args = args
        self.x = x # name of clean image
        self.y = y # steering angle label

        self.test_perturb = test_perturb
        self.test_num = test_num

        self.transform = transforms.Compose(
                [transforms.ToTensor()]
            )

        assert (len(self.x) == len(self.y))
    
    def perturb(self, x):
        def get_aug_method(test_perturb):
            if test_perturb == "random":
                return "random", 0

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

        if test_perturb_name == "random":
            noise_image = generate_random_image(x, 1) # the 2nd value being passed is different from above methods as its curr max

        return noise_image

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        label = self.y[key]

        if self.test_perturb != "random":
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

        else: # This else is just for applying random perturbations to the images to do a sanity check
            img_path = f'{self.args.data_dir}/{self.args.dataset}/test/clean/{self.x[key]}' + ".jpg"
            
            if not os.path.isfile(img_path):
                print(img_path, " not exists")

            img = Image.open(img_path)
            img = np.asarray(img).copy()
            img = self.perturb(img)
            img = np.moveaxis(img, 0, -1) 
            img = Image.fromarray(np.uint8(img), "RGB")
        
        if self.args.img_dim:
            img = img.resize((self.args.img_dim, self.args.img_dim))
        
        img = self.transform(img)

        return [img, label.astype(np.float32)]