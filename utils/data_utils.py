import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import random
import torchvision.transforms as T

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class DriveDataset(Dataset):
    def __init__(self, images, targets):
        self.image_list = images
        self.target_list = targets

        self.curriculum_max = 1

        assert (len(self.image_list) == len(self.target_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, key):
        img = self.image_list[key]
        target = self.target_list[key]

        return [img.astype(np.float32), target.astype(np.float32)]
    
    def increase_curr_max(self):
        self.curriculum_max += 0.1
    
    def get_curr_max(self):
        return self.curriculum_max
    
    def set_curr_max(self, cv):
        self.curriculum_max = cv


class DriveDatasetImage(Dataset):
    def __init__(self, args, x, y):
        self.args = args
        img_dim = int(args.img_dim)
        self.x = x # names of images
        self.y = y # steering angles of images

        # self.transform = transforms.Compose(
        #     [transforms.Resize((img_dim,img_dim)) ,
        #     transforms.ToTensor()]
        # )

        self.transform = T.Compose([
            T.Resize(256, interpolation=3),
            T.CenterCrop(img_dim),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

        self.curriculum_max = 1

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
    def __init__(self, images, targets, angles):
        self.image_list = images
        self.target_list = targets
        self.angle_list = angles
        assert (len(self.image_list) == len(self.target_list))
    
    def transform(self, x):
        x = np.resize(x, (3,64,64))
        x = x / 255.0

        return x

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, key):
        noise_img = self.image_list[key]
        clean_img = self.target_list[key]
        angle = self.angle_list[key]

        # noise_img = self.transform(noise_img)
        # clean_img = self.transform(clean_img)

        return [noise_img.astype(np.float32), clean_img.astype(np.float32), angle.astype(np.float32)]

'''
This section prepares the training, validation, and testing datasets for the pipeline
'''
def prepare_data_train(directory):
    print("Loading Train Data")

    train_path = directory + "train_honda.npz"
    val_path = directory + "val_honda.npz"

    train = np.load(train_path)
    val = np.load(val_path)

    return train['train_input_images'], train['train_target_angles'], val['val_input_images'], val['val_target_angles']

def prepare_data_test(directory, aug_method):
    print("Loading Test Data")

    test_set = f"test_{aug_method}.npz"
    test_path = os.path.join(directory, test_set)

    test = np.load(test_path)

    return test['test_input_images'], test['test_target_images'], test['test_target_angles'] 
