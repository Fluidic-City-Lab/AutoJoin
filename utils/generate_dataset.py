import cv2
import numpy as np
import csv
import random
from PIL import Image
import os
from tqdm import tqdm

image_path_train='./data/waymo_train/'
label_path_train='./data/labels_waymo_train.csv'

image_path_test = './data/testWaymo/'
label_path_test = './data/labels_waymotest.csv'

def generate_project_datasets():
    csv_train = []

    with open(label_path_train, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        for row in csvreader:
            csv_train.append(row)

    random.shuffle(csv_train)
 
    csv_train = np.array(csv_train)

    imgs_train = []
    angles_train = []

    imgs_val = []
    angles_val = []
    
    for i in tqdm(range(len(csv_train))):    
        img = cv2.imread(image_path_train + csv_train[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the images to be RGB from BGR
        org_target_angle = csv_train[i][-1]
    
        if i % 10 == 0:
            imgs_val.append(img)
            angles_val.append(org_target_angle)
        else:    
            imgs_train.append(img)
            angles_train.append(org_target_angle)

    imgs_train = np.array(imgs_train)
    angles_train = np.array(angles_train)

    imgs_val = np.array(imgs_val)
    angles_val = np.array(angles_val)

    print(imgs_train.shape, angles_train.shape)
    print(imgs_val.shape, angles_val.shape)
        
    np.savez('./data/train_waymo.npz', train_input_images=imgs_train, train_target_angles=angles_train)
    np.savez('./data/val_waymo.npz', val_input_images=imgs_val, val_target_angles=angles_val)

    csv_test = []

    with open(label_path_test, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            csv_test.append(row)
    
    csv_test = np.array(csv_test)

    imgs_test = []
    angles_test = []

    for j in tqdm(range(len(csv_test))):
        img = cv2.imread(image_path_test + csv_test[j][0])   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the images to be RGB from BGR b/c cv2 defaults to BGR
        target_angle = csv_test[j][-1]

        imgs_test.append(img)
        angles_test.append(target_angle)
    
    imgs_test = np.array(imgs_test)
    imgs_test = np.moveaxis(imgs_test, -1, 1)
    angles_test = np.array(angles_test)

    print(imgs_test.shape, angles_test.shape)

    np.savez('./data/test_waymo.npz', test_input_images=imgs_test, test_target_images=imgs_test, test_target_angles=angles_test)


if __name__ == "__main__":
    generate_project_datasets()
    