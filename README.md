# DSA

## Denoised Steering Angle
This GitHub is dedicated to the project for de-noising images that have been augmented in order to be able to get a better steering angle from them.

## Instructions to Setup the Project

1. Download the zip file for the dataset from this link: https://drive.google.com/file/d/1Xi0mA_mAj9Emp8DqUX45jyLqJDDQZGZQ/view. Within that zip file, there are 2 sets of data. One for Honda, and one for Waymo. The only data that is used for this project is the data associated with the Honda dataset. That correlates to 4 files within the zip folder. The 4 are: labelsHonda100k_train.csv, labelsHonda100k_val.csv, trainHonda100k, valHonda100k. For this project, the train files were used for training and validation, while the val files are used for testing. There are a total of 100k images and labels for training. Despite the name, there are 10,000 images and labels within the val files.
2. Clone this repository and create a folder in the project named "data", and put the 4 files listed above within that folder. Also create a "logs" folder and a "boxplot" folder and a "boxplob_csv" folder within that folder
3. Run the generate_dataset.py script. This will generate the training, validation, and testing datasets needed for this project within the data folder.
4. Download the unseen perturbations dataset from FIGURE SOMETHING OUT FOR THIS. The files in this dataset are all .npz files and should be placed within the data folder. These files will only be used during testing and never during training or validation.

Note that, during training, the data augmentations are applied to the original images at training time. This results in the overall training time being increased and the total number of instances being passed through the autoencoder to increase by a factor of 75 (meaning that if you 100,000 images in your training dataset, then you actually are passing a total of 7,500,000 images through the autoencoder). This is why only a subset of the data is being used as that is far too many examples for a 10 layer autoencoder to handle. A subset of the validation set is also used during training time for the exact same reason; however, the total number of testing images has been left alone (meaning that it is 10,000 images during testing time) for the sake being accurate.

5. Now, the project should be able to be run on your machine barring any issues from not having certain libraries installed.