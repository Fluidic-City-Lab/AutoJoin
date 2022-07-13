# AutoJoin

This GitHub is the official PyTorch implementation of the technique: AutoJoin

## Data

All data used is publicly available; however, a GDrive link has been provided to all of the data necessary for training and evaluating on the Sully dataset. 

https://drive.google.com/file/d/1JYDZ0szXORLlMM8B17kwJg2OyyqqLiZm/view?usp=sharing

Should this link expire, then creating an issue or sending me an email directly will work and I'll fix it.

After downloading the data, unzip it, and place it into the data folder. That should be it for the data.


## Training

To start training, a command like:
```
python3 main.py --dataset sully --model nvidia --train_epochs 500 --seed 34222 --batch_size 128 --run_mode train
```
Will start the training using the Nvidia architecture, on the Sully dataset, for 500 epochs with a batch_size of 128. There are more settings that can be played around with as well.

If on the vit-sam branch, a command like:
```
python3 main.py --dataset sully --model vit --train_epochs 500 --seed 34222 --batch_size 128 --img_dim 32 --run_mode train 
```
Will start the training using a ViT architecture (currently, you can switch off of which ViT is used by commenting/uncommenting different architectures in models/joint_vit.py). The img_dim arg resizes the images to be square images of the input size. This should be set when using the ViT architecture as it typically operates on square images (the raw image dimensions = 3x66x200).

## Testing

After training, a command like:
```
python3 main.py --dataset sully --model nvidia --run_mode test_autojoin
```
Will result in testing on the model trained using the training command for the Nvidia architecture.

After training on the ViT architecture, a command like:
```
python3 main.py --dataset sully --model vit --img_dim 32 --run_mode test_autojoin
```
Will start the testing of the ViT model trained using the above command. The img_dim arg should match the img_dim arg in the training command.
