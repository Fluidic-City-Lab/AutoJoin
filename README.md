# AutoJoin

This GitHub is the official PyTorch implementation of the robustness technique: AutoJoin

## Data

All data used is publicly available; however, a GDrive link has been provided to all of the data necessary for training and evaluating on the SullyChen and A2D2 datasets. If you would like the data in a ready for training/testing form for Honda and/or Waymo, please feel free to email me, and I will provide the necessary links for those datasets as well.

SullyChen:
https://drive.google.com/file/d/1V9S2rWtUXY9dJgYQOnaNx39mjG5jtknG/view?usp=share_link

A2D2:
https://drive.google.com/file/d/1XlZjB208f_8a97jxY5Qv_ZBl9ipPKTZv/view?usp=share_link

Should these links expire, then create an issue or send an email, and I will fix it.

After downloading the data, unzip it, and place it into the data folder. That's all that is necessary for the data.


## Training

To start training, a command like:
```
python3 main.py --dataset sully --model nvidia --train_epochs 500 --seed 34222 --batch_size 128 --run_mode train
```
Will start the training using the Nvidia architecture, on the Sully dataset, for 500 epochs with a batch_size of 128. There are more settings that can be played around with as well.

The above command can be swapped out to train on the ResNet50 architecture by swapping out --model nvidia for --model resnet. Only the RN50 architecture was evaluated for this paper, although the other ResNet architectures could be swapped in with additional code.

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
