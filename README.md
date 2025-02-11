# AutoJoin

This GitHub is the official PyTorch implementation of the robustness technique: AutoJoin

## Appendix
The [Appendix.pdf](https://github.com/tmvllrrl/AutoJoin/blob/main/Appendix.pdf) file contains 9 pages of appendix with following sections:
```
  I. DATASETS AND EXPERIMENT SETUP
  
  II. MAXIMUM AND MINIMUM INTENSITY
  
  III. FEEDBACK LOOP
  
  IV. FULL SET OF PERTURBATIONS NOT GUARANTEED AND NO RANDOM INTENSITIES
  
  V. PERTURBATION STUDY
  
  VI. TIMES FOR EXPERIMENTS
  
  VI. GRADIENT-BASED ADVERSARIAL TRANSFERABILITY
```
## Data

All data used is publicly available; however, if you would like the data in a ready for training/testing form, please feel free to reach out to me.

After downloading the data, unzip it, and place it into the data folder.


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
