import cv2
import time
import random
import os
import csv
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from models.joint_vit import DecoderViT, EncoderViT, RegressorViT


from utils.data_utils import DriveDataset, DriveDatasetImage, TestDriveDataset, prepare_data_test
from utils.generate_augs import generate_augmentations_batch, generate_augmentations_test
from utils.error_metrics import mae, ma, rmse

from models.joint_resnet50 import RN50Encoder, RN50Head, RN50Decoder
from models.joint_nvidia import EncoderNvidia, DecoderNvidia, RegressorNvidia
from models.nvidia import Nvidia
from models.resnet50 import ResNet50
from vit_pytorch import ViT

from PIL import Image

class PipelineJoint:
    def __init__(self, args, mode="train", aug_method="noise_3", aug_num=0):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: {}".format(self.device))

        print(f"Seed: {self.args.seed}")
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        results_dir = os.path.join(self.args.logs_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        
        checkpoints_dir = os.path.join(self.args.logs_dir, self.args.checkpoints_dir)
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        
        trained_models_dir = os.path.join(self.args.logs_dir, self.args.trained_models_dir)
        if not os.path.exists(trained_models_dir):
            os.mkdir(trained_models_dir)


        if mode == "train":
            self.batch_size = self.args.batch_size
            self.lr = self.args.lr
            self.train_epochs = self.args.train_epochs

            print(f"HYPERPARAMETERS\n------------------------")
            print(f"Train batch_size: {self.batch_size}")
            print(f"Learning rate: {self.lr}")
            print(f"Training Epochs: {self.train_epochs}\n")

            
            # This is for loading the data from image files (like png/jpg/etc.)
            label_path_train = os.path.join(self.args.data_dir, f"{self.args.dataset}", "labels_train.csv")

            x = []
            y = []

            with open(label_path_train, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                
                for row in csvreader:
                    x.append(str(row[0][:-4]))
                    y.append(float(row[-1]))
        
            x = np.array(x)
            y = np.array(y)

            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

            self.train_dataset = DriveDatasetImage(args, x_train, y_train)
            self.val_dataset = DriveDatasetImage(args, x_val, y_val)
            
            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                collate_fn=None,
                                                num_workers=8,
                                                prefetch_factor=8)

            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                collate_fn=None,
                                                num_workers=8,
                                                prefetch_factor=8)
            
            

            if self.args.model == "resnet":
                self.encoder = RN50Encoder([3, 4, 6, 3], 3, 1).to(self.device)
                self.regressor = RN50Head([3, 4, 6, 3], 3, 1).to(self.device)
                self.decoder = RN50Decoder().to(self.device)

            elif self.args.model == "nvidia":                
                self.encoder = EncoderNvidia().to(self.device)
                self.regressor = RegressorNvidia().to(self.device)
                self.decoder = DecoderNvidia().to(self.device)
            
            elif self.args.model == "vit":
                self.encoder = EncoderViT(self.args).to(self.device)
                self.regressor = RegressorViT().to(self.device)
                self.decoder = DecoderViT(self.args).to(self.device)

            print(self.encoder)
            print(self.regressor)

            self.recon_loss = nn.MSELoss()
            self.regr_loss = nn.L1Loss()
            self.emb_loss = nn.MSELoss()                                  
        
            self.params = list(self.encoder.parameters()) + list(self.regressor.parameters()) + list(self.decoder.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
            self.load_epoch = 0
            self.best_loss = float('inf')

            self.train_loss_collector = np.zeros(self.train_epochs)
            self.train_recon_loss_collector = np.zeros(self.train_epochs)
            self.train_reg_loss_collector = np.zeros(self.train_epochs)

            self.val_loss_collector = np.zeros(self.train_epochs)
            self.val_recon_loss_collector = np.zeros(self.train_epochs)
            self.val_reg_loss_collector = np.zeros(self.train_epochs)

            if self.args.load == "true":
                checkpoint = torch.load(f'{self.args.logs_dir}/{self.args.checkpoints_dir}/checkpoint.pt')

                self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
                self.regressor.load_state_dict(checkpoint["regressor_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                self.train_dataset.set_curr_max(checkpoint["cv"])
                self.val_dataset.set_curr_max(checkpoint["cv"])
                self.load_epoch = checkpoint["load_epoch"]
                self.best_loss = checkpoint["best_loss"]

                self.train_loss_collector = checkpoint["train_loss_collector"]
                self.train_recon_loss_collector = checkpoint["train_recon_loss_collector"]
                self.train_reg_loss_collector = checkpoint["train_reg_loss_collector"]

                self.val_loss_collector = checkpoint["val_loss_collector"]
                self.val_recon_loss_collector = checkpoint["val_recon_loss_collector"]
                self.val_reg_loss_collector = checkpoint["val_reg_loss_collector"]

        else:
            if aug_num < 75: # This corresponds to the single perturbations where we just want to load the clean dataset
                self.test_inputs, self.test_targets, self.test_angles = prepare_data_test(self.args.data_dir, "clean")
                self.test_dataset = TestDriveDataset(self.test_inputs, self.test_targets, self.test_angles)
            else: # This corresponds to the other 3 benchmarking datasets
                self.test_inputs, self.test_targets, self.test_angles = prepare_data_test(self.args.data_dir, aug_method)
                self.test_dataset = TestDriveDataset(self.test_inputs, self.test_targets, self.test_angles)

            self.test_dataloader = DataLoader(dataset=self.test_dataset, 
                                                batch_size=1, 
                                                shuffle=False)

    # Function that trains the model while validating the model at the same time
    def train(self):
        print("\nStarted Training\n")

        for ep in range(self.load_epoch, self.train_epochs):
            self.encoder.train()
            self.decoder.train()
            self.regressor.train()

            if ep==0:
                with open(f'{self.args.logs_dir}/encoder_init_weights.txt', 'w') as f:
                  for param in self.encoder.parameters():
                    f.write("%s\n" % param.data)
                
                with open(f'{self.args.logs_dir}/decoder_init_weights.txt', 'w') as f:
                  for param in self.decoder.parameters():
                    f.write("%s\n" % param.data)
                
                with open(f'{self.args.logs_dir}/regressor_init_weights.txt', 'w') as f:
                  for param in self.regressor.parameters():
                    f.write("%s\n" % param.data)

            start_time = time.time()

            train_batch_loss = 0
            train_batch_recon_loss = 0
            train_batch_reg_loss = 0

            # Below two arrays are used for calculing Mean Accuracy during training
            gt_train = [] 
            preds_train = []

            for bi, data in enumerate(tqdm(self.train_dataloader)):
                clean_batch, angle_batch = data
                gt_train.extend(angle_batch.numpy())

                clean_batch = clean_batch.numpy()
                clean_batch = clean_batch * 255.
                clean_batch = np.uint8(clean_batch) # Images need to be uint8 for cv2 when doing the augmentations
                clean_batch = np.moveaxis(clean_batch, 1, -1)
                
                noise_batch = generate_augmentations_batch(clean_batch, self.train_dataset.get_curr_max())

                clean_batch = np.moveaxis(clean_batch, -1, 1)

                noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                clean_batch = torch.tensor(clean_batch, dtype=torch.float32)   
                angle_batch = torch.unsqueeze(angle_batch, 1)     

                noise_batch = noise_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                angle_batch = angle_batch.to(self.device)

                # Passing it through model
                z = self.encoder(clean_batch)

                recon_batch = self.decoder(z)
                sa_batch = self.regressor(z)

                clean_batch = clean_batch / 255.

                recon_loss = self.recon_loss(recon_batch, clean_batch)
                regr_loss = self.regr_loss(sa_batch, angle_batch)

                loss = (recon_loss) + (10 * regr_loss) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_batch_loss += loss.item()
                train_batch_recon_loss += (1 * recon_loss.item())
                train_batch_reg_loss += (10 * regr_loss.item())

                preds_train.extend(sa_batch.cpu().detach().numpy())
            
            avg_train_batch_loss = round(train_batch_loss / len(self.train_dataloader), 3)
            avg_train_batch_recon_loss = round(train_batch_recon_loss / len(self.train_dataloader), 3)
            avg_train_batch_emb_loss = round(train_batch_emb_loss / len(self.train_dataloader), 3)
            avg_train_batch_reg_loss = round(train_batch_reg_loss / len(self.train_dataloader), 3)

            ma_train = ma(preds_train, gt_train)

            val_tuple = self.validate(self.val_dataloader)
            avg_val_batch_loss = val_tuple[0]
            ma_val = val_tuple[4]

            # Saving epoch train and val loss to their respective collectors
            self.train_loss_collector[ep] = avg_train_batch_loss
            self.train_recon_loss_collector[ep] = avg_train_batch_recon_loss
            self.train_emb_loss_collector[ep] = avg_train_batch_emb_loss
            self.train_reg_loss_collector[ep] = avg_train_batch_reg_loss

            self.val_loss_collector[ep] = avg_val_batch_loss
            self.val_recon_loss_collector[ep] = val_tuple[1]
            self.val_emb_loss_collector[ep] = val_tuple[2]
            self.val_reg_loss_collector[ep] = val_tuple[3]

            end_time = time.time()
            epoch_time = end_time - start_time
 
            print(f"Epoch: {ep+1}\t ATL: {avg_train_batch_loss:.3f}\t TMA: {ma_train:.2f}%\t AVL: {avg_val_batch_loss:.3f}\t VMA: {ma_val:.2f}%\t Time: {epoch_time:.3f}\t CV: {self.train_dataset.get_curr_max()}")

            with open(f'{self.args.logs_dir}/train_log_pp.txt', 'a') as train_log_pp:
                train_log_pp.write(f"Epoch: {ep+1}\t ATL: {avg_train_batch_loss:.3f}\t TMA: {ma_train:.2f}\t AVL: {avg_val_batch_loss:.3f}\t VMA: {ma_val:.2f}%\t Time: {epoch_time:.3f} CV: {self.train_dataset.get_curr_max()}\n")
            
            with open(f'{self.args.logs_dir}/train_log.txt', 'a') as train_log:
                train_log.write(f"{ep+1},{avg_train_batch_loss:.3f},{ma_train:.2f},{avg_val_batch_loss:.3f},{ma_val:.2f},{epoch_time:.3f},{self.train_dataset.get_curr_max()}\n")
            
            # Only saving the model if the average validation loss is better after another epoch
            if avg_val_batch_loss < self.best_loss:
                self.best_loss = avg_val_batch_loss

                print("Saving new model")
                with open(f'{self.args.logs_dir}/train_log_pp.txt', 'a') as train_log_pp:
                    train_log_pp.write("Saving new model\n")

                torch.save(self.encoder, f'{self.args.logs_dir}/{self.args.trained_models_dir}/encoder.pt')
                torch.save(self.decoder, f'{self.args.logs_dir}/{self.args.trained_models_dir}/decoder.pt')
                torch.save(self.regressor, f'{self.args.logs_dir}/{self.args.trained_models_dir}/regressor.pt')

                torch.save({
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                    "regressor_state_dict": self.regressor.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "load_epoch": ep+1,
                    "best_loss": self.best_loss,
                    "cv": self.train_dataset.get_curr_max(),
                    "train_loss_collector": self.train_loss_collector,
                    "train_recon_loss_collector": self.train_recon_loss_collector,
                    "train_emb_loss_collector": self.train_emb_loss_collector,
                    "train_reg_loss_collector": self.train_reg_loss_collector,
                    "val_loss_collector": self.val_loss_collector,
                    "val_recon_loss_collector": self.val_recon_loss_collector,
                    "val_emb_loss_collector": self.val_emb_loss_collector,
                    "val_reg_loss_collector": self.val_reg_loss_collector
                }, f'{self.args.logs_dir}/{self.args.checkpoints_dir}/checkpoint_best_loss.pt')

                if self.train_dataset.get_curr_max() < 0.99:
                    self.train_dataset.increase_curr_max()
                    self.val_dataset.increase_curr_max()
                    print(f"Increasing the curriculum value to {self.train_dataset.get_curr_max()}")
    
            torch.save({
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                    "regressor_state_dict": self.regressor.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "load_epoch": ep+1,
                    "best_loss": self.best_loss,
                    "cv": self.train_dataset.get_curr_max(),
                    "train_loss_collector": self.train_loss_collector,
                    "train_recon_loss_collector": self.train_recon_loss_collector,
                    "train_emb_loss_collector": self.train_emb_loss_collector,
                    "train_reg_loss_collector": self.train_reg_loss_collector,
                    "val_loss_collector": self.val_loss_collector,
                    "val_recon_loss_collector": self.val_recon_loss_collector,
                    "val_emb_loss_collector": self.val_emb_loss_collector,
                    "val_reg_loss_collector": self.val_reg_loss_collector
                }, f'{self.args.logs_dir}/{self.args.checkpoints_dir}/checkpoint.pt')
 
        print("\nFinished Training!\n")

        # Plotting the decrease in training and validation loss and then saving that as a figure
        fig, ax = plt.subplots(figsize=(16,5), dpi=200)
        xticks= np.arange(0,self.train_epochs,5)
        ax.set_ylabel("Avg. Loss") 
        ax.plot(np.array(self.train_loss_collector))
        ax.plot(np.array(self.val_loss_collector))
        ax.set_xticks(xticks) 
        ax.legend(["Training (MSE)", "Validation (MSE)"])
        fig.savefig(f'{self.args.logs_dir}/training_graph.png')

        fig, ax = plt.subplots(figsize=(16,5), dpi=200)
        xticks= np.arange(0,self.train_epochs,5)
        ax.set_ylabel("Avg. Loss") 
        ax.plot(np.array(self.train_recon_loss_collector))
        ax.plot(np.array(self.val_recon_loss_collector))
        ax.set_xticks(xticks) 
        ax.legend(["Training (MSE)", "Validation (MSE)"])
        fig.savefig(f'{self.args.logs_dir}/training_graph_recon.png')

        fig, ax = plt.subplots(figsize=(16,5), dpi=200)
        xticks= np.arange(0,self.train_epochs,5)
        ax.set_ylabel("Avg. Loss") 
        ax.plot(np.array(self.train_emb_loss_collector))
        ax.plot(np.array(self.val_emb_loss_collector))
        ax.set_xticks(xticks) 
        ax.legend(["Training (MSE)", "Validation (MSE)"])
        fig.savefig(f'{self.args.logs_dir}/training_graph_emb.png')

        fig, ax = plt.subplots(figsize=(16,5), dpi=200)
        xticks= np.arange(0,self.train_epochs,5)
        ax.set_ylabel("Avg. Loss") 
        ax.plot(np.array(self.train_reg_loss_collector))
        ax.plot(np.array(self.val_reg_loss_collector))
        ax.set_xticks(xticks) 
        ax.legend(["Training (MSE)", "Validation (MSE)"])
        fig.savefig(f'{self.args.logs_dir}/training_graph_reg.png')

    
    # Function that validates the current model on the validation set of images
    def validate(self, val_dataloader):
        self.encoder.eval()
        self.decoder.eval()
        self.regressor.eval()

        val_batch_loss = 0
        val_batch_recon_loss = 0
        val_batch_emb_loss = 0
        val_batch_reg_loss = 0

        gt_val = []
        preds_val = []

        with torch.no_grad():
            for bi, data in enumerate(val_dataloader):
                clean_batch, angle_batch = data
                gt_val.extend(angle_batch.numpy())

                clean_batch = clean_batch.numpy()
                clean_batch = clean_batch * 255.
                clean_batch = np.uint8(clean_batch) # Images need to be uint8 for cv2 when doing the augmentations
                clean_batch = np.moveaxis(clean_batch, 1, -1)

                # Initializing the batches for the first images. This is mainly just for noise_batch
                # noise_batch = generate_augmentations_batch(clean_batch, curriculum_max)

                clean_batch = np.moveaxis(clean_batch, -1, 1)

                # noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                clean_batch = torch.tensor(clean_batch, dtype=torch.float32)   
                angle_batch = torch.unsqueeze(angle_batch, 1)     

                # noise_batch = noise_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                angle_batch = angle_batch.to(self.device)

                # Passing it through model
                z = self.encoder(clean_batch)

                recon_batch = self.decoder(z)
                sa_batch = self.regressor(z)
                # sa_batch = z
                
                # sa_recon_batch = self.regressor(self.encoder(recon_batch*255.0))

                clean_batch = clean_batch / 255.

                recon_loss = self.recon_loss(recon_batch, clean_batch)
                regr_loss = self.regr_loss(sa_batch, angle_batch)
                # recon_regr_loss = self.regr_loss(sa_recon_batch, angle_batch)

                loss = (recon_loss)  + (1 * regr_loss) # + (1 * recon_regr_loss)
                # loss = regr_loss

                val_batch_loss += loss.item()
                val_batch_recon_loss += recon_loss.item()
                val_batch_reg_loss += (1 * regr_loss.item())

                preds_val.extend(sa_batch.cpu().detach().numpy())

        
        avg_val_batch_loss = round(val_batch_loss / len(val_dataloader), 3)
        avg_val_batch_recon_loss = round(val_batch_recon_loss / len(val_dataloader), 3)
        avg_val_batch_emb_loss = round(val_batch_emb_loss / len(val_dataloader), 3)
        avg_val_batch_reg_loss = round(val_batch_reg_loss / len(val_dataloader), 3)

        ma_val = ma(preds_val, gt_val)

        return (avg_val_batch_loss, avg_val_batch_recon_loss, avg_val_batch_emb_loss, avg_val_batch_reg_loss, ma_val)

    def test_single_pair(self, aug_method, aug_num):
        aug_method_name, aug_method_level = get_aug_method(aug_method)

        # robust1 = torch.load('./saved_models/standard1.pt')
        # robust1.eval()

        # robust2 = torch.load('./saved_models/standard2.pt')
        # robust2.eval()

        robust1 = Nvidia().to(self.device)
        robust1.load_state_dict(torch.load('./saved_models/standard1.pth'))
        robust1.eval()

        robust2 = Nvidia().to(self.device)
        robust2.load_state_dict(torch.load('./saved_models/standard1.pth'))
        robust2.eval()

        # robust1 = ViT(
        #             image_size=64,
        #             patch_size=16,
        #             num_classes=1,
        #             dim=128,
        #             depth=64,
        #             heads=16,
        #             mlp_dim=256,
        #             dropout=0.1,
        #             emb_dropout=0.1
        #         ).to(self.device)
        # robust1.load_state_dict(torch.load('./saved_models/standard1.pth'))
        # robust1.eval()

        # robust2 = ViT(
        #             image_size=64,
        #             patch_size=16,
        #             num_classes=1,
        #             dim=128,
        #             depth=64,
        #             heads=16,
        #             mlp_dim=256,
        #             dropout=0.1,
        #             emb_dropout=0.1
        #         ).to(self.device)
        # robust2.load_state_dict(torch.load('./saved_models/standard1.pth'))
        # robust2.eval()


        print("Started Regression")

        aug_steering_angles_robust1 = []
        clean_steering_angles_robust1 = []

        aug_steering_angles_robust2 = []
        clean_steering_angles_robust2 = []

        ground_truth = []

        with torch.no_grad():
            for batch, data in enumerate(tqdm(self.test_dataloader)):
                noise_batch, clean_batch, angle_batch = data
    
                # Only do this if we are looking at single perturbations
                if aug_num < 75:
                    # We've loaded the clean dataset for the single perturbations so that we can augment them at test time.
                    # This means that we can just augment the noise_batch and then pass it through the AE
                    noise_batch = np.squeeze(noise_batch.numpy())
                    noise_batch = np.moveaxis(noise_batch, 0, -1) # Dimensions are 66, 200, 3 after this
                    noise_batch = np.uint8(noise_batch)
                    noise_batch = generate_augmentations_test(noise_batch, aug_method_name, aug_method_level)
                    noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                    noise_batch = torch.unsqueeze(noise_batch, dim=0)

                noise_batch, clean_batch, angle_batch = noise_batch.to(self.device), clean_batch.to(self.device), angle_batch.to(self.device)

                # First Robust Regressor
                sa_preds_aug_robust1, sa_preds_clean_robust1 = feed_through_regressor(robust1, noise_batch, clean_batch)

                aug_steering_angles_robust1.append(sa_preds_aug_robust1)
                clean_steering_angles_robust1.append(sa_preds_clean_robust1)

                # Second Robust Regressor
                sa_preds_aug_robust2, sa_preds_clean_robust2 = feed_through_regressor(robust2, noise_batch, clean_batch)

                aug_steering_angles_robust2.append(sa_preds_aug_robust2)
                clean_steering_angles_robust2.append(sa_preds_clean_robust2)

                # Ground Truth
                ground_truth_angle = np.squeeze(angle_batch.cpu().detach().clone().numpy())
                ground_truth.append(ground_truth_angle)
        
        print("\nFinished Regression")

        aug_sa_robust1 = np.array(aug_steering_angles_robust1)
        clean_sa_robust1 = np.array(clean_steering_angles_robust1)

        aug_sa_robust2 = np.array(aug_steering_angles_robust2)
        clean_sa_robust2 = np.array(clean_steering_angles_robust2)

        ground_truth = np.array(ground_truth)

        results = []
        metric_list = [ma, rmse, mae]

        # Model 1
        calc_metrics(metric_list, results, "standard1", aug_sa_robust1, clean_sa_robust1, ground_truth)

        # Model 2
        calc_metrics(metric_list, results, "standard2", aug_sa_robust2, clean_sa_robust2, ground_truth)

        print("Writing Results to Logs")
        if aug_num < 117:
            for i in range(len(results)):
                current = results[i]
                write_results(aug_method, current[0], current[1], current[2])
        else:
            for i in range(len(results)):
                current = results[i]
                write_results(aug_method, current[0], current[1], current[2], adversarial=True)
        
        print("Finished Writing Results to Logs\n") 

    def test_our_approach(self, aug_method, aug_num):
        aug_method_name, aug_method_level = get_aug_method(aug_method)

        encoder = torch.load('./results/trained_models/encoder.pt')
        encoder.eval()

        regressor = torch.load('./results/trained_models/regressor.pt')
        regressor.eval()

        decoder = torch.load('./results/trained_models/decoder.pt')
        decoder.eval()

        print("Started Regression")

        aug_steering_angles_ours = []
        clean_steering_angles_ours = []

        ground_truth = []

        with torch.no_grad():
            for batch, data in enumerate(tqdm(self.test_dataloader)):
                noise_batch, clean_batch, angle_batch = data
    
                # Only do this if we are looking at single perturbations
                if aug_num < 75:
                    # We've loaded the clean dataset for the single perturbations so that we can augment them at test time.
                    # This means that we can just augment the noise_batch and then pass it through the AE
                    noise_batch = np.squeeze(noise_batch.numpy())
                    noise_batch = np.moveaxis(noise_batch, 0, -1) # Dimensions are 66, 200, 3 after this
                    noise_batch = np.uint8(noise_batch)
                    noise_batch = generate_augmentations_test(noise_batch, aug_method_name, aug_method_level)
                    noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                    noise_batch = torch.unsqueeze(noise_batch, dim=0)

                noise_batch, clean_batch, angle_batch = noise_batch.to(self.device), clean_batch.to(self.device), angle_batch.to(self.device)

                # Our Regressor
                aug_sa_ours = regressor(encoder(noise_batch))
                aug_sa_ours = torch.squeeze(aug_sa_ours)

                clean_sa_ours = regressor(encoder(clean_batch))
                clean_sa_ours = torch.squeeze(clean_sa_ours)

                recon_sa_ours = decoder(encoder(noise_batch))
                recon_batch = torch.squeeze(recon_sa_ours) * 255.

                aug_steering_angles_ours.append(aug_sa_ours.cpu().detach().clone().numpy())
                clean_steering_angles_ours.append(clean_sa_ours.cpu().detach().clone().numpy())
                
                # Ground Truth
                ground_truth_angle = np.squeeze(angle_batch.cpu().detach().clone().numpy())
                ground_truth.append(ground_truth_angle)

                # Saving sample images to test the efficacy of the autoencoder 
                # There's potential that the regressor is just that powerful so doing a sanity check to make sure that the reconstructed images are visually what they should look like 
                # if batch % 80 == 0:
                #     clean_batch_np = np.squeeze(clean_batch.cpu().detach().clone().numpy())
                #     noise_batch_np = np.squeeze(noise_batch.cpu().detach().clone().numpy())
                #     recon_batch_np = np.squeeze(recon_batch.cpu().detach().clone().numpy())

                #     noise_img = Image.fromarray(np.uint8(np.moveaxis(noise_batch_np, 0, -1))).convert('RGB')
                #     noise_img.save(f'./sample_images/{aug_method}_{batch}_noise.png')

                #     clean_img = Image.fromarray(np.uint8(np.moveaxis(clean_batch_np, 0, -1))).convert('RGB')
                #     clean_img.save(f'./sample_images/{aug_method}_{batch}_clean.png')
                
                #     recon_img = Image.fromarray(np.uint8(np.moveaxis(recon_batch_np, 0, -1))).convert('RGB')
                #     recon_img.save(f'./sample_images/{aug_method}_{batch}_recon.png')

                    # fig, ax = plt.subplots(3,1, figsize=(8,8), dpi=100)
                    # ax[0].imshow(Image.fromarray(np.uint8(np.moveaxis(clean_batch_np, 0, -1))).convert('RGB'))
                    # ax[1].imshow(Image.fromarray(np.uint8(np.moveaxis(noise_batch_np, 0, -1))).convert('RGB'))
                    # ax[2].imshow(Image.fromarray(np.uint8(np.moveaxis(recon_batch_np, 0, -1))).convert('RGB'))
                    # fig.savefig(f"./sample_images/{aug_method}_{batch}")

        print("\nFinished Regression")

        aug_sa_ours = np.array(aug_steering_angles_ours)
        clean_sa_ours = np.array(clean_steering_angles_ours)

        ground_truth = np.array(ground_truth)

        results = []
        metric_list = [ma, rmse, mae]

        # Ours
        calc_metrics(metric_list, results, "ours1", aug_sa_ours, clean_sa_ours, ground_truth)
               
        print("Writing Results to Logs")
        if aug_num < 117:
            for i in range(len(results)):
                current = results[i]
                write_results(aug_method, current[0], current[1], current[2])
        else:
            for i in range(len(results)):
                current = results[i]
                write_results(aug_method, current[0], current[1], current[2], adversarial=True)
        
        print("Finished Writing Results to Logs\n") 

# HELPER FUNCTIONS

def get_aug_method(aug_method):
    word_array = aug_method.split('_')
    aug_method = ""

    for x in range(len(word_array)):
        if x != len(word_array) - 1:
            aug_method += word_array[x] + " "
        else:
            aug_method += word_array[x]
    
    aug_method_test = aug_method[:-2]
    aug_level_test = aug_method[-1]

    return aug_method_test, aug_level_test

def write_results(aug_method, aug_acc, clean_acc, name, adversarial=False):
    results = f"{aug_method},{aug_acc},{clean_acc}\n"

    if adversarial == False:
        with open(f'./results/results_{name}.txt', 'a') as f:
            f.write(results)
    else:
        with open(f'./results/adversarial/results_{name}.txt', 'a') as f:
            f.write(results)

def calc_metrics(metric_list, results, name, aug_results, clean_results, truths):
    for i in range(len(metric_list)):
        func = metric_list[i]

        aug_metric_results = func(aug_results, truths)
        clean_metric_results = func(clean_results, truths) 

        results.append((aug_metric_results, clean_metric_results, f"{name}_{func.__name__}"))

def convert_to_bgr(rgb_batch):
    rgb_batch_cv2 = np.squeeze(rgb_batch)
    rgb_batch_cv2 = np.moveaxis(rgb_batch_cv2, 0, -1)
    bgr_batch = cv2.cvtColor(rgb_batch_cv2, cv2.COLOR_RGB2BGR)
    bgr_batch = np.moveaxis(bgr_batch, -1, 0)
    bgr_batch = torch.tensor(bgr_batch, dtype=torch.float32)
    bgr_batch = torch.unsqueeze(bgr_batch, dim=0)

    return bgr_batch

def feed_through_regressor(model, noise_batch, clean_batch):
    preds_aug = model(noise_batch) # Steering Angle predictions on the augmented images (no denoising)
    preds_clean = model(clean_batch)

    preds_aug = np.squeeze(preds_aug.cpu().detach().clone().numpy())
    preds_clean = np.squeeze(preds_clean.cpu().detach().clone().numpy())

    return preds_aug, preds_clean