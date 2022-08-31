import cv2
import time
import random
import os
import csv
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from models.joint_vit import DecoderViT, EncoderViT, RegressorViT


from utils.data_utils import TrainDriveDataset, TestDriveDataset
from utils.generate_augs import generate_augmentations_batch
from utils.error_metrics import mae, ma, rmse

from models.joint_resnet50 import EncoderRN50, DecoderRN50, RegressorRN50
from models.joint_nvidia import EncoderNvidia, DecoderNvidia, RegressorNvidia
from models.nvidia import Nvidia
from models.resnet50 import ResNet50
from vit_pytorch import ViT

from PIL import Image

class PipelineJoint:
    def __init__(self, args, mode="train", test_perturb="noise_3", test_num=0):
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

            self.lambda1 = self.args.lambda1
            self.lambda2 = self.args.lambda2

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

            self.train_dataset = TrainDriveDataset(args, x_train, y_train)
            self.val_dataset = TrainDriveDataset(args, x_val, y_val)
            
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
                self.encoder = EncoderRN50([3, 4, 6, 3], 3, 1).to(self.device)
                self.regressor = RegressorRN50([3, 4, 6, 3], 3, 1).to(self.device)
                self.decoder = DecoderRN50().to(self.device)

            elif self.args.model == "nvidia":                
                self.encoder = EncoderNvidia().to(self.device)
                self.regressor = RegressorNvidia().to(self.device)
                self.encoder_disc = EncoderNvidia().to(self.device)
                self.regressor_disc = RegressorNvidia().to(self.device)
                self.decoder = DecoderNvidia(self.args).to(self.device)
            
            elif self.args.model == "vit":
                self.encoder = EncoderViT(self.args).to(self.device)
                self.regressor = RegressorViT().to(self.device)
                self.decoder = DecoderViT(self.args).to(self.device)

            print(self.encoder)
            print(self.regressor)

            self.recon_loss = nn.MSELoss()
            self.regr_loss = nn.L1Loss()
            self.gan_loss = nn.BCELoss()
        
            # self.params = list(self.encoder.parameters()) + list(self.regressor.parameters()) + list(self.decoder.parameters())
            self.regr_params = list(self.encoder.parameters()) + list(self.regressor.parameters())
            self.disc_params = list(self.encoder_disc.parameters()) + list(self.regressor_disc.parameters())
            self.gen_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self.optimizer_regr = torch.optim.Adam(self.regr_params, lr=self.lr)
            self.optimizer_disc = torch.optim.Adam(self.disc_params, lr=self.lr)
            self.optimizer_gen = torch.optim.Adam(self.gen_params, lr=self.lr)
            
            self.real_label = 1
            self.fake_label = 0
            
            self.load_epoch = 0
            self.best_loss = float('inf')

            self.train_loss_collector = np.zeros(self.train_epochs)
            self.train_disc_loss_collector = np.zeros(self.train_epochs)
            self.train_gen_loss_collector = np.zeros(self.train_epochs)
            self.train_disc_clean_loss_collector = np.zeros(self.train_epochs)
            self.train_disc_gen_loss_collector = np.zeros(self.train_epochs)
            self.train_disc_regr_loss_collector = np.zeros(self.train_epochs)

            self.val_loss_collector = np.zeros(self.train_epochs)

            if self.args.load == "true":
                checkpoint = torch.load(f'{self.args.logs_dir}/{self.args.checkpoints_dir}/checkpoint.pt')

                self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
                self.regressor.load_state_dict(checkpoint["regressor_state_dict"])
                self.optimizer_disc.load_state_dict(checkpoint["optimizer_disc_state_dict"])
                self.optimizer_gen.load_state_dict(checkpoint["optimizer_gen_state_dict"])

                self.train_dataset.set_curr_max(checkpoint["cv"])
                self.val_dataset.set_curr_max(checkpoint["cv"])
                self.load_epoch = checkpoint["load_epoch"]
                self.best_loss = checkpoint["best_loss"]

                self.train_loss_collector = checkpoint["train_loss_collector"]
                self.train_disc_loss_collector = checkpoint["train_disc_loss_collector"]
                self.train_gen_loss_collector = checkpoint["train_gen_loss_collector"]
                self.train_disc_clean_loss_collector = checkpoint["train_disc_clean_loss_collector"]
                self.train_disc_gen_loss_collector = checkpoint["train_disc_gen_loss_collector"]
                self.train_disc_regr_loss_collector = checkpoint["train_disc_regr_loss_collector"]

                self.val_loss_collector = checkpoint["val_loss_collector"]

            self.train_dataset.set_curr_max(1)
            self.val_dataset.set_curr_max(1)

        else:
            self.test_perturb = test_perturb
            self.test_num = test_num

            # if test_num < 75: # This corresponds to the single perturbations where we just want to load the clean dataset
            #     self.test_inputs, self.test_targets, self.test_angles = prepare_data_test(self.args.data_dir, "clean")
            #     self.test_dataset = TestDriveDataset(self.test_inputs, self.test_targets, self.test_angles)
            # else: # This corresponds to the other 3 benchmarking datasets
            #     self.test_inputs, self.test_targets, self.test_angles = prepare_data_test(self.args.data_dir, aug_method)
            #     self.test_dataset = TestDriveDataset(self.test_inputs, self.test_targets, self.test_angles)

            # This is for loading the data from image files (like png/jpg/etc.)
            label_path_test = os.path.join(self.args.data_dir, f"{self.args.dataset}", "labels_test.csv")

            x_test = []
            y_test = []

            with open(label_path_test, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                
                for row in csvreader:
                    x_test.append(str(row[0][:-4]))
                    y_test.append(float(row[-1]))
        
            x_test = np.array(x_test)
            y_test = np.array(y_test)

            self.test_dataset = TestDriveDataset(self.args, x_test, y_test, self.test_perturb, self.test_num)

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

            train_batch_loss = 0.
            train_batch_disc_loss = 0.
            train_batch_gen_loss = 0.
            train_batch_disc_clean_loss = 0.
            train_batch_disc_gen_loss = 0.
            train_batch_disc_regr_loss = 0.

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
                
                noise_batch = noise_batch / 255.
                clean_batch = clean_batch / 255.

                clean_batch = np.moveaxis(clean_batch, -1, 1)
                noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                clean_batch = torch.tensor(clean_batch, dtype=torch.float32)   

                noise_batch, clean_batch, angle_batch = noise_batch.to(self.device), clean_batch.to(self.device), angle_batch.to(self.device)

                torch.autograd.set_detect_anomaly(True)

                # Discriminator Training 
                self.optimizer_disc.zero_grad()

                # Passing the clean images (or real images) through the discriminator
                gan_label = torch.full((angle_batch.shape[0], ), self.real_label, dtype=torch.float, device=self.device)
                disc_output_clean = torch.squeeze(torch.sigmoid(self.regressor_disc(self.encoder_disc(clean_batch))))
                err_disc_clean = self.gan_loss(disc_output_clean, gan_label)
                err_disc_clean.backward()

                # Generating images using the noisy images and then passing those through the
                # discriminator with fake labels
                gen_fake_images = self.decoder(self.encoder(noise_batch))

                if ep in [0, 20, 50, 200, 400]:
                    if bi == 0:
                        for i in range(len(gen_fake_images)):
                            print(gen_fake_images[i])
                            gen_fake_img = gen_fake_images[i] * 255.
                            img_i = Image.fromarray(np.uint8(np.moveaxis(gen_fake_img.cpu().detach().clone().numpy(), 0, -1))).convert('RGB')
                            img_i.save(f"./sample_images/generated_img_{ep}_{bi}_{i}.png")

                gan_label.fill_(self.fake_label)
                output_disc_fake = torch.squeeze(torch.sigmoid(self.regressor_disc(self.encoder_disc(gen_fake_images.detach()))))
                err_disc_gen = self.gan_loss(output_disc_fake, gan_label)
                err_disc_gen.backward()
                
                recon_loss = self.recon_loss(gen_fake_images, clean_batch)
                recon_loss.backward(retain_graph=True)

                # Generator Training
                self.optimizer_gen.zero_grad()

                gan_label.fill_(self.real_label)
                output_gen_fake = torch.squeeze(torch.sigmoid(self.regressor(self.encoder(gen_fake_images))))
                err_gen = self.gan_loss(output_gen_fake, gan_label)
                err_gen.backward()

                # Regressor Training
                sa_batch = torch.squeeze(self.regressor(self.encoder(noise_batch)))
                regr_loss = self.lambda2 * self.regr_loss(sa_batch, angle_batch)
                regr_loss.backward()

                self.optimizer_gen.step()
                self.optimizer_disc.step()
                self.optimizer_regr.step()
                
                # Logging the loss
                train_batch_loss += (err_disc_clean.item() + err_disc_gen.item() + regr_loss.item() + err_gen.item())
                train_batch_disc_loss += (err_disc_clean.item() + err_disc_gen.item() + regr_loss.item())
                train_batch_gen_loss += (err_gen.item())
                train_batch_disc_clean_loss += (err_disc_clean.item())
                train_batch_disc_gen_loss += (err_disc_gen.item())
                train_batch_disc_regr_loss += (regr_loss.item())

                preds_train.extend(sa_batch.cpu().detach().numpy())
            
            avg_train_batch_loss = round(train_batch_loss / len(self.train_dataloader), 3)
            avg_train_batch_disc_loss = round(train_batch_disc_loss / len(self.train_dataloader), 3)
            avg_train_batch_gen_loss = round(train_batch_gen_loss / len(self.train_dataloader), 3)
            avg_train_batch_disc_clean_loss = round(train_batch_disc_clean_loss / len(self.train_dataloader), 3)
            avg_train_batch_disc_gen_loss = round(train_batch_disc_gen_loss / len(self.train_dataloader), 3)
            avg_train_batch_disc_regr_loss = round(train_batch_disc_regr_loss / len(self.train_dataloader), 3)

            ma_train = ma(preds_train, gt_train)

            val_tuple = self.validate(self.val_dataloader)
            avg_val_batch_loss = val_tuple[0]
            ma_val = val_tuple[1]

            # Saving epoch train and val loss to their respective collectors
            self.train_loss_collector[ep] = avg_train_batch_loss
            self.train_disc_loss_collector[ep] = avg_train_batch_disc_loss
            self.train_gen_loss_collector[ep] = avg_train_batch_gen_loss
            self.train_disc_clean_loss_collector[ep] = avg_train_batch_disc_clean_loss
            self.train_disc_gen_loss_collector[ep] = avg_train_batch_disc_gen_loss
            self.train_disc_regr_loss_collector[ep] = avg_train_batch_disc_regr_loss

            self.val_loss_collector[ep] = avg_val_batch_loss

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

                torch.save(self.encoder.state_dict(), f'{self.args.logs_dir}/{self.args.trained_models_dir}/encoder.pth')
                torch.save(self.decoder.state_dict(), f'{self.args.logs_dir}/{self.args.trained_models_dir}/decoder.pth')
                torch.save(self.regressor.state_dict(), f'{self.args.logs_dir}/{self.args.trained_models_dir}/regressor.pth')

                torch.save({
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                    "regressor_state_dict": self.regressor.state_dict(),
                    "optimizer_disc_state_dict": self.optimizer_disc.state_dict(),
                    "optimizer_gen_state_dict": self.optimizer_gen.state_dict(),
                    "load_epoch": ep+1,
                    "best_loss": self.best_loss,
                    "cv": self.train_dataset.get_curr_max(),
                    "train_loss_collector": self.train_loss_collector,
                    "train_disc_loss_collector": self.train_disc_loss_collector,
                    "train_gen_loss_collector": self.train_gen_loss_collector,
                    "train_disc_clean_loss_collector": self.train_disc_clean_loss_collector,
                    "train_disc_gen_loss_collector": self.train_disc_gen_loss_collector,
                    "train_disc_regr_loss_collector": self.train_disc_regr_loss_collector,
                    "val_loss_collector": self.val_loss_collector,
                }, f'{self.args.logs_dir}/{self.args.checkpoints_dir}/checkpoint_best_loss.pt')

                if self.train_dataset.get_curr_max() < 0.99:
                    self.train_dataset.increase_curr_max()
                    self.val_dataset.increase_curr_max()
                    print(f"Increasing the curriculum value to {self.train_dataset.get_curr_max()}")
    
            torch.save({
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                    "optimizer_disc_state_dict": self.optimizer_disc.state_dict(),
                    "optimizer_gen_state_dict": self.optimizer_gen.state_dict(),
                    "load_epoch": ep+1,
                    "best_loss": self.best_loss,
                    "cv": self.train_dataset.get_curr_max(),
                    "train_loss_collector": self.train_loss_collector,
                    "train_disc_loss_collector": self.train_disc_loss_collector,
                    "train_gen_loss_collector": self.train_gen_loss_collector,
                    "train_disc_clean_loss_collector": self.train_disc_clean_loss_collector,
                    "train_disc_gen_loss_collector": self.train_disc_gen_loss_collector,
                    "train_disc_regr_loss_collector": self.train_disc_regr_loss_collector,
                    "val_loss_collector": self.val_loss_collector,
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
        ax.plot(np.array(self.train_disc_loss_collector))
        ax.plot(np.array(self.val_loss_collector))
        ax.set_xticks(xticks) 
        ax.legend(["Training (MSE)", "Validation (MSE)"])
        fig.savefig(f'{self.args.logs_dir}/training_graph_recon.png')

        fig, ax = plt.subplots(figsize=(16,5), dpi=200)
        xticks= np.arange(0,self.train_epochs,5)
        ax.set_ylabel("Avg. Loss") 
        ax.plot(np.array(self.train_gen_loss_collector))
        ax.plot(np.array(self.val_loss_collector))
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
        val_batch_reg_loss = 0

        gt_val = []
        preds_val = []

        with torch.no_grad():
            for bi, data in enumerate(val_dataloader):
                clean_batch, angle_batch = data
                gt_val.extend(angle_batch.numpy())

                angle_batch = torch.unsqueeze(angle_batch, 1)     

                clean_batch, angle_batch = clean_batch.to(self.device), angle_batch.to(self.device)

                # Passing it through model
                z = self.encoder(clean_batch)
                sa_batch = self.regressor(z)
                
                regr_loss = self.regr_loss(sa_batch, angle_batch)
                loss = (self.lambda2 * regr_loss)

                val_batch_loss += loss.item()

                preds_val.extend(sa_batch.cpu().detach().numpy())

        avg_val_batch_loss = round(val_batch_loss / len(val_dataloader), 3)
        ma_val = ma(preds_val, gt_val)

        return (avg_val_batch_loss, ma_val)

    def test_other(self):
        # other_method = Nvidia().to(self.device)
        # other_method.load_state_dict(torch.load('./saved_models/standard1.pth'))
        # other_method.eval()

        other_method = torch.load('./saved_models/shen1.pt')
        other_method.eval()

        print("Started Testing")

        gt_test = []
        preds_test = []

        with torch.no_grad():
            for batch, data in enumerate(tqdm(self.test_dataloader)):
                img_batch, angle_batch = data
                img_batch, angle_batch = img_batch.to(self.device), angle_batch.to(self.device)
    
                output, _ = other_method(img_batch)
                preds_test.append(np.squeeze(output.cpu().detach().clone().numpy()))

                gt_test.append(np.squeeze(angle_batch.cpu().detach().clone().numpy()))
        
        print("\nFinished Testing")

        preds_test = np.array(preds_test)
        gt_test = np.array(gt_test)

        results = []
        metric_list = [ma, rmse, mae]

        # Model 1
        calc_metrics(metric_list, results, "shen1", preds_test, gt_test)

        print("Writing Results to Logs")
        if self.test_num < 117:
            for i in range(len(results)):
                current = results[i]
                write_results(self.test_perturb, current[0], current[1])
        else:
            for i in range(len(results)):
                current = results[i]
                write_results(self.test_perturb, current[0], current[1], adversarial=True)
        
        print("Finished Writing Results to Logs\n") 

    def test_our_approach(self):
        if self.args.model == "resnet":
            encoder = EncoderRN50().to(self.device)
            regressor = RegressorRN50().to(self.device)
        elif self.args.model == "nvidia":
            encoder = EncoderNvidia().to(self.device)
            regressor = RegressorNvidia().to(self.device)
        elif self.args.model == "vit":
            encoder = EncoderViT(self.args).to(self.device)
            regressor = RegressorViT().to(self.device)

        encoder.load_state_dict(torch.load('./results/trained_models/encoder.pth'))
        encoder.eval()

        regressor.load_state_dict(torch.load('./results/trained_models/regressor.pth'))
        regressor.eval()

        print("Started Testing")

        preds_test = []
        gt_test = []

        with torch.no_grad():
            for batch, data in enumerate(tqdm(self.test_dataloader)):
                img_batch, angle_batch = data
                img_batch, angle_batch = img_batch.to(self.device), angle_batch.to(self.device)

                output = torch.squeeze(regressor(encoder(img_batch)))
            
                preds_test.append(output.cpu().detach().clone().numpy())
                gt_test.append(np.squeeze(angle_batch.cpu().detach().clone().numpy()))

                # fig, ax = plt.subplots(3,1, figsize=(8,8), dpi=100)
                # ax[0].imshow(Image.fromarray(np.uint8(np.moveaxis(clean_batch_np, 0, -1))).convert('RGB'))
                # ax[1].imshow(Image.fromarray(np.uint8(np.moveaxis(noise_batch_np, 0, -1))).convert('RGB'))
                # ax[2].imshow(Image.fromarray(np.uint8(np.moveaxis(recon_batch_np, 0, -1))).convert('RGB'))
                # fig.savefig(f"./sample_images/{aug_method}_{batch}")

        print("\nFinished Regression")

        preds_test = np.array(preds_test)

        gt_test = np.array(gt_test)

        results = []
        metric_list = [ma, rmse, mae]

        # Ours
        calc_metrics(metric_list, results, "ours1", preds_test, gt_test)
               
        print("Writing Results to Logs")
        if self.test_num < 117:
            for i in range(len(results)):
                current = results[i]
                write_results(self.test_perturb, current[0], current[1])
        else:
            for i in range(len(results)):
                current = results[i]
                write_results(self.test_perturb, current[0], current[1], adversarial=True)
        
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

def write_results(aug_method, aug_acc, name, adversarial=False):
    results = f"{aug_method},{aug_acc}\n"

    if adversarial == False:
        with open(f'./results/results_{name}.txt', 'a') as f:
            f.write(results)
    else:
        with open(f'./results/adversarial/results_{name}.txt', 'a') as f:
            f.write(results)

def calc_metrics(metric_list, results, name, aug_results, truths):
    for i in range(len(metric_list)):
        func = metric_list[i]

        aug_metric_results = func(aug_results, truths)

        results.append((aug_metric_results, f"{name}_{func.__name__}"))
