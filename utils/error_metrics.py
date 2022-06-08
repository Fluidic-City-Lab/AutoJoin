import numpy as np
import torch.nn as nn
import torch

def ma(predictions, targets, check=False):
        predictions, targets = np.squeeze(np.asarray(predictions)), np.asarray(targets) # Making sure the two are numpy arrays

        ground_truths = targets * 15.0
        predictions = predictions * 15.0
        error = np.abs(ground_truths - predictions)

        total = error.shape[0]
        count_1 = np.sum(np.asarray([ 1.0 if er <=1.5 else 0.0 for er in error]))
        count_3 = np.sum(np.asarray([ 1.0 if er <=3.0 else 0.0 for er in error]))
        count_7 = np.sum(np.asarray([ 1.0 if er <=7.5 else 0.0 for er in error]))
        count_15 = np.sum(np.asarray([ 1.0 if er <=15.0 else 0.0 for er in error]))
        count_30 = np.sum(np.asarray([ 1.0 if er <=30.0 else 0.0 for er in error]))
        count_75 = np.sum(np.asarray([ 1.0 if er <=75.0 else 0.0 for er in error]))

        acc_1 = 100*(count_1/total)
        acc_2 = 100*(count_3/total)
        acc_3 = 100*(count_7/total)
        acc_4 = 100*(count_15/total)
        acc_5 = 100*(count_30/total)
        acc_6 = 100*(count_75/total)
       
        mean_acc = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6)/6
        mean_acc = round(mean_acc, 4) 

        # The point of this is to see the actual number breakdown as the difference between Mean Acurracys isn't linear
        # if check == True:
        #     with open('./logs/results_count.txt', 'a') as f:
        #         f.write(f"Count 1.5: {count_1}/{total}\n")
        #         f.write(f"Count 3.0: {count_3}/{total}\n")
        #         f.write(f"Count 7.5: {count_7}/{total}\n")
        #         f.write(f"Count 15.0: {count_15}/{total}\n")
        #         f.write(f"Count 30.0: {count_30}/{total}\n")
        #         f.write(f"Count 75.0: {count_75}/{total}\n\n")

        return mean_acc 


def mape(predictions, targets):
    return np.mean(np.abs((targets - predictions)/targets))*100

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

class PSNR(nn.Module):
    def __init__(self) -> None:
        super(PSNR, self).__init__()

    def forward(self, x, y):
        mse = torch.mean((x - y)**2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
