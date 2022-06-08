from tqdm import tqdm
import argparse

from pipeline import PipelineJoint
from utils.stats_utils_joint import calc_comparison_baseline, calc_avg_categories, generate_average_file

from models.joint_nvidia import EncoderNvidia, DecoderNvidia, RegressorNvidia
from models.nvidia_advbn import HeadNvidia, FeatureXNvidia, NvidiaAdvBN
from models.nvidia import Nvidia
from models.resnet50 import ResNet50


####### HELPER METHODS #######

def get_aug_list(aug_list_path):
    aug_list = []

    with open(aug_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            aug_list.append(line) 
    
    return aug_list

def generate_average_files():
    '''
        Function that generates all of the average files for all data techniques, across 
        the 3 metrics, for a particular dataset and model architecture.
        The results files for each technique should be stored within the logs folder.
    '''

    aug_techs = ["ours", "shen", "standard", "advbn", "maxup", "augmax", "augmix"]
    aug_techs = ["ours"]
    # aug_techs = ["ours", "baseline", "augmix", ""]
    metrics = ["ma", "rmse", "mae"]

    for i in range(len(aug_techs)):
        for j in range(len(metrics)):
            generate_average_file(f'./logs/results_{aug_techs[i]}1_{metrics[j]}.txt', 
                f'./logs/results_{aug_techs[i]}2_{metrics[j]}.txt', 
                f"{aug_techs[i]}_{metrics[j]}")

def calc_comparisons_baseline():
    '''
        Function that calculates the comparisons of our approach versus the other data
        augmentation techniques using the standard trained model as a baseline for the
        comparisons.
    '''

    aug_techs = ["ours", "shen", "standard", "advbn", "maxup", "augmax", "augmix"]
    aug_techs = ["ours"]
    # aug_techs = ["augmix"]
    metrics = ["ma", "rmse", "mae"]

    for i in range(len(aug_techs)):
        for j in range(len(metrics)):
            calc_comparison_baseline(f'./logs/results_ours_{metrics[j]}.txt', 
                f'./logs/results_{aug_techs[i]}_{metrics[j]}.txt', 
                f'./logs/results_standard_{metrics[j]}.txt', metrics[j], aug_techs[i])

def calc_all_avgs_categories():
        aug_techs = ["standard", "shen", "advbn", "ours", "augmix", "maxup", "augmax"]
        aug_techs = ["ours"]
        # aug_techs = ["oursv2"]
        metrics = ["ma", "rmse", "mae"]

        for i in range(len(aug_techs)):
            for j in range(len(metrics)):
                calc_avg_categories(f'./logs/results_{aug_techs[i]}_{metrics[j]}.txt', metrics[j], aug_techs[i])


def main(args):
    if args.run_mode == "train":
        pl = PipelineJoint(args) # Training the AE on the batch method, but testing on the original pipeline b/c it's the same testing pipeline
        pl.train()

    if args.run_mode == "test_autojoin":
        aug_list = get_aug_list('./aug_list_all.txt')   

        # Testing on the benchmark datasets: single, clean, combined, & unseen
        for i in tqdm(range(117)):
            print(f"\n{i+1} {aug_list[i]}")

            pl = PipelineJoint(args, "test", aug_list[i], i)
            pl.test_our_approach()
    
    if args.run_mode == "test_others":
        aug_list = get_aug_list('./aug_list_all.txt')   

        # Testing on the benchmark datasets: single, clean, combined, & unseen
        for i in tqdm(range(117)):
            print(f"\n{i+1} {aug_list[i]}")

            pl = PipelineJoint(args, "test", aug_list[i], i)
            pl.test_other()

    # generate_average_files()
    # calc_all_avgs_categories()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Size of training batch")
    parser.add_argument("--lr", default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", default="data/", help="Data directory")
    parser.add_argument("--logs_dir", default="results")
    parser.add_argument("--checkpoints_dir", default="checkpoints/")
    parser.add_argument("--trained_models_dir", default="trained_models/")
    parser.add_argument("--train_epochs", type=int,  default=500, help="Number of training epochs")
    parser.add_argument("--model_dir", default='./saved_models/autoencoder.pt', help="Path to saved models")
    parser.add_argument("--seed", type=int, default=18474, help="Seed for the project")
    parser.add_argument("--dataset", default="honda")
    parser.add_argument("--model", default="resnet")
    parser.add_argument("--load", default="false")
    parser.add_argument("--run_mode", default="train", choices=["train", "test_autojoin", "test_others"])
    parser.add_argument("--img_dim", type=int, default=None)
    parser.add_argument("--lambda1", type=int, default=1)
    parser.add_argument("--lambda2", type=int, default=10)

    main(parser.parse_args())
