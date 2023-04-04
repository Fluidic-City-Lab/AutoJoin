import csv
import matplotlib.pyplot as plt
import numpy as np


def calc_average_stats(results_ours_file, results_robust_file, train_file, curriculum=True):
    aug_standard_total = 0.0
    aug_robust_total = 0.0

    num = 0
    
    with open(results_ours_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_standard_total += float(data[1])

            num += 1
    
    with open(results_robust_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_robust_total += float(data[1])
    
    aug_standard_average = aug_standard_total / num

    aug_robust_average = aug_robust_total / num

    print(f"The average accuracy for augmentations is: {aug_standard_average}")
    print(f"The average accuracy for robust augmentations is: {aug_robust_average}")

    total_time = 0.0
    num = 0

    if curriculum == True:
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                data = line.split(' ')
                data = data[7]

                total_time += float(data[:-2])
                num += 1
    else:
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                data = line.split(' ')

                total_time += float(data[-1])
                num += 1
    
    time_per_epoch_average = total_time / num

    print(f"The average time per epoch is: {time_per_epoch_average}")

    with open('./logs/average_stats', 'w') as f:
        f.write(f"Ours Average Aug Acc: {aug_standard_average}\n")
        f.write(f"Robust Average Aug Acc: {aug_robust_average}\n")
        f.write(f"Time Per Epoch Average: {time_per_epoch_average} s")


def generate_average_file(file1, file2, name):
    aug_methods = []

    aug_file1 = []
    clean_file1 = []

    aug_file2 = []
    clean_file2 = []

    with open(file1, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_methods.append(data[0])
            aug_file1.append(float(data[1]))
            clean_file1.append(float(data[2]))
    
    with open(file2, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')
            print(data)

            aug_file2.append(float(data[1]))
            clean_file2.append(float(data[2]))
    
    avg_aug = np.divide(np.add(aug_file1, aug_file2), 2.0)
    avg_clean = np.divide(np.add(clean_file1, clean_file2), 2.0)

    with open(f'./logs/results_{name}.txt', 'a') as f:
        for i in range(len(aug_methods)):
            f.write(f"{aug_methods[i]},{avg_aug[i]},{avg_clean[i]}\n")

def calc_comparison_baseline(results_ours_file, results_robust_file, results_baseline_file, metric, aug_tech):
    baseline_preds = []
    ours_preds = []
    robust_preds = []

    with open(results_ours_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            ours_preds.append(float(data[1]))
    
    with open(results_robust_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            robust_preds.append(float(data[1]))
    
    with open(results_baseline_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            baseline_preds.append(float(data[1]))
    
    ours_diffs = []
    robust_diffs = []

    if metric == "ma":
        for i in range(len(baseline_preds)):
            ours_diffs.append(ours_preds[i] - baseline_preds[i])
            robust_diffs.append(robust_preds[i] - baseline_preds[i])
    else:
        for i in range(len(baseline_preds)):
            ours_diffs.append(baseline_preds[i] - ours_preds[i])
            robust_diffs.append(baseline_preds[i] - robust_preds[i])

    assert len(ours_diffs) == len(baseline_preds)

    recon_diff_max = 0
    robust_diff_max = 0

    for i in range(len(ours_diffs)):
        if ours_diffs[i] > recon_diff_max:
            recon_diff_max = ours_diffs[i]
        
        if robust_diffs[i] > robust_diff_max:
            robust_diff_max = robust_diffs[i]
    
    recon_diff_avg_overall = np.average(ours_diffs)
    robust_diff_avg_overall = np.average(robust_diffs)

    ours_amai_clean = np.average(ours_diffs[0])
    shen_amai_clean = np.average(robust_diffs[0])

    ours_amai_single = np.average(ours_diffs[1:76])
    shen_amai_single = np.average(robust_diffs[1:76])

    ours_mmai_single = np.max(ours_diffs[0:76])
    shen_mmai_single = np.max(robust_diffs[0:76])

    ours_amai_combined = np.average(ours_diffs[76:82])
    shen_amai_combined = np.average(robust_diffs[76:82])

    ours_mmai_combined = np.max(ours_diffs[76:82])
    shen_mmai_combined = np.max(robust_diffs[76:82])

    ours_amai_unseen = np.average(ours_diffs[82:])
    shen_amai_unseen = np.average(robust_diffs[82:])

    ours_mmai_unseen = np.max(ours_diffs[82:])
    shen_mmai_unseen = np.max(robust_diffs[82:])

    print(f"{aug_tech} Overall AMAI: {robust_diff_avg_overall}\t MMAI: {robust_diff_max}")
    print(f"Ours Overall AMAI: {recon_diff_avg_overall}\t MMAI: {recon_diff_max}\n")
    
    print(f"Clean\tSingle Perturb\tCombined Pert.\t Unseen Perturb\n")
    print(f"AMAI\tAMAI\tMMAI\tAMAI\tMMAI\tAMAI\tMMAI\n")
    print(f"{shen_amai_clean:.2f}\t{shen_amai_single:.2f}\t{shen_mmai_single:.2f}\t{shen_amai_combined:.2f}\t{shen_mmai_combined:.2f}\t{shen_amai_unseen:.2f}\t{shen_mmai_unseen:.2f}\n")
    print(f"{ours_amai_clean:.2f}\t{ours_amai_single:.2f}\t{ours_mmai_single:.2f}\t{ours_amai_combined:.2f}\t{ours_mmai_combined:.2f}\t{ours_amai_unseen:.2f}\t{ours_mmai_unseen:.2f}")

    with open(f'./logs/comparison_baseline_{metric}.txt', 'a') as f:
        f.write(f"{aug_tech} Overall AMAI: {robust_diff_avg_overall}\t MMAI: {robust_diff_max}\n")
        f.write(f"Ours Overall AMAI: {recon_diff_avg_overall}\t MMAI: {recon_diff_max}\n\n")

        f.write(f"Clean\tSingle Perturb\tCombined Pert.\t Unseen Perturb\n")
        f.write(f"AMAI\tAMAI\tMMAI\tAMAI\tMMAI\tAMAI\tMMAI\n")
        f.write(f"{shen_amai_clean:.2f}\t{shen_amai_single:.2f}\t{shen_mmai_single:.2f}\t{shen_amai_combined:.2f}\t{shen_mmai_combined:.2f}\t{shen_amai_unseen:.2f}\t{shen_mmai_unseen:.2f}\n")
        f.write(f"{ours_amai_clean:.2f}\t{ours_amai_single:.2f}\t{ours_mmai_single:.2f}\t{ours_amai_combined:.2f}\t{ours_mmai_combined:.2f}\t{ours_amai_unseen:.2f}\t{ours_mmai_unseen:.2f}\n")


def calc_avg_categories(results_file, metric, aug_tech):
    results = []

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            results.append(float(data[1]))

    if metric == "ma":
        preds_avg_overall = np.average(results)
        preds_avg_clean = np.average(results[0])
        preds_avg_single = np.average(results[1:76])
        preds_avg_combined = np.average(results[76:82])
        preds_avg_unseen = np.average(results[82:])
    else:
        preds_avg_overall = 15 * np.average(results)
        preds_avg_clean = 15 * np.average(results[0])
        preds_avg_single = 15 * np.average(results[1:76])
        preds_avg_combined = 15 * np.average(results[76:82])
        preds_avg_unseen = 15 * np.average(results[82:])

    print(f"{aug_tech} Overall AMAI: {preds_avg_overall}\n")
    print(f"Clean\tSingle Perturb\tCombined Pert.\t Unseen Perturb\n")
    print(f"AMAI\tAMAI\tAMAI\tAMAI\n")
    print(f"{preds_avg_clean:.2f}\t{preds_avg_single:.2f}\t{preds_avg_combined:.2f}\t{preds_avg_unseen:.2f}\n")

    with open(f'./results/averages_{aug_tech}_{metric}.txt', 'a') as f:
        f.write(f"{aug_tech} Overall AMAI: {preds_avg_overall}\n")
        f.write(f"Clean\tSingle Perturb\tCombined Pert.\t Unseen Perturb\n")
        f.write(f"AMAI\tAMAI\tAMAI\tAMAI\n")
        f.write(f"{preds_avg_clean:.2f}\t{preds_avg_single:.2f}\t{preds_avg_combined:.2f}\t{preds_avg_unseen:.2f}\n")


def calc_comparison_adversarial(results_robust1, results_robust2, results_standard1, results_standard2):
    def get_averages(array1, array2):
        assert len(array1) == len(array2)

        averages = []

        for i in range(len(array1)):
            average = (array1[i] + array2[i]) / 2
            averages.append(round(average,2))
        
        return averages

    def get_differences(array1, array2):
        assert len(array1) == len(array2)

        diffs = []

        for i in range(len(array1)):
            diff = array1[i] - array2[i]

            diffs.append(round(diff,2))
        
        return diffs

    robust1_advex_preds = []
    robust1_recon_preds = []

    robust2_advex_preds = []
    robust2_recon_preds = []

    with open(results_robust1, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')
            
            robust1_advex_preds.append(float(data[1]))
            robust1_recon_preds.append(float(data[2]))
    
    with open(results_robust2, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            robust2_advex_preds.append(float(data[1]))
            robust2_recon_preds.append(float(data[2]))

    
    robust1_advex_preds_fgsm = robust1_advex_preds[36:45]
    robust1_recon_preds_fgsm = robust1_recon_preds[36:45]

    robust1_advex_preds_pgd = robust1_advex_preds[45:54]
    robust1_recon_preds_pgd = robust1_recon_preds[45:54]

    robust2_advex_preds_fgsm = robust2_advex_preds[54:63]
    robust2_recon_preds_fgsm = robust2_recon_preds[54:63]

    robust2_advex_preds_pgd = robust2_advex_preds[63:]
    robust2_recon_preds_pgd = robust2_recon_preds[63:]

    robust_advex_fgsm_avgs = get_averages(robust1_advex_preds_fgsm, robust2_advex_preds_fgsm)
    robust_recon_fgsm_avgs = get_averages(robust1_recon_preds_fgsm, robust2_recon_preds_fgsm)

    robust_fgsm_diffs = get_differences(robust_recon_fgsm_avgs, robust_advex_fgsm_avgs)

    robust_advex_pgd_avgs = get_averages(robust1_advex_preds_pgd, robust2_advex_preds_pgd)
    robust_recon_pgd_avgs = get_averages(robust1_recon_preds_pgd, robust2_recon_preds_pgd)

    robust_pgd_diffs = get_differences(robust_recon_pgd_avgs, robust_advex_pgd_avgs)


    standard1_advex_preds = []
    standard1_recon_preds = []

    standard2_advex_preds = []
    standard2_recon_preds = []

    with open(results_standard1, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')
            
            standard1_advex_preds.append(float(data[1]))
            standard1_recon_preds.append(float(data[2]))
    
    with open(results_standard2, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            standard2_advex_preds.append(float(data[1]))
            standard2_recon_preds.append(float(data[2]))
    
    standard1_advex_preds_fgsm = standard1_advex_preds[0:9]
    standard1_recon_preds_fgsm = standard1_recon_preds[0:9]

    standard1_advex_preds_pgd = standard1_advex_preds[9:18]
    standard1_recon_preds_pgd = standard1_recon_preds[9:18]

    standard2_advex_preds_fgsm = standard2_advex_preds[18:27]
    standard2_recon_preds_fgsm = standard2_recon_preds[18:27]

    standard2_advex_preds_pgd = standard2_advex_preds[27:36]
    standard2_recon_preds_pgd = standard2_recon_preds[27:36]

    standard_advex_fgsm_avgs = get_averages(standard1_advex_preds_fgsm, standard2_advex_preds_fgsm)
    standard_recon_fgsm_avgs = get_averages(standard1_recon_preds_fgsm, standard2_recon_preds_fgsm)

    standard_fgsm_diffs = get_differences(standard_recon_fgsm_avgs, standard_advex_fgsm_avgs)

    standard_advex_pgd_avgs = get_averages(standard1_advex_preds_pgd, standard2_advex_preds_pgd)
    standard_recon_pgd_avgs = get_averages(standard1_recon_preds_pgd, standard2_recon_preds_pgd)

    standard_pgd_diffs = get_differences(standard_recon_pgd_avgs, standard_advex_pgd_avgs)

    levels = [0.01, 0.025, 0.05, 0.075, 10.10, 0.125, 0.15, 0.175, 10.20]

    with open('./logs/gradient_based_paper_results.txt', 'a') as f:
        f.write(f"Level\tRobust FGSM\tRecon Robust FGSM\tRobust Diff\tStandard FGSM\tRecon Standard FGSM\tStandard Diff\n")

        for i in range(len(robust1_advex_preds_fgsm)):
            f.write(f"{levels[i]}\t{robust_advex_fgsm_avgs[i]}\t{robust_recon_fgsm_avgs[i]}\t{robust_fgsm_diffs[i]}\t{standard_advex_fgsm_avgs[i]}\t{standard_recon_fgsm_avgs[i]}\t{standard_fgsm_diffs[i]}\n")
        
        f.write(f"\n")
        f.write(f"Robust PGD\tRecon Robust PGD\tRobust Diff\tStandard PGD\tRecon Standard PGD\tStandard Diff\n")

        for i in range(len(robust1_advex_preds_pgd)):
            f.write(f"{levels[i]}\t{robust_advex_pgd_avgs[i]}\t{robust_recon_pgd_avgs[i]}\t{robust_pgd_diffs[i]}\t{standard_advex_pgd_avgs[i]}\t{standard_recon_pgd_avgs[i]}\t{standard_pgd_diffs[i]}\n")

def basic_stats(results_file, metric):
    results = []

    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            results.append(float(data[1]))
    
    avg = np.mean(results)
    min_value = np.min(results)
    max_value = np.max(results)

    print(f"Average {metric}: {avg}")
    print(f"Minimum {metric}: {min_value}")
    print(f"Maximum {metric}: {max_value}\n")