import csv
import matplotlib.pyplot as plt
import numpy as np

def generate_boxplot_csv(data_file):
    l1 = ""
    l2 = ""
    l3 = ""
    l4 = ""
    l5 = ""

    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')
            aug_level = int(data[0][-1])

            if aug_level == 1:
                l1 += data[2] + ","
            elif aug_level == 2:
                l2 += data[2] + ","
            elif aug_level == 3:
                l3 += data[2] + ","
            elif aug_level == 4:
                l4 += data[2] + ","
            elif aug_level == 5:
                l5 += data[2] + ","
            else:
                pass
    
    with open('./boxplot/boxplot_csv/resnet_recon_AT20k_curriculumhalf.csv', 'w') as f:
        f.write(l1[:-1] + "\n")
        f.write(l2[:-1] + "\n")
        f.write(l3[:-1] + "\n")
        f.write(l4[:-1] + "\n")
        f.write(l5[:-1] + "\n")

def generate_boxplot(csv_file, name, bound):

    data_list = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            for j in range(len(line)):
                line[j] = float(line[j])
            data_list.append(line)
    
    fig, ax = plt.subplots()
    plt.axis([0, 6, bound[0], bound[1]])
    ax.boxplot(data_list)
    plt.xticks([1,2,3,4,5], ["L1", "L2", "L3", "L4", "L5"])
    fig.savefig('./boxplot/' + name + '.png')

def generate_linegraph_augvsrecon(results_standard_file, results_robust_file):
    aug_results = []
    recon_results = []
    clean_results = []

    aug_robust_results = []
    recon_robust_results = []
    clean_robust_results = []

    with open(results_standard_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_results.append(float(data[1])) # Average results of standard regressor on reconstructed data
            recon_results.append(float(data[2])) # Average results of standard regressor on original augmented data
            clean_results.append(float(data[3])) # Average results of standard regressor on clean data
    
    with open(results_robust_file, "r") as g:
        for line in g:
            line = line.strip()
            data = line.split(',')

            aug_robust_results.append(float(data[1]))
            recon_robust_results.append(float(data[2]))
            clean_robust_results.append(float(data[3]))

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(aug_results))
    ax.plot(np.array(recon_results))
    ax.plot(np.array(clean_results))
    ax.legend(["Aug Results", "Recon Results", "Clean Results"])
    fig.savefig('./logs/standard_results_graph.png')

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(aug_robust_results))
    ax.plot(np.array(recon_robust_results))
    ax.plot(np.array(clean_robust_results))
    ax.legend(["Aug Results", "Recon Results", "Clean Results"])
    fig.savefig('./logs/robust_results_graph.png')

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(aug_results))
    ax.plot(np.array(recon_results))
    ax.plot(np.array(clean_results))
    ax.plot(np.array(aug_robust_results))
    ax.plot(np.array(recon_robust_results))
    ax.plot(np.array(clean_robust_results))
    ax.legend(["Aug Results", "Recon Results", "Clean Results", "Aug Robust Results", "Recon Robust Results", "Clean Robust Results"])
    fig.savefig('./logs/combined_results_graph.png')

    fig, ax = plt.subplots(figsize=(16,5), dpi=200)
    ax.set_ylabel("Avg. Accuracy") 
    ax.plot(np.array(recon_results))
    ax.plot(np.array(clean_results))
    ax.plot(np.array(aug_robust_results))
    ax.plot(np.array(clean_robust_results))
    ax.legend(["Recon Results", "Clean Results", "Aug Robust Results", "Recon Robust Results"])
    fig.savefig('./logs/target_results_graph.png')

def calc_average_stats(results_standard_file, results_robust_file, train_file, curriculum=True):
    aug_standard_total = 0.0
    recon_standard_total = 0.0
    
    aug_robust_total = 0.0
    recon_robust_total = 0.0

    num = 0
    
    with open(results_standard_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_standard_total += float(data[1])
            recon_standard_total += float(data[2])

            num += 1
    
    with open(results_robust_file, "r") as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_robust_total += float(data[1])
            recon_robust_total += float(data[2])
    
    aug_standard_average = aug_standard_total / num
    recon_standard_average = recon_standard_total / num

    aug_robust_average = aug_robust_total / num
    recon_robust_average = recon_robust_total / num

    print(f"The average accuracy for augmentations is: {aug_standard_average}")
    print(f"The average accuracy for reconstruction is: {recon_standard_average}")
    print(f"The average accuracy for robust augmentations is: {aug_robust_average}")
    print(f"The average accuracy for robust reconstructions is: {recon_robust_average}")

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
        f.write(f"Standard Average Aug Acc: {aug_standard_average}\n")
        f.write(f"Standard Average Recon Acc: {recon_standard_average}\n")
        f.write(f"Robust Average Aug Acc: {aug_robust_average}\n")
        f.write(f"Robust Average Recon Acc: {recon_robust_average}\n")
        f.write(f"Time Per Epoch Average: {time_per_epoch_average} s")

def generate_overleaf_dmproject(file):
    count = 0
    overleaf_dmproject = "& "
    with open(file, 'r') as f:
        for line in f:
          line = line.strip()
          data = line.split(',')

          if count > 6:
              if (count - 6) % 5 == 0:
                  recon_acc = f"{float(data[2]):.2f}\\% \\\\"
                  overleaf_dmproject += recon_acc
                  with open('./logs/overleaf_dmproject.txt', 'a') as g:
                      g.write(overleaf_dmproject + "\n")
                  
                  overleaf_dmproject = "& "
              
              else:
                  recon_acc = f"{float(data[2]):.2f}\\% & "
                  overleaf_dmproject += recon_acc
          
          count += 1

def generate_average_file(file1, file2, name):
    aug_methods = []

    aug_file1 = []
    recon_file1 = []
    clean_file1 = []

    aug_file2 = []
    recon_file2 = []
    clean_file2 = []

    with open(file1, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_methods.append(data[0])
            aug_file1.append(float(data[1]))
            recon_file1.append(float(data[2]))
            clean_file1.append(float(data[3]))
    
    with open(file2, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            aug_file2.append(float(data[1]))
            recon_file2.append(float(data[2]))
            clean_file2.append(float(data[3]))
    
    avg_aug = np.divide(np.add(aug_file1, aug_file2), 2.0)
    avg_recon = np.divide(np.add(recon_file1, recon_file2), 2.0)
    avg_clean = np.divide(np.add(clean_file1, clean_file2), 2.0)

    with open(f'./logs/results_{name}.txt', 'a') as f:
        for i in range(len(aug_methods)):
            f.write(f"{aug_methods[i]},{avg_aug[i]},{avg_recon[i]},{avg_clean[i]}\n")

def calc_comparison_baseline(results_standard_file, results_robust_file):
    baseline_preds = []
    recon_preds = []
    robust_preds = []

    with open(results_standard_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            baseline_preds.append(float(data[1]))
            recon_preds.append(float(data[2]))
    
    with open(results_robust_file, 'r') as f:
        for line in f:
            line = line.strip()
            data = line.split(',')

            robust_preds.append(float(data[1]))
    
    recon_diffs = []
    robust_diffs = []

    for i in range(len(baseline_preds)):
        recon_diffs.append(recon_preds[i] - baseline_preds[i])
        robust_diffs.append(robust_preds[i] - baseline_preds[i])

    assert len(recon_diffs) == len(baseline_preds)

    recon_diff_max = 0
    robust_diff_max = 0


    for i in range(len(recon_diffs)):
        if recon_diffs[i] > recon_diff_max:
            recon_diff_max = recon_diffs[i]
        
        if robust_diffs[i] > robust_diff_max:
            robust_diff_max = robust_diffs[i]
    
    recon_diff_avg_overall = np.average(recon_diffs)
    robust_diff_avg_overall = np.average(robust_diffs)

    ours_amai_clean = np.average(recon_diffs[75])
    shen_amai_clean = np.average(robust_diffs[75])

    ours_amai_single = np.average(recon_diffs[0:75])
    shen_amai_single = np.average(robust_diffs[0:75])

    ours_mmai_single = np.max(recon_diffs[0:75])
    shen_mmai_single = np.max(robust_diffs[0:75])

    ours_amai_combined = np.average(recon_diffs[76:82])
    shen_amai_combined = np.average(robust_diffs[76:82])

    ours_mmai_combined = np.max(recon_diffs[77:82])
    shen_mmai_combined = np.max(robust_diffs[77:82])

    ours_amai_unseen = np.average(recon_diffs[82:])
    shen_amai_unseen = np.average(robust_diffs[82:])

    ours_mmai_unseen = np.max(recon_diffs[82:])
    shen_mmai_unseen = np.max(robust_diffs[82:])


    print(f"Shen Overall AMAI: {robust_diff_avg_overall}\t MMAI: {robust_diff_max}")
    print(f"Ours Overall AMAI: {recon_diff_avg_overall}\t MMAI: {recon_diff_max}\n")

    # print(f"Shen Clean AMAI: {shen_amai_clean}")
    # print(f"Ours Clean AMAI: {ours_amai_clean}")

    # print(f"Shen Single AMAI: {shen_amai_single}\t MMAI: {shen_mmai_single}")
    # print(f"Ours Single AMAI: {ours_amai_single}\t MMAI: {ours_mmai_single}")

    # print(f"Shen Single AMAI: {shen_amai_combined}\t MMAI: {shen_mmai_combined}")
    # print(f"Ours Single AMAI: {ours_amai_combined}\t MMAI: {ours_mmai_combined}")

    # print(f"Shen Single AMAI: {shen_amai_unseen}\t MMAI: {shen_mmai_unseen}")
    # print(f"Ours Single AMAI: {ours_amai_unseen}\t MMAI: {ours_mmai_unseen}")

    print(f"Clean\tSingle Perturb\tCombined Pert.\t Unseen Perturb\n")
    print(f"AMAI\tAMAI\tMMAI\tAMAI\tMMAI\tAMAI\tMMAI\n")
    print(f"{shen_amai_clean:.2f}\t{shen_amai_single:.2f}\t{shen_mmai_single:.2f}\t{shen_amai_combined:.2f}\t{shen_mmai_combined:.2f}\t{shen_amai_unseen:.2f}\t{shen_mmai_unseen:.2f}\n")
    print(f"{ours_amai_clean:.2f}\t{ours_amai_single:.2f}\t{ours_mmai_single:.2f}\t{ours_amai_combined:.2f}\t{ours_mmai_combined:.2f}\t{ours_amai_unseen:.2f}\t{ours_mmai_unseen:.2f}")

def calc_comparison_adversarial(results_robust1, results_robust2, results_standard1, results_standard2):
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