import pandas as pd
import matplotlib.pyplot as plt
import itertools
import glob
import os
import numpy as np



def get_last_row_each_step(df):
    return df.groupby('global_step').tail(1)

# Function to find the worst group accuracy for each global_step
def get_worst_group_acc(df):
    acc_columns = [col for col in df.columns if 'avg_acc_group' in col]
    worst_acc_row = df[acc_columns].idxmin(axis=1)
    worst_acc = df[acc_columns].min(axis=1)
    worst_group = worst_acc_row.apply(lambda x: int(acc_columns[int(x[-1])][-1]))
    return df['global_step'], worst_acc, worst_group


def plot(run_name, dataset, algo, seed,log_path):
    grad_alpha_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0]
    hess_beta_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0][::-1]
    # Initialize a dictionary to store the worst-case loss
    worst_case_accuracies = []

    for grad_alpha, hess_beta in itertools.product(grad_alpha_values, hess_beta_values):
        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')

        train_path = os.path.join(log_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/train.csv')
        val_path = os.path.join(log_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/val.csv')
        config_key = f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}"
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            continue
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        val_df = get_last_row_each_step(val_df)


        # Calculate worst-case (minimum) accuracy per global_step for both training and validation data
        # worst_case_train_acc = train_df.groupby('global_step')['avg_acc'].min().reset_index()
        # worst_case_val_acc = val_df.groupby('global_step')['avg_acc'].min().reset_index()



        # Get worst-case accuracy for training and validation
        train_global_steps, train_worst_acc, train_worst_group = get_worst_group_acc(train_df)
        val_global_steps, val_worst_acc, val_worst_group = get_worst_group_acc(val_df)
        # Assuming the last row corresponds to the end of training
        worst_case_accuracies.append({
            'dataset': dataset,
            'grad_alpha': grad_alpha_formatted,
            'hess_beta': hess_beta_formatted,
            'worst_case_acc_train': train_worst_acc.iloc[-1],
            'worst_case_group_train': train_worst_group.iloc[-1],
            'worst_case_acc_val': val_worst_acc.iloc[-1],
            'worst_case_group_val': val_worst_group.iloc[-1],
            'worst_case_gap': val_worst_acc.iloc[-1] - train_worst_acc.iloc[-1],
        })

        # Plotting Worst-case Group Accuracy
        plt.figure(figsize=(7, 5))
        plt.plot(train_global_steps, train_worst_acc, label='Training', linestyle='-')
        plt.plot(val_global_steps, val_worst_acc, label='Validation', linestyle='-')
        plt.xlabel('global_step')
        plt.ylabel('Worst-case Group Accuracy')
        plt.title(f'{dataset}--Worst-case Group Accuracy\ngrad_alpha={grad_alpha_formatted}, hess_beta={hess_beta_formatted}')
        plt.legend()
        plt.show()
        plt.close()
        #
        # # Plot Training and Validation Loss
        # plt.figure(figsize=(7, 5))
        # if 'Hessian' in algo:
        #     plt.plot(train_df['global_step'], train_df['hessian_aligned_loss'], label='Training')
        #     plt.plot(val_df['global_step'], val_df['hessian_aligned_loss'], label='Validation')
        # else:
        #     plt.plot(train_df['global_step'], train_df['avg_actual_loss'], label='Training')
        #     plt.plot(val_df['global_step'], val_df['avg_actual_loss'], label='Validation')
        # plt.xlabel('global_step')
        # plt.ylabel('Loss')
        # title_suffix = 'Hessian Aligned Loss' if 'Hessian' in algo else 'ERM Loss'
        # plt.title(f'{dataset}--{title_suffix}\ngrad_alpha={grad_alpha_formatted}, hess_beta={hess_beta_formatted}')
        # plt.legend()
        # plt.ylim(0, min(train_df['hessian_aligned_loss'].max(), val_df['hessian_aligned_loss'].max()), 2)
        # plt.show()
        # plt.close()

        # # Store the worst-case accuracies for training and validation
        # worst_case_accuracies[(grad_alpha_formatted, hess_beta_formatted)] = {
        #     'training': train_worst_acc.iloc[-1],  # Take the last value as the end of training
        #     'validation': val_worst_acc.iloc[-1],  # Take the last value as the end of training
        # }

        # # Plotting
        # plt.figure(figsize=(7, 5))
        # plt.plot(train_global_steps, train_worst_acc, label='Training', linestyle='-')
        # plt.plot(val_global_steps, val_worst_acc, label='Validation', linestyle='-')
        # plt.xlabel('global_step')
        # plt.ylabel('Worst-case Group Accuracy')
        # plt.title(f'{dataset}--Worst-case Group Accuracy')
        # plt.legend()
        # # plt.xlim(0, 20)
        # # plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_worst_group_acc_scheduler.png')
        # plt.show()
        # plt.close()
        #
        # # Plot Training and Validation Loss
        # plt.figure(figsize=(7, 5))
        # if 'Hessian' in algo:
        #     plt.plot(train_df['global_step'], train_df['hessian_aligned_loss'], label='Training')
        #     plt.plot(val_df['global_step'], val_df['hessian_aligned_loss'], label='Validation')
        # else:
        #     plt.plot(train_df['global_step'], train_df['avg_actual_loss'], label='Training')
        #     plt.plot(val_df['global_step'], val_df['avg_actual_loss'], label='Validation')
        # plt.xlabel('global_step')
        # plt.ylabel('Loss')
        # if 'Hessian' in algo:
        #     plt.title(f'{dataset}--Hessian Aligned Loss')
        # else:
        #     plt.title(f'{dataset}--ERM Loss')
        # plt.legend()
        # # plt.xlim(0, 20)
        # # plt.ylim(0, 2)
        # # plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_loss_scheduler.png')
        # plt.show()
        # plt.close()



        # # plot average accuracy of training and validation
        # plt.figure(figsize=(7, 5))
        # plt.plot(train_df['global_step'], train_df['avg_acc'], label='Training', linestyle='-')
        # plt.plot(val_df['global_step'], val_df['avg_acc'], label='Validation', linestyle='-')
        # plt.xlabel('global_step')
        # plt.ylabel('Average Group Accuracy')
        # plt.title(f'{dataset}--Average Group Accuracy')
        # plt.legend()
        # # plt.xlim(0, 20)
        # # plt.savefig(f'../logs/{dataset}/{model}/{algo}/s{seed}/{dataset}_avg_group_acc_scheduler.png')
        # plt.show()
        # plt.close()
    pd.DataFrame(worst_case_accuracies).to_csv(f'./results/{run_name}/worst_case_accuracies.csv', index = False)





def collect_test_data(run_name, dataset, algo, result_path="./results/"):
    dir_pattern = os.path.join(result_path, f"{run_name}/{dataset}/ViT/ViT-S_16/{algo}/grad_alpha_*_hess_beta_*/")
    hyperparam_dirs = glob.glob(dir_pattern)
    data = []

    for hp_dir in hyperparam_dirs:
        parts = hp_dir.split('/')
        grad_alpha_part = [part for part in parts if 'grad_alpha_' in part][0]
        hess_beta_part = [part for part in parts if 'hess_beta_' in part][0]

        grad_alpha = grad_alpha_part.split('grad_alpha_')[-1].split('_hess_beta_')[0]
        hess_beta = hess_beta_part.split('hess_beta_')[-1]

        test_accuracy_files = glob.glob(os.path.join(hp_dir, 's*/ViT-S_16/test_accuracy.csv'))

        avg_test_accuracies = []
        worst_test_accuracies = []

        for file_path in test_accuracy_files:
            try:
                test_df = pd.read_csv(file_path)
                avg_test_accuracies.append(test_df['Avg_Test_Acc'].iloc[0])
                worst_test_accuracies.append(test_df['Worst_Test_Acc'].iloc[0])
            except pd.errors.EmptyDataError:
                print(f"Empty or malformed file detected: {file_path}")
                continue

        if avg_test_accuracies:
            # Calculate average and standard error for avg_test_accuracy and worst_test_accuracy
            avg_test_acc_mean = np.mean(avg_test_accuracies)
            avg_test_acc_sem = np.std(avg_test_accuracies, ddof=1) / np.sqrt(len(avg_test_accuracies))
            worst_test_acc_mean = np.mean(worst_test_accuracies)
            worst_test_acc_sem = np.std(worst_test_accuracies, ddof=1) / np.sqrt(len(worst_test_accuracies))

            data.append({
                'dataset': dataset,
                'split': 'test',  # Assuming 'test' as the split
                'grad_alpha': grad_alpha,
                'hess_beta': hess_beta,
                'avg_test_accuracy': avg_test_acc_mean,
                'avg_test_accuracy_sem': avg_test_acc_sem,
                'worst_test_accuracy': worst_test_acc_mean,
                'worst_test_accuracy_sem': worst_test_acc_sem,
                'num_runs': len(avg_test_accuracies)
            })

    results_df = pd.DataFrame(data)
    results_df['grad_alpha'] = pd.to_numeric(results_df['grad_alpha'])
    results_df['hess_beta'] = pd.to_numeric(results_df['hess_beta'])
    results_df.to_csv(f'./results/{run_name}/{dataset}/ViT/ViT-S_16/{algo}/all_test_results.csv', index=False)
    return results_df


def collect_val_data(run_name, dataset, algo, log_path="./logs/"):
    dir_pattern = os.path.join(log_path, f"{run_name}/{dataset}/ViT/ViT-S_16/{algo}/grad_alpha_*_hess_beta_*/")
    hyperparam_dirs = glob.glob(dir_pattern)
    data = []

    for hp_dir in hyperparam_dirs:
        # Splitting the path and extracting grad_alpha and hess_beta correctly
        path_parts = hp_dir.split('/')
        grad_alpha_part = [part for part in path_parts if 'grad_alpha_' in part][0]
        hess_beta_part = [part for part in path_parts if 'hess_beta_' in part][0]

        grad_alpha = grad_alpha_part.split('grad_alpha_')[-1].split('_hess_beta_')[0]
        hess_beta = hess_beta_part.split('hess_beta_')[-1]

        seed_dirs = glob.glob(os.path.join(hp_dir, 's*'))

        val_results = []
        avg_accs = []
        worst_case_accs = []

        for seed_dir in seed_dirs:
            train_file = os.path.join(seed_dir, 'train.csv')
            val_file = os.path.join(seed_dir, 'val.csv')

            if os.path.exists(train_file) and len(pd.read_csv(train_file)) >= 700:
                if os.path.exists(val_file):
                    val_df = pd.read_csv(val_file)
                    if not val_df.empty:
                        val_results.append(val_df.iloc[-1])

        for val_result in val_results:
            group_accs = [val_result[f'avg_acc_group:{i}'] for i in range(4)]
            avg_accs.append(np.mean(group_accs))
            worst_case_acc = min(group_accs)
            worst_case_accs.append(worst_case_acc)

        if val_results:  # Ensuring there are results before attempting to compute averages
            avg_acc_mean = np.mean(avg_accs)
            avg_acc_sem = np.std(avg_accs, ddof=1) / np.sqrt(len(avg_accs))
            worst_case_acc_mean = np.mean(worst_case_accs)
            worst_case_acc_sem = np.std(worst_case_accs, ddof=1) / np.sqrt(len(worst_case_accs))

            # Append collected data for this hyperparameter combination
            data.append({
                'dataset': dataset,
                'split': 'val',  # Assuming 'split' is static or derived from elsewhere
                'grad_alpha': grad_alpha,
                'hess_beta': hess_beta,
                'avg_acc_mean': avg_acc_mean,
                'avg_acc_sem': avg_acc_sem,
                'worst_case_acc_mean': worst_case_acc_mean,
                'worst_case_acc_sem': worst_case_acc_sem,
                'num_runs': len(val_results)
            })

    # Convert the collected data to a DataFrame and return it
    results_df = pd.DataFrame(data)
    results_df['grad_alpha'] = pd.to_numeric(results_df['grad_alpha'])
    results_df['hess_beta'] = pd.to_numeric(results_df['hess_beta'])
    results_df.to_csv(f'./results/{run_name}/{dataset}/ViT/ViT-S_16/{algo}/all_val_results.csv', index=False)
    return results_df


def compute_stats(run_name, dataset, algo = 'ERM', grad_alpha = 1e-4, hess_beta = 1e-4):
    grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
    result_path = f"./results/{run_name}/{dataset}/ViT/ViT-S_16/{algo}/"

    result_path = os.path.join(result_path,
                 f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}')

    dfs = []
    for dir in os.listdir(result_path):
        csv_path = os.path.join(result_path, dir, 'ViT-S_16/test_accuracy.csv')
        df = pd.read_csv(csv_path)
        dfs.append(df)
    all_df = pd.concat(dfs)
    train_avg_mean = all_df['Avg_Train_Acc'].mean()
    train_avg_std = all_df['Avg_Train_Acc'].std()
    train_worst_mean = all_df['Worst_Train_Acc'].mean()
    train_worst_std = all_df['Worst_Train_Acc'].std()

    test_avg_mean = all_df['Avg_Test_Acc'].mean()
    test_avg_std = all_df['Avg_Test_Acc'].std()
    test_worst_mean = all_df['Worst_Test_Acc'].mean()
    test_worst_std = all_df['Worst_Test_Acc'].std()
    print(f"Train Avg: {train_avg_mean} +/- {train_avg_std}")
    print(f"Train Worst: {train_worst_mean} +/- {train_worst_std}")
    print(f"Test Avg: {test_avg_mean} +/- {test_avg_std}")
    print(f"Test Worst: {test_worst_mean} +/- {test_worst_std}")

    # return train_avg_mean, train_avg_std, train_worst_mean, train_worst_std, test_avg_mean, test_avg_std, test_worst_mean, test_worst_std

    # get the mean and std of the worst-test accuracy

def find_baseline(val_df, test_df):
    baseline_val = val_df[(val_df['grad_alpha'] == 0) & (val_df['hess_beta'] == 0)]
    baseline_test = test_df[(test_df['grad_alpha'] == 0) & (test_df['hess_beta'] == 0)]
    print("Baseline Performance: grad_alpha=0, hess_beta=0")
    print(f"Baseline avergage test accuracy: {baseline_test['avg_test_accuracy'].item()} ± {baseline_test['avg_test_accuracy_sem'].item()}")
    print(f"Baseline worst-case test accuracy: {baseline_test['worst_test_accuracy'].item()} ± {baseline_test['worst_test_accuracy_sem'].item()}")
    return baseline_val, baseline_test

def find_best_gm(val_df, test_df, worst_case=False):
    val_df[(val_df['grad_alpha'] != 0) & (val_df['hess_beta'] == 0)]
    test_df[(test_df['grad_alpha'] != 0) & (test_df['hess_beta'] == 0)]
    print("Finding best hyperparameters for GM")
    return find_best_hyperparameters(val_df, test_df, worst_case)

def find_best_hm(val_df, test_df, worst_case=False):
    val_df = val_df[(val_df['grad_alpha'] == 0) & (val_df['hess_beta'] != 0)]
    test_df = test_df[(test_df['grad_alpha'] == 0) & (test_df['hess_beta'] != 0)]
    print("Finding best hyperparameters for HM")
    return find_best_hyperparameters(val_df, test_df, worst_case)

def find_best_gm_hm(val_df, test_df, worst_case=False):
    val_df = val_df[(val_df['grad_alpha'] != 0) & (val_df['hess_beta'] != 0)]
    test_df = test_df[(test_df['grad_alpha'] != 0) & (test_df['hess_beta'] != 0)]
    print("Finding best hyperparameters for GM + HM")
    return find_best_hyperparameters(val_df, test_df, worst_case)


def find_best_hyperparameters(val_df, test_df, worst_case=False):
    # Adjust metric names to match the new DataFrame structure
    primary_metric = 'worst_case_acc_mean' if worst_case else 'avg_acc_mean'
    secondary_metric = 'worst_case_acc_sem' if worst_case else 'avg_acc_sem'

    # Find the max value of the primary metric
    max_primary_metric_value = val_df[primary_metric].max()

    # Filter rows with the max primary metric value
    candidates = val_df[val_df[primary_metric] == max_primary_metric_value]

    # If there are multiple candidates, choose the one with the smallest corresponding SEM
    if len(candidates) > 1:
        best_candidate = candidates.loc[candidates[secondary_metric].idxmin()]
    else:
        best_candidate = candidates.iloc[0]

    # Extract the best grad_alpha and hess_beta
    best_grad_alpha = best_candidate['grad_alpha']
    best_hess_beta = best_candidate['hess_beta']

    # Find the performance of these hyperparameters in test_df
    # Ensure grad_alpha and hess_beta values are compared correctly
    test_performance = test_df[(test_df['grad_alpha'] == best_grad_alpha) & (test_df['hess_beta'] == best_hess_beta)]


    print(f"Test Performance for {'worst' if worst_case else 'average'} case:")
    if worst_case:
        print(f"Best grad_alpha: {best_grad_alpha}, Best hess_beta: {best_hess_beta}, Best worst-case test accuracy: {test_performance['worst_test_accuracy'].item()} ± {test_performance['worst_test_accuracy_sem'].item()}")
    else:
        print(f"Best grad_alpha: {best_grad_alpha}, Best hess_beta: {best_hess_beta}, Best average test accuracy: {test_performance['avg_test_accuracy'].item()} ± {test_performance['avg_test_accuracy_sem'].item()}")
    return best_grad_alpha, best_hess_beta, test_performance


def main():
    # plot()
    # run_name = 'celeba_hessian'
    # run_name = 'celeba_erm'
    # dataset = 'celebA'
    run_name = 'waterbirds_hessian'
    # run_name = 'waterbirds_erm'
    dataset = 'waterbirds'
    algo = 'HessianERM'
    # algo = 'ERM'
    log_path = f"./logs/{run_name}/{dataset}/ViT/ViT-S_16/{algo}/"
    # plot(run_name, dataset, algo, 0, log_path)
    # Load the training and validation data
    # dataset = 'CelebA'
    # algo = 'HessianERM'
    # algo = 'ERM'
    # grad_alpha = 1e-4
    # hess_beta = 1e-4

    grad_alpha_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0]
    hess_beta_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0][::-1]

    # compute_stats(run_name, dataset, algo, grad_alpha, hess_beta, )

    # grad_alpha, hess_beta = find_best_gm(run_name, dataset)
    # grad_alpha, hess_beta = find_best_hm(run_name, dataset)

    # grad_alpha, hess_beta = find_best_alpha_beta(run_name, dataset)
    # compute_stats(run_name, dataset, algo, grad_alpha, hess_beta, )

    run_name = "celeba_hessian"
    dataset = "celebA"
    # run_name = "waterbirds_hessian"
    # dataset = "waterbirds"
    algo = "HessianERM"
    val_df = collect_val_data(run_name, dataset, algo)
    test_df = collect_test_data(run_name, dataset, algo)
    find_baseline(val_df, test_df)
    find_best_gm(val_df, test_df, worst_case=True)
    find_best_hm(val_df, test_df, worst_case=True)
    find_best_gm_hm(val_df, test_df, worst_case=True)

    # best_alpha, best_beta, best_row = find_best_hyperparameters(val_df, test_df, worst_case=True)
    # print(best_row)




if __name__ == "__main__":
    main()