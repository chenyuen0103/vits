import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os



def get_last_row_each_step(df):
    return df.groupby('global_step').tail(1)

# Function to find the worst group accuracy for each global_step
def get_worst_group_acc(df):
    acc_columns = [col for col in df.columns if 'avg_acc_group' in col]
    worst_acc_row = df[acc_columns].idxmin(axis=1)
    worst_acc = df[acc_columns].min(axis=1)
    worst_group = worst_acc_row.apply(lambda x: int(acc_columns[int(x[-1])][-1]))
    return df['global_step'], worst_acc, worst_group


def plot():
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



def main():
    # plot()
    # run_name = 'celeba_hessian'
    run_name = 'celeba_erm'
    dataset = 'celebA'
    # run_name = 'waterbirds_hessian'
    # run_name = 'waterbirds_erm'
    # dataset = 'waterbirds'
    # algo = 'HessianERM'
    algo = 'ERM'
    log_path = f"./logs/{run_name}/{dataset}/ViT/ViT-S_16/{algo}/"
    # Load the training and validation data
    # dataset = 'CelebA'
    # algo = 'HessianERM'
    # algo = 'ERM'
    seed = 0
    # grad_alpha = 1e-4
    # hess_beta = 1e-4

    grad_alpha_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0]
    hess_beta_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0][::-1]
    grad_alpha = 1e-4
    hess_beta = 1e-4
    compute_stats(run_name, dataset, algo, grad_alpha, hess_beta, )

if __name__ == "__main__":
    main()