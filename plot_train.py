import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

run_name = 'celeba_hessian'
dataset = 'celebA'
algo = 'HessianERM'
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


def get_last_row_each_step(df):
    return df.groupby('global_step').tail(1)

# Function to find the worst group accuracy for each global_step
def get_worst_group_acc(df):
    # Extract only columns that contain group accuracy
    acc_columns = [col for col in df.columns if 'avg_acc_group' in col]
    # Find the minimum accuracy across these columns for each global_step
    worst_acc = df[acc_columns].min(axis=1)
    return df['global_step'], worst_acc


# Initialize a dictionary to store the worst-case loss
worst_case_accuracies = {}

for grad_alpha, hess_beta in itertools.product(grad_alpha_values, hess_beta_values):
    grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')

    train_path = os.path.join(log_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/train.csv')
    val_path = os.path.join(log_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/val.csv')
    config_key = f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    val_df = get_last_row_each_step(val_df)


    # Calculate worst-case (minimum) accuracy per global_step for both training and validation data
    # worst_case_train_acc = train_df.groupby('global_step')['avg_acc'].min().reset_index()
    # worst_case_val_acc = val_df.groupby('global_step')['avg_acc'].min().reset_index()



    # Get worst-case accuracy for training and validation
    train_global_steps, train_worst_acc = get_worst_group_acc(train_df)
    val_global_steps, val_worst_acc = get_worst_group_acc(val_df)

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

    # Plot Training and Validation Loss
    plt.figure(figsize=(7, 5))
    if 'Hessian' in algo:
        plt.plot(train_df['global_step'], train_df['hessian_aligned_loss'], label='Training')
        plt.plot(val_df['global_step'], val_df['hessian_aligned_loss'], label='Validation')
    else:
        plt.plot(train_df['global_step'], train_df['avg_actual_loss'], label='Training')
        plt.plot(val_df['global_step'], val_df['avg_actual_loss'], label='Validation')
    plt.xlabel('global_step')
    plt.ylabel('Loss')
    title_suffix = 'Hessian Aligned Loss' if 'Hessian' in algo else 'ERM Loss'
    plt.title(f'{dataset}--{title_suffix}\ngrad_alpha={grad_alpha_formatted}, hess_beta={hess_beta_formatted}')
    plt.legend()
    plt.ylim(0, min(train_df['hessian_aligned_loss'].max(), val_df['hessian_aligned_loss'].max()), 2)
    plt.show()
    plt.close()

    # Store the worst-case accuracies for training and validation
    worst_case_accuracies[(grad_alpha_formatted, hess_beta_formatted)] = {
        'training': train_worst_acc.iloc[-1],  # Take the last value as the end of training
        'validation': val_worst_acc.iloc[-1],  # Take the last value as the end of training
    }

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


    #
    #
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

# Display the worst-case accuracies for all parameter pairs
for params, accuracies in worst_case_accuracies.items():
    grad_alpha, hess_beta = params
    print(f"grad_alpha={grad_alpha}, hess_beta={hess_beta}, Training Worst-case Accuracy: {accuracies['training']}, Validation Worst-case Accuracy: {accuracies['validation']}")

