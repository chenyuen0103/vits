import os
import pandas as pd

# Define the base path for the directories to search
out_path = "./output/celeba_hessian/celebA/ViT/ViT-S_16/HessianERM/"
results_path = "./results/celeba_hessian/celebA/ViT/ViT-S_16/HessianERM/"
# List to hold directories with train.csv having less than 700 rows



def run_eval():
    dataset = 'waterbirds'
    run_name = f'{dataset.lower()}_hessian'
    # Loop through each subdirectory in the base path
    for dir in os.listdir(out_path):
        for seed in list(range(0, 5)):
        # Construct the path to the train.csv file
            model_path = os.path.join(out_path, dir, f"s{seed}", "ViT-S_16")
            result_file = os.path.join(results_path, dir, f"s{seed}", "ViT-S_16", "test_accuracy.csv")
            # Check if the train.csv file exists
            if not os.path.exists(result_file):
                eval_command = (
                    f'python evaluate.py --name {run_name} --model_arch ViT --model_type ViT-S_16 --dataset {dataset} --batch_size 64 --img_size 384 '
                    f'--checkpoint_dir {model_path}')
                print("running command", eval_command)
                os.system(eval_command)



def combine_results():
    dataset = 'celebA'
    # Loop through each subdirectory in the base path

    concatenated_df = pd.DataFrame()
    for dir in os.listdir(results_path):
        # Construct the path to the train.csv file
        result_file = os.path.join(results_path, dir, "s0", "ViT-S_16", "test_accuracy.csv")


if __name__ == '__main__':
    # combine_results()
    run_eval()

