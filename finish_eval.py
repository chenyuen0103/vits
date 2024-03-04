import os
import pandas as pd

# Define the base path for the directories to search
out_path = "./output/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/"
results_path = "./results/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/"
# List to hold directories with train.csv having less than 700 rows


if __name__ == '__main__':
    dataset = 'waterbirds'
    # Loop through each subdirectory in the base path
    for root, dirs, files in os.walk(out_path):
        for dir in dirs:
            # Construct the path to the train.csv file
            model_path = os.path.join(root, dir, "s0", "ViT-S_16.bin")
            result_file = os.path.join(results_path, dir, "s0", "test_accuracy.csv")
            # Check if the train.csv file exists
            if os.path.exists(model_path) and not os.path.exists(result_file):
                eval_command = (
                    f'python evaluate.py --name {dataset}_hessian --model_arch ViT --model_type ViT-S_16 --dataset {dataset} --batch_size 64 --img_size 384 '
                    f'--checkpoint_dir {model_path}')
                print("running command", eval_command)
                os.system(eval_command)

