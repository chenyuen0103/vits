import os
# import pandas as pd

# Define the base path for the directories to search
out_path = "./output/celeba_hessian/celebA/ViT/ViT-S_16/HessianERM/"
results_path = "./results/celeba_hessian/celebA/ViT/ViT-S_16/HessianERM/"
# List to hold directories with train.csv having less than 700 rows


if __name__ == '__main__':
    dataset = 'celebA'
    # Loop through each subdirectory in the base path
    for dir in os.listdir(out_path):
        # Construct the path to the train.csv file
        model_path = os.path.join(out_path, dir, "s0", "ViT-S_16")
        result_file = os.path.join(results_path, dir, "s0", "ViT-S_16", "test_accuracy.csv")
        # Check if the train.csv file exists
        if not os.path.exists(result_file):
            eval_command = (
                f'python evaluate.py --name celeba_hessian --model_arch ViT --model_type ViT-S_16 --dataset {dataset} --batch_size 64 --img_size 384 '
                f'--checkpoint_dir {model_path}')
            print("running command", eval_command)
            os.system(eval_command)

