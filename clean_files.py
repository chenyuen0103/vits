import os
import pandas as pd

# Define the base path for the directories to search
base_path = "./logs/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/"

# List to hold directories with train.csv having less than 700 rows
directories_with_less_than_700_rows = []

# Loop through each subdirectory in the base path
for root, dirs, files in os.walk(base_path):
    for dir in dirs:
        # Construct the path to the train.csv file
        train_csv_path = os.path.join(root, dir, "s0", "train.csv")

        # Check if the train.csv file exists
        if os.path.exists(train_csv_path):
            # Read the CSV file
            df = pd.read_csv(train_csv_path)

            # Check the number of rows
            if len(df) < 700:
                # Add the directory to the list
                directories_with_less_than_700_rows.append(os.path.join(root, dir))

# Output the directories
directories_with_less_than_700_rows
print(directories_with_less_than_700_rows)
