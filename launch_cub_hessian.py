import os
import itertools
from tqdm import tqdm
import pandas as pd



def run_hessian():
    seed_list = [1,2,3,4]
    num_tries = 3
    # grad_alpha_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # hess_beta_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9][::-1]
    # grad_alpha_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0]
    grad_alpha_hess_beta_pairs = [
        (0.0, 0.0001),
        (0.01, 1e-08),
        (1.0, 1e-06),
        (0.01, 1e-06),
        (1e-08, 0.0001),
        (0.0001, 0.0001)
    ]

    base_path = "./logs/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/"
    for seed, (grad_alpha, hess_beta) in tqdm(itertools.product(seed_list, grad_alpha_hess_beta_pairs),
                                              desc='CUB Experiments'):
        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')

        existing_path = os.path.join(base_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/train.csv')
        for _ in range(num_tries):
            if os.path.exists(existing_path):
                existing_df = pd.read_csv(existing_path)
                # Check the number of rows
                if len(existing_df) >= 700:
                    print(f'Experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta} and seed={seed} already exists')
                    continue

            print(f'Running experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta}')
            train_command = (f'python train.py --name waterbirds_hessian --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --warmup_steps 100 '
                       f'--num_steps 700 --learning_rate 0.03 --batch_split 16 --img_size 384 --hessian_align --grad_alpha {grad_alpha} --hess_beta {hess_beta} '
                       f'--seed {seed}')
            os.system(train_command)

        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
        eval_command = (f'python evaluate.py --name waterbirds_hessian --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --batch_size 64 --img_size 384 '
                        f'--checkpoint_dir output/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/ViT-S_16')
        os.system(eval_command)


def erm():
    seed_list = [0, 1, 2]
    grad_alpha  = 1e-4
    hess_beta = 1e-4
    base_path = "./logs/waterbirds_erm/waterbirds/ViT/ViT-S_16/ERM/"
    grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
    for seed in seed_list:
        existing_path = os.path.join(base_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/train.csv')
        if os.path.exists(existing_path):
            existing_df = pd.read_csv(existing_path)
            # Check the number of rows
            if len(existing_df) >= 700:
                print(f'Experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta} already exists')
                continue

        print(f'Running experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta}')
        train_command = (f'python train.py --name waterbirds_erm --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --warmup_steps 100 '
                   f'--num_steps 700 --learning_rate 0.03 --batch_split 16 --img_size 384 '
                   f'--seed {seed}')
        os.system(train_command)

        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
        eval_command = (f'python evaluate.py --name waterbirds_erm --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --batch_size 64 --img_size 384 '
                        f'--checkpoint_dir output/waterbirds_erm/waterbirds/ViT/ViT-S_16/ERM/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/ViT-S_16')
        os.system(eval_command)

def main():
    run_hessian()
    # erm()

if __name__ == '__main__':
    main()