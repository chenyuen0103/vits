import os
import itertools
from tqdm import tqdm
import pandas as pd


def erm():
    seed_list = [3, 4]
    grad_alpha = 1e-4
    hess_beta = 1e-4
    base_path = "./logs/celeba_erm/celebA/ViT/ViT-S_16/ERM/"
    grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
    for seed in seed_list:
        existing_path = os.path.join(base_path,
                                     f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/train.csv')
        if os.path.exists(existing_path):
            existing_df = pd.read_csv(existing_path)
            # Check the number of rows
            if len(existing_df) >= 700:
                print(f'Experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta} already exists')
                continue

        print(f'Running experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta}')
        train_command = (
            f'python train.py --name celeba_erm --model_arch ViT --model_type ViT-S_16 --dataset celebA --warmup_steps 100 '
            f'--num_steps 700 --learning_rate 0.03 --batch_split 16 --img_size 384 '
            f'--seed {seed}')
        os.system(train_command)

        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
        eval_command = (
            f'python evaluate.py --name celeba_erm --model_arch ViT --model_type ViT-S_16 --dataset celebA --batch_size 64 --img_size 384 '
            f'--checkpoint_dir output/celeba_erm/celebA/ViT/ViT-S_16/ERM/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/ViT-S_16')
        os.system(eval_command)


def hessian():
    seed_list = [1,2,3,4]
    num_tries = 3
    grad_alpha_hess_beta_pairs = [
        (0.01, 0.01),
        (0.0001, 0.01),
        (0.0001, 1e-08),
        (1e-06, 0.0001),
        (0.01, 1e-08),
        (1e-06, 1e-08),
        (1.0, 1e-08),
        (0.01, 1e-06),
        (1e-06, 1e-06)
    ]

    base_path = "./logs/celeba_hessian/celebA/ViT/ViT-S_16/HessianERM/"
    for seed, (grad_alpha, hess_beta) in tqdm(itertools.product(seed_list, grad_alpha_hess_beta_pairs),
                                              desc='CelebA Experiments'):
        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
        existing_path = os.path.join(base_path, f'grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/train.csv')
        for _ in range(num_tries):
            if os.path.exists(existing_path):
                existing_df = pd.read_csv(existing_path)
                # Check the number of rows
                if len(existing_df) >= 700:
                    print(f'Experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta} already exists')
                    continue

            print(f'Running experiment with grad_alpha={grad_alpha} and hess_beta={hess_beta}')
            train_command = (f'python train.py --name celeba_hessian --model_arch ViT --model_type ViT-S_16 --dataset celebA --warmup_steps 100 '
                       f'--num_steps 700 --learning_rate 0.03 --batch_split 16 --img_size 384 --hessian_align --grad_alpha {grad_alpha} --hess_beta {hess_beta} '
                       f'--seed {seed}')
            os.system(train_command)
        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
        eval_command = (f'python evaluate.py --name celeba_hessian --model_arch ViT --model_type ViT-S_16 --dataset celebA --batch_size 64 --img_size 384 '
                        f'--checkpoint_dir output/celeba_hessian/celebA/ViT/ViT-S_16/HessianERM/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/ViT-S_16')
        os.system(eval_command)



def main():
    # erm()
    hessian()

if __name__ == '__main__':
    main()