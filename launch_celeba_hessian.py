import os
import itertools
from tqdm import tqdm



def main():
    seed_list = [0]
    grad_alpha_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    hess_beta_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    for seed, grad_alpha, hess_beta in tqdm(itertools.product(seed_list, grad_alpha_values, hess_beta_values), desc='Training'):
        train_command = (f'python train.py --name celeba_hessian --model_arch ViT --model_type ViT-S_16 --dataset celebA --warmup_steps 100 '
                   f'--num_steps 700 --learning_rate 0.03 --batch_split 16 --img_size 384 --grad_alpha {grad_alpha} --hess_beta {hess_beta} '
                   f'--seed {seed}')
        os.system(train_command)

    for seed, grad_alpha, hess_beta in tqdm(itertools.product(seed_list, grad_alpha_values, hess_beta_values),
                                            desc='Evaluation'):
        eval_command = (f'python evaluate.py --name celeba_hessian --model_arch ViT --model_type ViT-S_16 --dataset celebA --batch_size 64 --img_size 384 '
                        f'--checkpoint_dir output/waterbirds_hessian/waterbirds/ViT/ViT-S_16')