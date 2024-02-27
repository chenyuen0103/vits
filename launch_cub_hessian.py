import os
import itertools
from tqdm import tqdm


def main():
    seed_list = [0]
    # grad_alpha_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # hess_beta_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9][::-1]
    grad_alpha_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0]
    hess_beta_values = [1, 1e-2, 1e-4, 1e-6, 1e-8, 0][::-1] #to reduce the number of experiments
    grad_alpha_values = [1e-4]
    hess_beta_values = [1e-4]
    for seed, grad_alpha, hess_beta in tqdm(itertools.product(seed_list, grad_alpha_values, hess_beta_values), desc='Training'):
        # if grad_alpha == 1e-4 and hess_beta == 1e-4:
        #     continue
        train_command = (f'python train.py --name waterbirds_hessian --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --warmup_steps 100 '
                   f'--num_steps 700 --learning_rate 0.03 --batch_split 16 --img_size 384 --hessian_align --grad_alpha {grad_alpha} --hess_beta {hess_beta} '
                   f'--seed {seed}')
        os.system(train_command)

        grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
        hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
        eval_command = (f'python evaluate.py --name waterbirds_hessian --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --batch_size 64 --img_size 384 '
                        f'--checkpoint_dir output/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/ViT-S_16')
        os.system(eval_command)
    # for seed, grad_alpha, hess_beta in tqdm(itertools.product(seed_list, grad_alpha_values, hess_beta_values),
    #                                         desc='Evaluation'):
    #     grad_alpha_formatted = "{:.1e}".format(grad_alpha).replace('.0e', 'e')
    #     hess_beta_formatted = "{:.1e}".format(hess_beta).replace('.0e', 'e')
    #     eval_command = (f'python evaluate.py --name waterbirds_hessian --model_arch ViT --model_type ViT-S_16 --dataset waterbirds --batch_size 64 --img_size 384 '
    #                     f'--checkpoint_dir output/waterbirds_hessian/waterbirds/ViT/ViT-S_16/HessianERM/grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{seed}/ViT-S_16')
    #     os.system(eval_command)


if __name__ == '__main__':
    main()