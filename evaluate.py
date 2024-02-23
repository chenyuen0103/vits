import logging
import argparse
import numpy as np
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.parallel
from evaluation_utils.evaluate_acc import calculate_acc
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="help identify checkpoint")
    parser.add_argument("--dataset", choices=["waterbirds","cmnist","celebA"], default="waterbirds",
                        help="Which downstream task.")
    parser.add_argument("--model_arch", choices=["ViT", "BiT"],
                        default="ViT",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint_dir",
                        help="directory of saved model checkpoint")
    parser.add_argument("--model_type", default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The directory where checkpoints are stored.")
    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--hessian_align', default=False, action='store_true')
    parser.add_argument('--run_name', default=None, type=str)
    
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    calculate_acc(args)

if __name__=="__main__":
    main()