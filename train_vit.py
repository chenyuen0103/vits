import os
import random
import numpy as np
from itertools import chain

from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader_train
from utils.dist_util import get_world_size
from utils.loss_utils import LossComputer
import timm
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.comm_utils import AverageMeter
import math
import logging
import csv
from utils.train_util import CSVBatchLogger, set_seed, log_args, Logger
from utils.comm_utils import accuracy_func as accuracy_fn

model_dict = {'ViT-B_16':'vit_base_patch16_224_in21k', 
'ViT-S_16':'vit_small_patch16_224_in21k',
'ViT-Ti_16':'vit_tiny_patch16_224_in21k'}


def save_model(args, model,logger, save_dir = None):
    model_to_save = model.module if hasattr(model, 'module') else model
    if args.hessian_align:
        algo = "HessianERM"
    else:
        algo = "ERM"

    grad_alpha_formatted = "{:.1e}".format(args.grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(args.hess_beta).replace('.0e', 'e')
    if save_dir is None:
        model_checkpoint_dir = os.path.join(args.output_dir, args.name, args.dataset, args.model_arch, args.model_type, algo, f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{args.seed}")
    else:
        model_checkpoint_dir = save_dir
    checkpoint_path = os.path.join(model_checkpoint_dir,args.model_type + ".bin")
    if os.path.exists(checkpoint_path) != True:
         os.makedirs(model_checkpoint_dir, exist_ok=True)
    breakpoint()
    torch.save(model_to_save.state_dict(), checkpoint_path)
    logger.write("Saved model checkpoint at %s" % checkpoint_path)


def setup(args, logger):
    num_classes = 2
    model_name = model_dict[args.model_type]
    if args.resume:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate = 0.1,
            img_size = args.img_size
        )
        model.reset_classifier(num_classes)
        model.to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.model_type + ".bin")))
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate = 0.1,
            img_size = args.img_size
        )
        model.reset_classifier(num_classes)
        model.to(args.device)
        num_params = count_parameters(model)
        logger.write("Training parameters %s" % args)
        logger.write("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def valid(args, model, writer, logger, val_csv_logger, testset, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.write("***** Running Validation *****")
    logger.write(f"  Num steps = {len(test_loader)}")
    logger.write(f"  Batch size = {args.eval_batch_size}")



    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()

    val_loss_computer = LossComputer(
        loss_fct,
        is_robust=False,
        dataset=testset)

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y, env = batch;

        with torch.no_grad():
            # logits = model(x)
            outputs = model.forward_features(x)
            features = model.forward_head(outputs, pre_logits=True)
            logits = model.head(features)

            if args.hessian_align:
                eval_loss,_,_,_ = val_loss_computer.exact_hessian_loss(logits, features, y, env, grad_alpha = args.grad_alpha, hess_beta=args.hess_beta)
                eval_losses.update(eval_loss.item())
            else:
                eval_loss = loss_fct(logits, y)
                val_loss_computer.loss(logits, y, env)
                eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
        val_csv_logger.log(global_step, step, val_loss_computer.get_stats(model, args))
        val_csv_logger.flush()
        val_loss_computer.log_stats(logger, False)

    curr_val_acc = min(val_loss_computer.avg_group_acc)
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = accuracy_fn(all_preds, all_label)

    logger.write("\n")
    logger.write("Validation Results\n")
    logger.write("Global Steps: %d \t" % global_step)
    logger.write("Valid Loss: %2.5f \t" % eval_losses.avg)
    logger.write("Average Valid Accuracy: %2.5f \t" % accuracy)
    logger.write(f'worst-group validation accuracy: {curr_val_acc}\n')

    writer.add_scalar("val/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train_model(args):
    if args.hessian_align:
        algo = "HessianERM"
    else:
        algo = "ERM"

    grad_alpha_formatted = "{:.1e}".format(args.grad_alpha).replace('.0e', 'e')
    hess_beta_formatted = "{:.1e}".format(args.hess_beta).replace('.0e', 'e')

    log_dir = os.path.join("logs", args.name, args.dataset, args.model_arch, args.model_type, algo,
                 f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{args.seed}")



    model_dir = os.path.join(args.output_dir, args.name, args.dataset, args.model_arch, args.model_type, algo,f"grad_alpha_{grad_alpha_formatted}_hess_beta_{hess_beta_formatted}/s{args.seed}")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(log_dir) and args.resume:
        resume = True
        mode = 'a'
    else:
        resume = False
        mode = 'w'


    logger = Logger(os.path.join(log_dir, 'log.txt'), mode)
    logger.write(f"Fine-tuning {args.model_type} on {args.dataset}")
    args, model = setup(args, logger)




    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=log_dir,)


    args.train_batch_size = args.train_batch_size // args.batch_split
    trainset, train_loader, testset, test_loader = get_loader_train(args)

    # Record args
    log_args(args, logger)
    train_csv_logger = CSVBatchLogger(csv_path=os.path.join(log_dir, "train.csv"),
                                      n_groups=trainset.n_groups,
                                      mode=mode)
    val_csv_logger = CSVBatchLogger(csv_path=os.path.join(log_dir, "val.csv"), n_groups=trainset.n_groups,
                                    mode=mode)
    cri = torch.nn.CrossEntropyLoss().to(args.device)

    train_loss_computer = LossComputer(
        cri,
        is_robust=False,
        dataset=trainset)


    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.write("***** Running training *****")
    logger.write("  Total optimization steps = %d" %args.num_steps)
    logger.write("  Instantaneous batch size per GPU = %d" %args.train_batch_size)
    logger.write("  Total train batch size (w. parallel, distributed & accumulation) = %d" %
                args.train_batch_size * args.batch_split * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.write("  Gradient Accumulation steps = %d"  %args.batch_split)

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y, env = batch;
            outputs = model.forward_features(x)
            features = model.forward_head(outputs, pre_logits=True)
            logits = model.head(features)
            # logits = model(x)
            if args.hessian_align:
                loss, erm, accum_hess_loss, accum_grad_loss = train_loss_computer.exact_hessian_loss(logits.view(-1, 2),features, y.view(-1), env, grad_alpha = args.grad_alpha, hess_beta=args.hess_beta)
            else:
                loss = cri(logits.view(-1, 2), y.view(-1))
                train_loss_computer.loss(logits, y, env)

            if args.batch_split > 1:
                loss = loss / args.batch_split
        
            loss.backward()

            if (step + 1) % args.batch_split == 0:
                losses.update(loss.item()*args.batch_split)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                train_csv_logger.log(global_step, step, train_loss_computer.get_stats(model, args))
                train_csv_logger.flush()
                logger.write("Step: %d \t" % global_step)
                train_loss_computer.log_stats(logger, is_training=True)
                train_loss_computer.reset_stats()

                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)


                if (global_step % args.eval_every == 0 and args.local_rank in [-1, 0]) or global_step == 1:
                    logger.write("Validate at step: %d" % global_step)
                    accuracy = valid(args, model, writer, logger, val_csv_logger, testset,test_loader, global_step)
#                     if best_acc < accuracy:
#                         save_model(args, model)
#                         best_acc = accuracy
                    model.train()
                    save_model(args, model, logger, save_dir = model_dir)
                    logger.write("Model saved at step: %d \n" % global_step)

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
    save_model(args, model, logger)
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.write("Best Accuracy: \t%f" % best_acc)
    logger.write("End Training!")

