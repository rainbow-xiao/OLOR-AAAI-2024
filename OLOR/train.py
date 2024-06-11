import os
import time
import argparse
import datetime
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from logger import create_logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils import *
from dataset import Dataset
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from build_model import Model_Cls
import copy
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune-mode', type=str, required=True)
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nbatch_log', type=int, default=500)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def train_epoch(cur_epoch, model, train_loader, optimizer, criterion, criterion_l2_sp, scaler, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    bar = tqdm(train_loader)
    steps = 0
    for (images, labels) in bar:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).long()
        if cur_epoch<=args.warmup_epochs:
            lr = get_warm_up_lr(cur_epoch, steps, args.warmup_epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        else:
            lr = get_train_epoch_lr(cur_epoch, steps, args.epochs, args.init_lr, len(bar))
            set_lr(optimizer, lr)
        with autocast():
            preds = model(images)
            loss = criterion(preds, labels)
            if criterion_l2_sp is not None:
                l2_sp_loss = criterion_l2_sp(model.module.backbone.parameters())
                loss = loss + (0.01 * 0.5) * l2_sp_loss  # alpha = 0.01 is optimal
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        reduced_loss = reduce_tensor(loss)
        losses.update(reduced_loss, images.size(0))
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        bar.set_description('lr: %.6f, loss_cur: %.5f, loss_avg: %.5f' % (lr, losses.val, losses.avg))
        if batch_time.count%args.nbatch_log==0 and args.local_rank==0:
            mu = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info('epoch: %d, iter: [%d/%d] || lr: %.6f, memory_used: %.0fMB, loss_cur: %.5f, loss_avg: %.5f, \
                        time_avg: %.3f, time_total: %.3f' % (cur_epoch, batch_time.count, len(train_loader), lr, mu, losses.val, losses.avg, batch_time.avg, batch_time.sum))
        steps += 1
    return

def val_epoch(model, valid_loader, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    bar = tqdm(valid_loader)
    end = time.time()
    with torch.no_grad():
        for (images, labels) in bar:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).long()
            preds = model(images)
            loss = criterion(preds, labels)
            top1, top5 = accuracy(preds, labels, topk=(1, 5))
            reduced_loss = reduce_tensor(loss)
            reduced_top1 = reduce_tensor(top1)
            reduced_top5 = reduce_tensor(top5)
            losses.update(reduced_loss, images.size(0))
            top1_acc.update(reduced_top1, images.size(0))
            top5_acc.update(reduced_top5, images.size(0))
            bar.set_description('loss_avg: %.5f || top1_avg: %.3f || top5_avg: %.3f' % (losses.avg, top1_acc.avg, top5_acc.avg))
            batch_time.update(time.time() - end)
            end = time.time()
    return losses.avg, top1_acc.avg

def main(config):
    df = pd.read_csv(args.csv_dir)
    dataset_train = Dataset(df, args.val_fold, 'train', config.MODEL.img_size)
    dataset_valid = Dataset(df, args.val_fold, 'valid', config.MODEL.img_size)
    dataset_train_val = Dataset(df, args.val_fold, 'train_val', config.MODEL.img_size)
    train_sampler = DistributedSampler(dataset_train)
    valid_sampler = DistributedSampler(dataset_valid)
    train_val_sampler = DistributedSampler(dataset_train_val)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True, sampler=valid_sampler, drop_last=True)
    train_val_loader = DataLoader(dataset_train_val, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True, sampler=train_val_sampler, drop_last=True)
    model = Model_Cls(config)
    if config.MODEL.finetune != None:
        load_ckpt_finetune(config.MODEL.finetune, model, logger=logger, args=args)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None, find_unused_parameters=True) #find_unused_parameters=True
    model = torch.compile(model)
    optimizer = get_optim_from_config(config, model)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_l2_sp = None
    if args.finetune_mode == 'L2-SP':
        criterion_l2_sp = L2_SP_Loss(copy.deepcopy(model.module.backbone.state_dict()))
    scaler = GradScaler()

    start_time = time.time()
    best_val_top1 = -1
    args.epochs += 1
    for epoch in range(1, args.epochs):
        if args.local_rank==0:
            logger.info(f"----------[Epoch {epoch}]----------")   
        train_sampler.set_epoch(epoch)
        train_epoch(epoch, model, train_loader, optimizer, criterion, criterion_l2_sp, scaler, args)
        train_loss, train_top1 = val_epoch(model, train_val_loader, criterion)
        val_loss, val_top1 = val_epoch(model, valid_loader, criterion)
        if args.local_rank==0:
            logger.info(f"epoch: {epoch} || loss_train: {train_loss:.5f}, train_top1: {train_top1:.5f}, loss_val: {val_loss:.5f}, val_top1: {val_top1:.5f}")
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_best.pth')
                save_checkpoint(model, save_path)
                logger.info(f"Save best model to {save_path}, with best top1: {best_val_top1}")
            logger.info(f'Epoch {epoch} time cost: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}')
    if args.local_rank==0:
        logger.info(f"Best top1_acc: {best_val_top1}")
        save_path = os.path.join(config.MODEL.output_dir, f'{config.MODEL.backbone.model_name}_best.pth')
        save_checkpoint(model, save_path)

if __name__ == '__main__':
    args, config = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    config.defrost()
    config.MODEL.ft_mode = args.finetune_mode
    config.MODEL.type = args.model_type
    config.OOD = False
    if 'Stanford_Cars' in args.csv_dir:
        config.MODEL.num_classes = 196
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/Stanford_Cars'
    elif 'CUB_200_2011' in args.csv_dir:
        config.MODEL.num_classes = 200
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/CUB_200_2011'
    elif 'Cifar_100' in args.csv_dir:
        config.MODEL.num_classes = 100
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/Cifar_100'
    elif 'SVHN' in args.csv_dir:
        config.MODEL.num_classes = 10
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/SVHN'
    elif 'ip102' in args.csv_dir:
        config.OOD = True
        config.MODEL.num_classes = 102
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/ip102'
    elif 'Places_LT' in args.csv_dir:
        config.OOD = True
        config.MODEL.num_classes = 365
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/Places_LT'
    elif 'PACS' in args.csv_dir:
        config.OOD = True
        config.MODEL.num_classes = 7
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/PACS'
    elif 'OfficeHome' in args.csv_dir:
        config.OOD = True
        config.MODEL.num_classes = 65
        config.MODEL.output_dir = config.MODEL.output_dir + '/root/autodl-tmp/output/OfficeHome'


    if args.model_type == 'conv':
        config.MODEL.output_dir = config.MODEL.output_dir + '/conv'
    elif args.model_type == 'vit':
        config.MODEL.output_dir = config.MODEL.output_dir + '/vit'

    if args.finetune_mode == 'pretrain':
        config.MODEL.finetune = None
        config.MODEL.output_dir = config.MODEL.output_dir + '-pretrain'
    elif args.finetune_mode == 'linear':
        config.MODEL.output_dir = config.MODEL.output_dir + '-linear'
    elif args.finetune_mode == 'full':
        config.MODEL.output_dir = config.MODEL.output_dir + '-full'
    elif args.finetune_mode == 'vpt':
        config.MODEL.output_dir = config.MODEL.output_dir + '-vpt'
    elif args.finetune_mode == 'L2-SP':
        config.MODEL.output_dir = config.MODEL.output_dir + '-L2-SP'
    elif args.finetune_mode == 'SGDB':
        config.Optimizer.name = 'SGDB'
        config.MODEL.output_dir = config.MODEL.output_dir + '-SGDB'
    elif args.finetune_mode == 'AdamB':
        config.Optimizer.name = 'AdamB'
        config.MODEL.output_dir = config.MODEL.output_dir + '-AdamB'

    config.MODEL.img_size = args.image_size
    config.init_lr = args.init_lr
    config.Optimizer.back_level_max /= args.init_lr
    config.batch_size = args.batch_size
    config.local_rank = args.local_rank
    config.world_size = args.world_size
    config.freeze()

    set_seed(args.seed)
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    os.makedirs(config.MODEL.output_dir, exist_ok=True)
    logger = create_logger(output_dir=config.MODEL.output_dir, dist_rank=args.local_rank, name=f"{config.MODEL.backbone.model_name}")
    if args.local_rank==0:
        logger.info(config.dump())
    main(config)

