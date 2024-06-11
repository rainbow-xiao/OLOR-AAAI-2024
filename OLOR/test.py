import os
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from logger import create_logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from dataset import Dataset
from build_model import Model_Cls
from thop import profile
torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--csv-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, default='config')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2023)
    args, _ = parser.parse_known_args()
    config = config_from_name(args.config_name)
    return args, config

def test_epoch(model, test_loader, criterion):
    model.etest()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    bar = tqdm(test_loader)
    end = time.time()
    all_embeds = []
    # all_labels = []
    with torch.no_grad():
        for (images, labels) in bar:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True).long()
            preds = model(images)
            # all_embeds.append(embeds.detach().cpu())
            # all_labels.append(labels.detach().cpu())
            loss = criterion(preds, labels)
            top1, top5 = accuracy(preds, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1_acc.update(top1.item(), images.size(0))
            top5_acc.update(top5.item(), images.size(0))
            bar.set_description('loss_avg: %.5f || top1_avg: %.3f || top5_avg: %.3f' % (losses.avg, top1_acc.avg, top5_acc.avg))
            batch_time.update(time.time() - end)
            end = time.time()
    # all_embeds = torch.cat(all_embeds, dim=0)
    # all_labels = torch.cat(all_labels, dim=0)
    # all = torch.cat([all_embeds, all_labels.unsqueeze(1)], dim=1)
    # np.save(args.model_path.rsplit('.')[0]+'.npy', all.numpy())
    return losses.avg, top1_acc.avg, top5_acc.avg

def main(config):
    df = pd.read_csv(args.csv_dir)
    dataset_test = Dataset(df, None, 'test', config.MODEL.img_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                               shuffle=False, pin_memory=True)
    model = Model_Cls(config)
    dicts = torch.load(args.model_path)['state_dict']
    model.load_state_dict(dicts, strict=True)
    model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    
    logger.info('Results->>>')
    test_loss, test_top1_acc, top5_acc = test_epoch(model, test_loader, criterion)
    logger.info(f"test_loss: {test_loss:.5f}, test_top1_acc: {test_top1_acc:.5f}, test_top5_acc: {test_top5_acc:.5f}")



if __name__ == '__main__':
    args, config = parse_args()
    config.defrost()
    if 'Stanford_Cars' in args.csv_dir:
        config.MODEL.num_classes = 196
    elif 'CUB_200_2011' in args.csv_dir:
        config.MODEL.num_classes = 200
    elif 'Cifar_100' in args.csv_dir:
        config.MODEL.num_classes = 100
    elif 'SVHN' in args.csv_dir:
        config.MODEL.num_classes = 10
    elif 'ip102' in args.csv_dir:
        config.MODEL.num_classes = 102
    elif 'Places_LT' in args.csv_dir:
        config.MODEL.num_classes = 365
    elif 'PACS' in args.csv_dir:
        config.MODEL.num_classes = 7
    elif 'OfficeHome' in args.csv_dir:
        config.MODEL.num_classes = 65


    log_path = args.model_path.rsplit('/', 1)[0]

    config.MODEL.img_size = args.image_size
    config.local_rank=0
    config.batch_size = args.batch_size
    config.freeze()
    set_seed(args.seed)
    os.makedirs(log_path, exist_ok=True)
    logger = create_logger(output_dir=log_path, dist_rank=args.local_rank, name=args.model_path.replace('.pth', '_test'))
    # logger.info(config.dump())
    main(config)

