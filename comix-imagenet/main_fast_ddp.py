# This module is adapted from https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/master/main_free.py
# Which in turn was adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
from lib.utils import *
from lib.mixup_parallel import MixupProcessParallel
from lib.validation import validate
import models
from models.imagenet_resnet import BasicBlock, Bottleneck

from apex import amp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def setup(rank, world_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--output_prefix',
                        default='fast_adv',
                        type=str,
                        help='prefix used to define output path')
    parser.add_argument('-c',
                        '--config',
                        default='configs/comix/configs_fast_phase2.yml',
                        type=str,
                        metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--restarts', default=1, type=int)
    return parser.parse_args()
    # return parser.parse_args(r"/home/hosan/data/public/ImageNet-Fast/imagenet-sz/352 --output_prefix debug".split())
    # return parser.parse_args(r"/data_large/readonly/ --output_prefix debug".split())


def main(rank, configs, world_size):
    setup(rank, world_size)

    # Parase config file and initiate logging
    logger = initiate_logger(rank, configs.output_name, configs.evaluate)

    def print(s):
        if rank == 0:
            logger.info(s)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()

    # Scale and initialize the parameters
    best_prec1 = 0

    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))

    # Log the config details
    print(pad_str(' ARGUMENTS '))
    for k, v in configs.items():
        print('{}: {}'.format(k, v))
    print(pad_str(''))

    # Create the model
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch]()

    def init_dist_weights(model):
        for m in model.modules():
            if isinstance(m, BasicBlock):
                m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
            if isinstance(m, Bottleneck):
                m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    init_dist_weights(model)
    model.cuda()

    # reverse mapping
    param_to_moduleName = {}
    for m in model.modules():
        for p in m.parameters(recurse=False):
            param_to_moduleName[p] = str(type(m).__name__)

    group_decay = [
        p for p in model.parameters()
        if 'BatchNorm' not in param_to_moduleName[p]
    ]
    group_no_decay = [
        p for p in model.parameters() if 'BatchNorm' in param_to_moduleName[p]
    ]
    groups = [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=0)
    ]
    optimizer = torch.optim.SGD(groups,
                                0,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    if not configs.evaluate:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level="O1",
                                          loss_scale=1024)

    model = DDP(model, device_ids=[0])

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')  # @debug
    valdir = os.path.join(configs.data, 'val')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(configs.DATA.crop_size,
                                     scale=(configs.DATA.min_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.DATA.batch_size,
        num_workers=configs.DATA.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, test_transform),
        batch_size=configs.DATA.batch_size,
        shuffle=False,
        num_workers=configs.DATA.workers,
        pin_memory=True,
        drop_last=False)

    if configs.evaluate:
        validate(rank, val_loader, model, criterion, configs, logger)
        return

    lr_schedule = lambda t: np.interp([t], configs.TRAIN.lr_epochs, configs.
                                      TRAIN.lr_values)[0]

    # mixup parallel
    mpp = MixupProcessParallel(part=16, num_thread=3)

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        train_sampler.set_epoch(epoch)  # for ddp training

        # train for one epoch
        train(rank, mpp, print, configs, criterion, criterion_batch,
              train_loader, model, optimizer, epoch, lr_schedule)

        # evaluate on validation set
        prec1 = validate(rank, val_loader, model, criterion, configs, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': configs.TRAIN.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best,
                os.path.join('trained_models', f'{configs.output_name}'))
        dist.barrier()
        # break #@debug
    print("end epoch")
    mpp.close()
    print("mpp close")
    cleanup()
    print("end cleanup")


def train(rank, mpp: MixupProcessParallel, print, configs, criterion,
          criterion_batch, train_loader, model, optimizer, epoch, lr_schedule):
    mean = torch.Tensor(
        np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size,
                       configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    param_list = {
        'mixup_alpha': configs.TRAIN.alpha,
        'set_resolve': configs.TRAIN.set_resolve,
        'thres': configs.TRAIN.thres,
        'm_block_num': configs.TRAIN.block_num,
        'lam_dist': configs.TRAIN.lam_dist,
        'm_beta': configs.TRAIN.m_beta
    }

    #pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        # update learning rate
        lr = lr_schedule(epoch + (i + 1) / len(train_loader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        input.sub_(mean).div_(std)
        input_var = Variable(input, requires_grad=True)

        if configs.TRAIN.clean_lam == 0:
            model.eval()

        output = model(input_var)
        if configs.TRAIN.clean_lam > 0:
            loss_clean = configs.TRAIN.clean_lam * criterion(output, target)
        else:
            loss_clean = criterion(output, target)

        with amp.scale_loss(loss_clean, optimizer) as scaled_loss:
            scaled_loss.backward()

        unary = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

        if configs.TRAIN.clean_lam == 0:
            model.train()

        # input = input.detach().cpu()
        target_reweighted = to_onehot(target, 1000)

        # Calculating the distance between most salient regions
        with torch.no_grad():
            z = F.avg_pool2d(unary, kernel_size=8)
            z_reshape = z.reshape(configs.DATA.batch_size, -1)
            z_idx_1d = torch.argmax(z_reshape, dim=1)
            z_idx_2d = torch.zeros(configs.DATA.batch_size, 2)

            z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
            z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]

            A_dist = distance(z_idx_2d, dist_type='l1').cuda()
            
            # parallel
            input, target_reweighted = mpp(input, target_reweighted,
                                           param_list, unary, A_dist)

        output = model(input)

        loss = torch.mean(
            torch.sum(-target_reweighted * nn.LogSoftmax(-1)(output), dim=1))

        # compute gradient and do SGD step
        if configs.TRAIN.clean_lam == 0:
            optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        # ----------------------------  ---------------------------- #
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % configs.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.3f}'.format(epoch,
                                       i,
                                       len(train_loader),
                                       batch_time=batch_time,
                                       data_time=data_time,
                                       top1=top1,
                                       top5=top5,
                                       cls_loss=losses,
                                       lr=lr))

            sys.stdout.flush()


def run(configs):
    world_size = torch.cuda.device_count()
    configs.DATA.batch_size = configs.DATA.batch_size // world_size
    mp.spawn(main, args=[configs, world_size], nprocs=world_size, join=True)


if __name__ == '__main__':
    configs = parse_config_file(parse_args())
    run(configs)
