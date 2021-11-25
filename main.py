from __future__ import division
import os, sys, shutil, time, random
from posix import CLD_CONTINUED

sys.path.append('..')
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
from collections import OrderedDict
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import models
from load_data import load_data_subset
from logger import plotting, copy_script_to_folder, AverageMeter, RecorderMeter, time_string, convert_secs2time
from utils import to_one_hot, distance
from mixup import mixup_process
from mixup_parallel import MixupProcessParallel

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Train Classifier with mixup',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument('--dataset',
                    type=str,
                    default='cifar100',
                    choices=['cifar10', 'cifar100', 'tiny-imagenet-200'],
                    help='Choose between Cifar10/100 and Tiny-ImageNet.')
parser.add_argument('--data_dir',
                    type=str,
                    default='~/Datasets/cifar100',
                    help='file where results are to be written')
parser.add_argument('--root_dir',
                    type=str,
                    default='experiments',
                    help='folder where results are to be stored')
parser.add_argument('--labels_per_class',
                    type=int,
                    default=500,
                    metavar='NL',
                    help='labels_per_class')
parser.add_argument('--valid_labels_per_class',
                    type=int,
                    default=0,
                    metavar='NL',
                    help='validation labels_per_class')

# Model
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='preactresnet18',
                    choices=model_names,
                    help='model architecture')
parser.add_argument('--initial_channels', type=int, default=64, choices=(16, 64))

# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--dropout',
                    type=str2bool,
                    default=False,
                    help='whether to use dropout or not in final layer')

# Co-Mixup
parser.add_argument('--comix', type=str2bool, default=True, help='true for Co-Mixup')
parser.add_argument('--m_block_num',
                    type=int,
                    default=4,
                    help='resolution of labeling, -1 for random')
parser.add_argument('--m_part', type=int, default=20, help='partition size')
parser.add_argument('--m_beta', type=float, default=0.32, help='label smoothness coef, 0.16~1.0')
parser.add_argument('--m_gamma', type=float, default=1.0, help='supermodular diversity coef')
parser.add_argument('--m_thres',
                    type=float,
                    default=0.83,
                    help='threshold for over-penalization, tau, 0.81~0.86')
parser.add_argument('--m_thres_type',
                    type=str,
                    default='hard',
                    choices=['soft', 'hard'],
                    help='thresholding type')
parser.add_argument('--m_eta', type=float, default=0.05, help='prior coef')
parser.add_argument('--mixup_alpha',
                    type=float,
                    default=2.0,
                    help='alpha parameter for dirichlet prior')
parser.add_argument('--m_omega', type=float, default=0.001, help='input compatibility coef, \omega')
parser.add_argument('--set_resolve',
                    type=str2bool,
                    default=True,
                    help='post-processing for resolving the same outputs')
parser.add_argument('--m_niter', type=int, default=4, help='number of outer iteration')
parser.add_argument('--clean_lam', type=float, default=1.0, help='clean input regularization')
parser.add_argument("--parallel", type=str2bool, default=True, help="mixup_process parallelization")

# training
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=0.0001, help='weight decay (L2 penalty)')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[100, 200],
                    help='decrease learning rate at these epochs')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, help='print frequency (default: 200)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU')
parser.add_argument('--workers',
                    type=int,
                    default=0,
                    help='number of data loader processors. 0 for CIFAR, 8 for Tiny.')

# random seed
parser.add_argument('--seed', default=0, type=int, help='manual seed')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--log_off', action='store_true')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

# random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True


def define_exp_name(args=args):
    '''
    function for experiment result folder name.
    '''
    exp_name = args.dataset
    exp_name += '_per_' + str(args.labels_per_class)
    exp_name += '_arch_' + str(args.arch)
    exp_name += '_eph_' + str(args.epochs)
    exp_name += '_lr_' + str(args.learning_rate)
    if args.comix:
        exp_name += '_mblock_' + str(args.m_block_num) + '_mbeta_' + str(
            args.m_beta) + '_mgamma_' + str(args.m_gamma) + '_mthres_' + str(
                args.m_thres_type) + str(args.m_thres) + '_meta_' + str(
                    args.m_eta) + '_m_alpha_' + str(args.mixup_alpha)
        exp_name += '_mpart_' + str(args.m_part) + '_niter_' + str(args.m_niter) + '_omega_' + str(
            args.m_omega)
        if args.set_resolve:
            exp_name += '_set'
    if args.clean_lam > 0:
        exp_name += '_clean_' + str(args.clean_lam)
    exp_name += '_seed_' + str(args.seed)
    if args.tag != '':
        exp_name += '_' + str(args.tag)

    return exp_name


def print_log(print_string, log, end='\n'):
    '''print log'''
    print("{}".format(print_string), end=end)
    if log is not None:
        if end == '\n':
            log.write('{}\n'.format(print_string))
        else:
            log.write('{} '.format(print_string))
        log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    '''save checkpoint'''
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule)
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


bce_loss = nn.BCELoss().cuda()
bce_loss_sum = nn.BCELoss(reduction='sum').cuda()
softmax = nn.Softmax(dim=1).cuda()
criterion = nn.CrossEntropyLoss().cuda()
criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()


def train(train_loader, model, optimizer, epoch, args, log, mpp=None):
    '''train given model and dataloader'''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mixing_avg = []

    # switch to train mode
    model.train()

    end = time.time()
    for input, target in train_loader:
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        input = input.cuda()
        target = target.long().cuda()
        sc = None

        # train with clean images
        if not args.comix:
            target_reweighted = to_one_hot(target, args.num_classes)
            output = model(input)
            loss = bce_loss(softmax(output), target_reweighted)

        # train with Co-Mixup images
        else:
            input_var = Variable(input, requires_grad=True)
            target_var = Variable(target)
            A_dist = None

            # Calculate saliency (unary)
            if args.clean_lam == 0:
                model.eval()
                output = model(input_var)
                loss_batch = criterion_batch(output, target_var)
            else:
                model.train()
                output = model(input_var)
                loss_batch = 2 * args.clean_lam * criterion_batch(output,
                                                                  target_var) / args.num_classes
            loss_batch_mean = torch.mean(loss_batch, dim=0)
            loss_batch_mean.backward(retain_graph=True)
            sc = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

            # Here, we calculate distance between most salient location (Compatibility)
            # We can try various measurements
            with torch.no_grad():
                z = F.avg_pool2d(sc, kernel_size=8, stride=1)
                z_reshape = z.reshape(args.batch_size, -1)
                z_idx_1d = torch.argmax(z_reshape, dim=1)
                z_idx_2d = torch.zeros((args.batch_size, 2), device=z.device)
                z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
                z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
                A_dist = distance(z_idx_2d, dist_type='l1')

            if args.clean_lam == 0:
                model.train()
                optimizer.zero_grad()

            # Perform mixup and calculate loss
            target_reweighted = to_one_hot(target, args.num_classes)
            if args.parallel:
                device = input.device
                out, target_reweighted = mpp(input.cpu(),
                                             target_reweighted.cpu(),
                                             args=args,
                                             sc=sc.cpu(),
                                             A_dist=A_dist.cpu())
                out = out.to(device)
                target_reweighted = target_reweighted.to(device)

            else:
                out, target_reweighted = mixup_process(input,
                                                       target_reweighted,
                                                       args=args,
                                                       sc=sc,
                                                       A_dist=A_dist)

            out = model(out)
            loss = bce_loss(softmax(out), target_reweighted)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print_log(
        '**Train** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f}'.format(
            top1=top1, top5=top5, error1=100 - top1.avg), log)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, log):
    '''evaluate trained model'''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            input = input.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model(input)
            target_reweighted = to_one_hot(target, args.num_classes)
            loss = bce_loss(softmax(output), target_reweighted)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print_log(
        '**Test ** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Loss: {losses.avg:.3f} '
        .format(top1=top1, top5=top5, error1=100 - top1.avg, losses=losses), log)

    return top1.avg, losses.avg


best_acc = 0


def main():
    # For CUDA multi-processing
    if args.parallel:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")

    # Set up the experiment directories
    if not args.log_off:
        exp_name = define_exp_name()
        exp_dir = os.path.join(args.root_dir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        copy_script_to_folder(os.path.abspath(__file__), exp_dir)
        result_png_path = os.path.join(exp_dir, 'results.png')
        log = open(os.path.join(exp_dir, 'log.txt'.format(args.seed)), 'w')
        print_log('save path : {}'.format(exp_dir), log)
    else:
        log = None
        exp_dir = None
        result_png_path = None

    global best_acc

    state = {k: v for k, v in args._get_kwargs()}
    print("")
    print_log(state, log)
    print("")
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # Dataloader
    train_loader, _, _, test_loader, num_classes = load_data_subset(
        args.batch_size,
        args.workers,
        args.dataset,
        args.data_dir,
        labels_per_class=args.labels_per_class,
        valid_labels_per_class=args.valid_labels_per_class)

    if args.dataset == 'tiny-imagenet-200':
        stride = 2
        args.mean = torch.tensor([0.5] * 3, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        args.std = torch.tensor([0.5] * 3, dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        args.labels_per_class = 500
    elif args.dataset == 'cifar10':
        stride = 1
        args.mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]],
                                 dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        args.std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]],
                                dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        args.labels_per_class = 5000
    elif args.dataset == 'cifar100':
        stride = 1
        args.mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]],
                                 dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        args.std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]],
                                dtype=torch.float32).reshape(1, 3, 1, 1).cuda()
        args.labels_per_class = 500
    else:
        raise AssertionError('Given Dataset is not supported!')

    # Create model
    print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](num_classes, args.dropout, stride).cuda()
    args.num_classes = num_classes

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    optimizer = torch.optim.SGD(list(net.parameters()),
                                state['learning_rate'],
                                momentum=state['momentum'],
                                weight_decay=state['decay'],
                                nesterov=True)

    if args.parallel:
        mpp = MixupProcessParallel(args.m_part, args.batch_size, 1)
    else:
        mpp = None

    recorder = RecorderMeter(args.epochs)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("\n=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']),
                log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, log)
        if args.parallel:
            mpp.close()
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # Train for one epoch
        tr_acc, tr_acc5, tr_los = train(train_loader, net, optimizer, epoch, args, log, mpp)

        # Evaluate on validation set
        val_acc, val_los = validate(test_loader, net, log)

        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(val_los)
        test_acc.append(val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        # Measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if args.log_off:
            continue

        # Save log
        dummy = recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)
        if (epoch + 1) % 100 == 0:
            recorder.plot_curve(result_png_path)

        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc'] = train_acc
        train_log['test_loss'] = test_loss
        train_log['test_acc'] = test_acc

        pickle.dump(train_log, open(os.path.join(exp_dir, 'log.pkl'), 'wb'))
        plotting(exp_dir)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, is_best, exp_dir, 'checkpoint.pth.tar')

    acc_var = np.maximum(
        np.max(test_acc[-10:]) - np.median(test_acc[-10:]),
        np.median(test_acc[-10:]) - np.min(test_acc[-10:]))
    print_log(
        "\nfinal 10 epoch acc (median) : {:.2f} (+- {:.2f})".format(np.median(test_acc[-10:]),
                                                                    acc_var), log)

    if not args.log_off:
        log.close()

    if args.parallel:
        mpp.close()


if __name__ == '__main__':
    main()
