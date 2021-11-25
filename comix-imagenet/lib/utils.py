import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
import numpy as np
import torch.nn.functional as F
import gco


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1**(epoch // int(math.ceil(30. / n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def to_onehot(inp, num_classes, device='cuda'):
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)

    y_onehot.scatter_(1, inp.unsqueeze(1), value=1.)

    return y_onehot


def random_initialize(n_input, n_output, height, width):
    return np.random.randint(0, n_input, (n_output, width, height))


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(rank, output_path, evaluate):
    if not os.path.isdir(os.path.join('output', output_path)) and rank == 0:
        os.makedirs(os.path.join('output', output_path))
    else:
        import time
        while not os.path.isdir(os.path.join('output', output_path)):
            time.sleep(1)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(
            os.path.join('output', output_path, 'eval.txt' if evaluate else 'log.txt'), 'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\


def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v

    # Add the output path
    config.output_name = '{:s}'.format(args.output_prefix)
    return config


def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def distance(z, dist_type='l2'):
    with torch.no_grad():
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        if dist_type[:2] == 'l2':
            A_dist = (diff**2).sum(-1)
            if dist_type == 'l2':
                A_dist = torch.sqrt(A_dist)
            elif dist_type == 'l22':
                pass
        elif dist_type == 'l1':
            A_dist = diff.abs().sum(-1)
        elif dist_type == 'linf':
            A_dist = diff.abs().max(-1)[0]
        else:
            return None
    return A_dist


def mix_input(mask_onehot, input_sp, target_reweighted):
    n_output, height, width, n_input = mask_onehot.shape
    _, n_class = target_reweighted.shape
    mask_onehot_im = F.interpolate(mask_onehot.permute(0, 3, 1, 2),
                                   size=input_sp.shape[-1],
                                   mode='nearest')
    output = torch.sum(mask_onehot_im.unsqueeze(2) * input_sp.unsqueeze(0), dim=1)

    mask_target = torch.matmul(mask_onehot, target_reweighted)
    target = mask_target.reshape(n_output, height * width, n_class).sum(-2) / height / width

    return output, target


def graphcut_multi(cost, beta=1, algorithm='swap', n_label=0, add_idx=None):
    height, width, n_input = cost.shape

    unary = np.ascontiguousarray(cost)

    if add_idx is not None:
        add_idx = add_idx.astype(np.bool)
    pairwise = (np.ones(shape=(n_input, n_input), dtype=np.float32) -
                np.eye(n_input, dtype=np.float32))
    if n_label == 2:
        pairwise[-1, :-1][add_idx] = 0.25
        pairwise[:-1, -1][add_idx] = 0.25
    elif n_label == 3:
        pairwise[-3:, :-3][:, add_idx] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])
        pairwise[:-3, -3:][add_idx, :] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])

    cost_v = beta * np.ones(shape=[height - 1, width], dtype=np.float32)
    cost_h = beta * np.ones(shape=[height, width - 1], dtype=np.float32)

    mask_idx = gco.cut_grid_graph(unary, pairwise, cost_v, cost_h, algorithm='swap')

    return mask_idx


def mixup_match(out, target_reweighted, param_list, sc=None, A_dist=None, device='cuda'):
    mixup_alpha = param_list['mixup_alpha']
    lam_dist = param_list['lam_dist']
    m_part = 16
    m_block_num = param_list['m_block_num']
    m_beta = param_list['m_beta']
    m_gamma = 1
    m_eta = 0.05
    m_thres = param_list['thres']
    m_thres_type = 'hard'
    set_resolve = param_list['set_resolve']
    m_niter = 3

    with torch.no_grad():
        n_input = out.shape[0]
        n_output = n_input
        width = out.shape[-1]

        if A_dist is None:
            A_dist = torch.eye(n_input, device=device)
        A_base = torch.eye(m_part, device=device)

        block_size = width // m_block_num
        sc = F.avg_pool2d(sc, block_size)
        sc_norm = sc / sc.view(n_input, -1).sum(1).view(n_input, 1, 1)
        cost_matrix = -sc_norm

        out_list = []
        target_part_list = []

        for i in range(n_output // m_part):
            A_dist_part = A_dist[i * m_part:(i + 1) * m_part, i * m_part:(i + 1) * m_part]
            A_dist_part = A_dist_part / torch.sum(A_dist_part) * m_part
            A = (1 - lam_dist) * A_base + lam_dist * A_dist_part

            cost_matrix_part = cost_matrix[i * m_part:(i + 1) * m_part]
            mask_onehot = get_onehot_matrix(cost_matrix_part,
                                            A,
                                            m_part,
                                            beta=m_beta,
                                            gamma=m_gamma,
                                            eta=m_eta,
                                            thres=m_thres,
                                            thres_type=m_thres_type,
                                            set_resolve=set_resolve,
                                            niter=m_niter,
                                            device=device)
            output_part, target_part = mix_input(mask_onehot, out[i * m_part:(i + 1) * m_part],
                                                 target_reweighted[i * m_part:(i + 1) * m_part])

            out_list.append(output_part)
            target_part_list.append(target_part)

        out = torch.cat(out_list, dim=0)
        target_reweighted = torch.cat(target_part_list, dim=0)

    return out, target_reweighted


def get_onehot_matrix(cost_matrix,
                      A,
                      n_output,
                      idx=None,
                      beta=1,
                      gamma=1,
                      eta=0.,
                      thres=0.,
                      thres_type='hard',
                      set_resolve=False,
                      niter=2,
                      print_log=False,
                      device='cuda'):
    n_input, height, width = cost_matrix.shape
    thres = thres * height * width
    beta = beta / height / width
    gamma = gamma / height / width
    eta = eta / height / width

    add_cost = None

    # prior
    lam = torch.ones(n_input, device=device)
    alpha = torch.distributions.dirichlet.Dirichlet(lam).sample().reshape(n_input, 1, 1)
    cost_matrix -= eta * torch.log(alpha + 1e-8)

    with torch.no_grad():
        # Init
        if idx is None:
            mask_idx = torch.tensor(random_initialize(n_input, n_output, height, width),
                                    device=device)
            mask_onehot = to_onehot(mask_idx.reshape(-1), n_input,
                                    device=device).reshape([n_output, height, width, n_input])

        loss_prev = obj_fn(cost_matrix, mask_onehot, beta, gamma)
        penalty = to_onehot(mask_idx.reshape(-1), n_input, device=device).sum(0).reshape(-1, 1, 1)

        # main loop
        for iter_idx in range(niter):
            for i in range(n_output):
                label_count = mask_onehot[i].reshape([height * width, n_input]).sum(0)
                penalty -= label_count.reshape(-1, 1, 1)
                if thres_type == 'hard':
                    modular_penalty = (2 * gamma * (
                        (A @ penalty.squeeze() > thres).float() * A @ penalty.squeeze())).reshape(
                            -1, 1, 1)
                elif thres_type == 'soft':
                    modular_penalty = (2 * gamma * ((A @ penalty.squeeze() > thres).float() *
                                                    (A @ penalty.squeeze() - thres))).reshape(
                                                        -1, 1, 1)
                else:
                    raise AssertionError("wrong threshold type!")

                if add_cost is not None:
                    cost_penalty = (cost_matrix + modular_penalty +
                                    gamma * add_cost[i].reshape(-1, 1, 1)).permute(1, 2, 0)
                else:
                    cost_penalty = (cost_matrix + modular_penalty).permute(1, 2, 0)
                mask_onehot[i] = graphcut_wrapper(cost_penalty, label_count, n_input, height, width,
                                                  beta, device, iter_idx)
                penalty += mask_onehot[i].reshape([height * width,
                                                   n_input]).sum(0).reshape(-1, 1, 1)

            if iter_idx == niter - 2 and set_resolve:
                assigned_label_total = (mask_onehot.reshape(n_output, -1, n_input).sum(1) >
                                        0).float()
                add_cost = resolve_label(assigned_label_total, device=device)

            loss = obj_fn(cost_matrix, mask_onehot, beta, gamma)
            if (loss_prev - loss).abs() / loss.abs() < 1e-6:
                break
            loss_prev = loss

    return mask_onehot


def resolve_label(assigned_label_total, device='cuda'):
    n_output, n_input = assigned_label_total.shape
    add_cost = torch.zeros_like(assigned_label_total)

    dist = torch.min(
        (assigned_label_total.unsqueeze(1) - assigned_label_total.unsqueeze(0)).abs().sum(-1),
        torch.tensor(1.0, device=device))
    coincide = torch.triu(1. - dist, diagonal=1)

    for i1, i2 in coincide.nonzero():
        nonzeros = assigned_label_total[i1].nonzero()
        if len(nonzeros) == 1:
            continue
        else:
            add_cost[i1][nonzeros[0]] = 1.
            add_cost[i2][nonzeros[1]] = 1.

    return add_cost


def graphcut_wrapper(cost_penalty, label_count, n_input, height, width, beta, device, iter_idx=0):
    '''Wrapper of graphcut_multi performing efficient extension to multi-label'''
    assigned_label = (label_count > 0)
    if iter_idx > 0:
        n_label = int(assigned_label.float().sum())
    else:
        n_label = 0

    if n_label == 2:
        cost_add = cost_penalty[:, :, assigned_label].mean(-1, keepdim=True) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=2,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 1).reshape(height, width, n_input + 1)

        idx_matrix = torch.zeros([1, 1, n_input], device=device)
        idx_matrix[:, :, assigned_label] = 0.5
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + mask_idx_onehot[:, :,
                                                                          n_input:] * idx_matrix
    elif n_label >= 3:
        soft_label = torch.tensor([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], device=device)

        _, indices = torch.topk(label_count, k=3)
        assigned_label = torch.zeros_like(assigned_label)
        assigned_label[indices] = True

        cost_add = torch.matmul(cost_penalty[:, :, assigned_label], soft_label) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=3,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 3).reshape(height, width, n_input + 3)

        idx_matrix = torch.zeros([3, n_input], device=device)
        idx_matrix[:, assigned_label] = soft_label
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + torch.matmul(
            mask_idx_onehot[:, :, n_input:], idx_matrix)
    else:
        unary = cost_penalty.cpu().numpy()
        mask_idx_np = graphcut_multi(unary, beta=beta, algorithm='swap')
        mask_onehot_i = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                  n_input).reshape(height, width, n_input)

    return mask_onehot_i


def obj_fn(cost_matrix, mask_onehot, beta, gamma):
    n_output, height, width, n_input = mask_onehot.shape
    mask_idx_sum = mask_onehot.reshape(n_output, height * width, n_input).sum(1)

    loss = 0
    loss += torch.sum(cost_matrix.permute(1, 2, 0).unsqueeze(0) * mask_onehot)  # unary
    loss += beta / 2 * (
        (mask_onehot[:, :-1, :, :] - mask_onehot[:, 1:, :, :]).abs().sum() +
        (mask_onehot[:, :, :-1, :] - mask_onehot[:, :, 1:, :]).abs().sum())  # submodular
    loss += gamma * (torch.sum(mask_idx_sum.sum(0)**2) - torch.sum(mask_idx_sum**2))

    return loss
