import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.multiprocessing as mp
from match import get_onehot_matrix, mix_input
from mixup import mixup_process
import numpy as np
import os
from math import ceil


def mixup_process_worker(out: torch.Tensor,
                         target_reweighted: torch.Tensor,
                         hidden=0,
                         args=None,
                         sc: torch.Tensor = None,
                         A_dist: torch.Tensor = None,
                         debug=False):
    """Perform Co-Mixup"""
    m_block_num = args.m_block_num
    n_input = out.shape[0]
    width = out.shape[-1]

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)

    block_size = width // m_block_num

    with torch.no_grad():
        if A_dist is None:
            A_dist = torch.eye(n_input, device=out.device)
        A_base = torch.eye(n_input, device=out.device)

        sc = F.avg_pool2d(sc, block_size)
        sc_norm = sc / sc.view(n_input, -1).sum(1).view(n_input, 1, 1)
        cost_matrix = -sc_norm

        A_dist = A_dist / torch.sum(A_dist) * n_input
        A = (1 - args.m_omega) * A_base + args.m_omega * A_dist

        # Return a batch(partitioned) of mixup labeling
        mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                        A,
                                        n_output=n_input,
                                        beta=args.m_beta,
                                        gamma=args.m_gamma,
                                        eta=args.m_eta,
                                        mixup_alpha=args.mixup_alpha,
                                        thres=args.m_thres,
                                        thres_type=args.m_thres_type,
                                        set_resolve=args.set_resolve,
                                        niter=args.m_niter,
                                        device=out.device)
        # Generate image and corrsponding soft target
        out, target_reweighted = mix_input(mask_onehot, out, target_reweighted)

    return out.contiguous(), target_reweighted


def mixup_process_worker_wrapper(q_input: mp.Queue, q_output: mp.Queue, device: int):
    """
    :param q_input:		input queue
    :param q_output:	output queue
    :param device:		running gpu device
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    print(f"Process generated with cuda:{device}")
    device = torch.device(f"cuda:{device}")
    while True:
        # Get and load on gpu
        out, target_reweighted, hidden, args, sc, A_dist, debug = q_input.get()
        out = out.to(device)
        target_reweighted = target_reweighted.to(device)
        sc = sc.to(device)
        A_dist = A_dist.to(device)

        # Run
        out, target_reweighted = mixup_process_worker(out, target_reweighted, hidden, args, sc,
                                                      A_dist, debug)
        # To cpu and return
        out = out.cpu()
        target_reweighted = target_reweighted.cpu()
        q_output.put([out, target_reweighted])


class MixupProcessWorker:
    def __init__(self, device: int):
        """
        :param device: gpu device id
        """
        self.q_input = mp.Queue()
        self.q_output = mp.Queue()
        self.worker = mp.Process(target=mixup_process_worker_wrapper,
                                 args=[self.q_input, self.q_output, device])
        self.worker.deamon = True
        self.worker.start()

    def start(self,
              out: torch.Tensor,
              target_reweighted: torch.Tensor,
              hidden=0,
              args=None,
              sc: torch.Tensor = None,
              A_dist: torch.Tensor = None,
              debug=True):
        self.q_input.put([out, target_reweighted, hidden, args, sc, A_dist, debug])

    def join(self):
        input, target = self.q_output.get()
        return input, target

    def close(self):
        self.worker.terminate()


class MixupProcessParallel:
    def __init__(self, part, batch_size, num_gpu=1):
        """
        :param part:
        :param batch_size:
        :param num_gpu:
        """
        self.part = part
        self.batch_size = batch_size
        self.n_workers = ceil(batch_size / part)
        self.workers = [MixupProcessWorker(device=i % num_gpu) for i in range(self.n_workers)]

    def __call__(self,
                 out: torch.Tensor,
                 target_reweighted: torch.Tensor,
                 hidden=0,
                 args=None,
                 sc: torch.Tensor = None,
                 A_dist: torch.Tensor = None,
                 debug=False):
        '''
        :param out:					cpu tensor
        :param target_reweighted: 	cpu tensor
        :param hidden:
        :param args:				cpu args
        :param sc: 					cpu tensor
        :param A_dist: 				cpu tensor
        :param debug:
        :return:					out, target_reweighted (cpu tensor)
        '''

        for idx in range(self.n_workers):
            self.workers[idx].start(
                out[idx * self.part:(idx + 1) * self.part].contiguous(),
                target_reweighted[idx * self.part:(idx + 1) * self.part].contiguous(), hidden, args,
                sc[idx * self.part:(idx + 1) * self.part].contiguous(),
                A_dist[idx * self.part:(idx + 1) * self.part,
                       idx * self.part:(idx + 1) * self.part].contiguous(), debug)
        # join
        out_list = []
        target_list = []
        for idx in range(self.n_workers):
            out, target = self.workers[idx].join()
            out_list.append(out)
            target_list.append(target)

        return torch.cat(out_list), torch.cat(target_list)

    def close(self):
        for w in self.workers:
            w.close()


if __name__ == "__main__":
    '''unit test'''
    mp.set_start_method("spawn")

    # inputs (cpu) : out0, target_reweighted0, out, target_reweighted, args, sc, A_dist
    d = torch.load("input.pt")
    out0 = d["out0"]
    target_reweighted0 = d["target_reweighted0"]
    args = d["args"]
    sc = d["sc"]
    A_dist = d["A_dist"]

    # Parallel mixup wrapper
    mpp = MixupProcessParallel(args.m_part, args.batch_size, num_gpu=1)

    # For cuda initialize
    torch.ones(3).cuda()
    for iter in tqdm(range(1), desc="initialize"):
        out, target_reweighted = mpp(out0,
                                     target_reweighted0,
                                     args=args,
                                     sc=sc,
                                     A_dist=A_dist,
                                     debug=True)

    # Parallel run
    for iter in tqdm(range(100), desc="parallel"):
        out, target_reweighted = mpp(out0,
                                     target_reweighted0,
                                     args=args,
                                     sc=sc,
                                     A_dist=A_dist,
                                     debug=True)

    print((d["out"].cpu() == out.cpu()).float().mean())
    print((d["target_reweighted"].cpu() == target_reweighted.cpu()).float().mean())

    # Original run
    out0cuda = out0.cuda()
    target_reweighted0cuda = target_reweighted0.cuda()
    sccuda = sc.cuda()
    A_distcuda = A_dist.cuda()
    for iter in tqdm(range(100), desc="original"):
        out, target_reweighted = mixup_process(out0cuda,
                                               target_reweighted0cuda,
                                               args=args,
                                               sc=sccuda,
                                               A_dist=A_distcuda,
                                               debug=True)

    print((d["out"].cpu() == out.cpu()).float().mean())
    print((d["target_reweighted"].cpu() == target_reweighted.cpu()).float().mean())

    print("end")
