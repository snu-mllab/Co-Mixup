import torch
from tqdm import tqdm
import torch.multiprocessing as mp
# from lib.utils import *
from lib.utils import mixup_match
import os
import typing


def mixup_process_worker_wrapper(q_input: mp.Queue, q_output: mp.Queue):
    """
	:param q_input:		input queue
	:param q_output:	output queue
	:param device:		running gpu device
	"""
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}" # not to call torch.cuda initializer in device-0
    # print(f"cuda visible devices = {device}")
    # device = torch.device(f"cuda:0")
    while True:
        # get args
        key, out, target_reweighted, param_list, sc, A_dist, device = q_input.get(
        )

        # run
        out, target_reweighted = mixup_match(out, target_reweighted,
                                             param_list, sc, A_dist, device)

        # return args
        q_output.put([key, out, target_reweighted])


class MixupProcessWorker:
    def __init__(self):
        """
		:param device: gpu device id
		"""
        self.q_input = mp.Queue()
        self.q_output = mp.Queue()
        self.worker = mp.Process(target=mixup_process_worker_wrapper,
                                 args=[self.q_input, self.q_output])
        self.worker.deamon = True
        self.worker.start()

    def start(self,
              key,
              out: torch.Tensor,
              target_reweighted: torch.Tensor,
              param_list: typing.Dict,
              sc: torch.Tensor = None,
              A_dist: torch.Tensor = None,
              device="cuda"):
        self.q_input.put(
            [key, out, target_reweighted, param_list, sc, A_dist, device])

    def join(self):
        key, input, target = self.q_output.get()
        return key, input, target

    def close(self):
        self.worker.terminate()


class MixupProcessParallel:
    def __init__(self, part, num_thread):
        """
		:param part:
		:param batch_size:
		:param num_thread:
		"""
        self.part = part
        self.num_thread = num_thread
        self.workers = [MixupProcessWorker() for i in range(num_thread)]

    def __call__(self,
                 out: torch.Tensor,
                 target_reweighted: torch.Tensor,
                 param_list: typing.Dict,
                 sc: torch.Tensor = None,
                 A_dist: torch.Tensor = None,
                 device="cuda") -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
		:param out:					gpu tensor
		:param target_reweighted: 	gpu tensor
		:param param_list:
		:param sc: 					gpu tensor
		:param A_dist: 				gpu tensor
		:param device:
		:return:					out, target_reweighted (gpu tensor)
		"""
        assert out.shape[0] == target_reweighted.shape[0] == sc.shape[0]
        assert out.shape[0] % self.part == 0
        batch_size = out.shape[0]

        # start
        num_chunk = batch_size // self.part
        for idx in range(num_chunk):
            self.workers[idx % self.num_thread].start(
                idx, out[idx * self.part:(idx + 1) * self.part].contiguous(),
                target_reweighted[idx * self.part:(idx + 1) *
                                  self.part].contiguous(), param_list,
                sc[idx * self.part:(idx + 1) * self.part].contiguous(),
                A_dist[idx * self.part:(idx + 1) * self.part, idx *
                       self.part:(idx + 1) * self.part].contiguous(), device)
        # join
        out_list = [None] * num_chunk
        target_list = [None] * num_chunk
        for idx in range(num_chunk):
            key, out, target = self.workers[idx % self.num_thread].join()
            out_list[key] = out
            target_list[key] = target
        return torch.cat(out_list), torch.cat(target_list)

    def close(self):
        for w in self.workers:
            w.close()


# ======================================================== #
# unit test
# ======================================================== #
if __name__ == "__main__":
    mp.set_start_method("spawn")  # for cuda use
    # inputs (cpu)
    # d = torch.load("input.pt") # out0, target_reweighted0, out, target_reweighted, args, sc, A_dist
    # out0 = d["out0"] # cpu
    # target_reweighted0 = d["target_reweighted0"] # cpu
    # args = d["args"] # cpu
    # sc = d["sc"] # cpu
    # A_dist = d["A_dist"] # cpu

    # inputs (gpu)
    # d = torch.load("input.pt")  # out0, target_reweighted0, out, target_reweighted, args, sc, A_dist
    # out0 = d["out0"].cuda()
    # target_reweighted0 = d["target_reweighted0"].cuda()
    # args = d["args"]
    # sc = d["sc"].cuda()
    # A_dist = d["A_dist"].cuda()

    # saved input
    d = torch.load("input-imagenet.pt")
    out0 = d["input0"].cuda()
    target_reweighted0 = d["target_reweighted0"].cuda()
    param_list = d["param_list"]
    sc = d["sc"].cuda()
    A_dist = d["A_dist"].cuda()

    # parallel mixup wrapper
    mpp = MixupProcessParallel(part=16, batch_size=112, num_thread=3)

    # for cuda initialize
    torch.ones(3).cuda()
    for iter in tqdm(range(1), desc="initialize"):
        out, target_reweighted = mpp(out0,
                                     target_reweighted0,
                                     param_list,
                                     sc=sc,
                                     A_dist=A_dist)

    # parallel run
    for iter in tqdm(range(100), desc="parallel"):
        out, target_reweighted = mpp(out0,
                                     target_reweighted0,
                                     param_list,
                                     sc=sc,
                                     A_dist=A_dist)

    # chk sanity
    print((d["input"].cpu() == out.cpu()).float().mean())
    print((d["target_reweighted"].cpu() == target_reweighted.cpu()
           ).float().mean())

    # original run
    for iter in tqdm(range(100), desc="original"):
        out, target_reweighted = mixup_match(out0,
                                             target_reweighted0,
                                             param_list,
                                             sc=sc,
                                             A_dist=A_dist)

    # chk sanity
    print((d["input"].cpu() == out.cpu()).float().mean())
    print((d["target_reweighted"].cpu() == target_reweighted.cpu()
           ).float().mean())

    print("end")

    # # imshow
    # from pc7.util.all import *
    # imshowc(out)
