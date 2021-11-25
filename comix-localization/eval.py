import torch
import torch.utils.data as data

import argparse
from collections import OrderedDict
import re
from tqdm import tqdm

from models import resnet50, get_feature_and_linear_resnet50
from util.all import AverageEpochMeter, CAM, BboxGenerator, Denorm, eval_iou, FixedSubsetSampler
from util.ioparser import mkdir, join
from util.logger import print_args, Logger
from dataset import ImageNetTestWithCenterCropAndNormalize
import config

# ---------------------------- args ---------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir",
    type=str,
    default=None,
    help=
    "result directory. if None, the directory will be ./result/resnet50@{model name}@{dataset name}"
)
parser.add_argument("--model_name",
                    type=str,
                    default="vanilla",
                    choices=["vanilla", "inputmix", "cutmix", "manifold", "puzmix", "comix-blk4"],
                    help="pretrained model name. the corresponding path reside in config.py")
parser.add_argument(
    "--data_name",
    type=str,
    default="clean",
    choices=["clean", "rep", "noise"],
    help="name for clean or corrupted dataset. the corresponding path reside in config.py")
parser.add_argument("--th", type=float, default=0.25, help="CAM threshold. 0~1 value")
parser.add_argument("--debug", type=int, default=0, help="debug flag")
parser.add_argument("--tqdm", type=int, default=0, help="on/off tqdm")

args = parser.parse_args()

# ---------------------------- setup ---------------------------- #
if args.root_dir is None:
    args.root_dir = config.path_result_format.format(args.model_name, args.data_name)
mkdir(args.root_dir)

lg = Logger()
lg.add_console_logger("console")  # add console handler with name console
lg.add_file_logger("file_logger", join(args.root_dir, f"log.txt"))
device = torch.device("cuda:0")

path_model = config.path_model[args.model_name]
path_dataset = config.path_dataset[args.data_name]
path_dataset_clean = config.path_dataset["clean"]

print_args(args, lg)

# ---------------------------- dataset ---------------------------- #
testset = ImageNetTestWithCenterCropAndNormalize(path_dataset)

if args.debug:
    testloader = data.DataLoader(testset,
                                 sampler=FixedSubsetSampler(
                                     torch.arange(0, len(testset),
                                                  len(testset) // 1000)))
else:
    # batch_size should be 1 due to inconsistent bbox_gt.shape
    testloader = data.DataLoader(testset, batch_size=1)

denorm = Denorm()

# ---------------------------- model ---------------------------- #
d = torch.load(path_model)
epoch = d["epoch"]
state_dict = d["state_dict"]
state_dict_cvt = OrderedDict()
for k, v in state_dict.items():
    state_dict_cvt[re.sub(r"module\.(.*)", r"\1", k)] = v
model = resnet50()
model.load_state_dict(state_dict_cvt)
model.to(device)
model = model.eval()

print(f"Model: {args.model_name}/ data: {args.data_name} are loaded!")

# ---------------------------- cam setup ---------------------------- #
feature, linear, factor = get_feature_and_linear_resnet50(model)
camgen = CAM(model, feature, linear, factor, 3, 1)
bboxgen = BboxGenerator(args.th)

# ---------------------------- eval loc ---------------------------- #
top1_meter = AverageEpochMeter("top1")
top5_meter = AverageEpochMeter("top5")
top1_loc_meter = AverageEpochMeter("top1-loc")
pbar = tqdm(testloader, disable=not args.tqdm)
with torch.no_grad():
    # assume batch_size=1 due to inconsistent bbox_gt.shape
    for idx, (input, label, bbox_gt) in enumerate(pbar):
        input = input.to(device)
        label = label.to(device)
        bbox_gt = bbox_gt.to(device)
        out = model(input)

        # classification
        top1 = (out.argmax(1) == label).float().sum().item() * 100
        top5 = (out.topk(5, 1)[1] == label[:, None]).float().sum().item() * 100
        top1_meter.update(top1)
        top5_meter.update(top5)

        # find cam
        cam = camgen(input)

        # find bbox
        bbox = bboxgen(cam)

        # calculate iou
        # we calculate maximum iou value among all ground truth bboxes
        iou = torch.zeros(bbox.shape[1], bbox_gt.shape[1])
        for i in range(bbox.shape[1]):
            for j in range(bbox_gt.shape[1]):
                iou[i, j] = eval_iou(bbox[0, i], bbox_gt[0, j])
        iou = iou.max()

        # calculate top1-localization
        # localization is correct only if iou >= 0.5 and classification correct
        if iou >= 0.5 and top1 == 100:
            top1_loc_meter.update(1)
        else:
            top1_loc_meter.update(0)

        msg = f"[{idx}/50000] [{top1_meter}] [{top5_meter}] [{top1_loc_meter}]"
        pbar.set_description(msg)
        if idx % 500 == 0:
            lg.info(msg)
    lg.info(msg)
    torch.save({
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
        "top1-loc": top1_loc_meter.avg
    }, join(args.root_dir, f"result.pt"))
