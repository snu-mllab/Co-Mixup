import torch
from config import path_model, path_result_format
from util.ioparser import join

print("** result localization **")
print("{:10s}\t{:8s}\t{:8s}\t{:8s}".format("model name", "cls-top1", "cls-top5", "loc-top1"))
for model_name in path_model:
    try:
        d = torch.load(join(path_result_format.format(model_name, "clean"), "result.pt"))
        print("{:10s}\t{:8.2f}\t{:8.2f}\t{:8.2f}".format(model_name, d["top1"], d["top5"],
                                                         d["top1-loc"] * 100))
    except:
        print("{:10s}\t-\t-\t-".format(model_name))
print()

print("** result robust (image replacement dataset) **")
print("{:10s}\t{}\t{}\t{}".format("model name", "cls-top1", "cls-top5", "loc-top1"))
for model_name in path_model:
    try:
        d = torch.load(join(path_result_format.format(model_name, "rep"), "result.pt"))
        print("{:10s}\t{:8.2f}\t{:8.2f}\t{:8.2f}".format(model_name, d["top1"], d["top5"],
                                                         d["top1-loc"] * 100))
    except:
        print("{:10s}\t-\t-\t-".format(model_name))
print()

print("** result robust (noise dataset) **")
print("{:10s}\t{}\t{}\t{}".format("model name", "cls-top1", "cls-top5", "loc-top1"))
for model_name in path_model:
    try:
        d = torch.load(join(path_result_format.format(model_name, "noise"), "result.pt"))
        print("{:10s}\t{:8.2f}\t{:8.2f}\t{:8.2f}".format(model_name, d["top1"], d["top5"],
                                                         d["top1-loc"] * 100))
    except:
        print("{:10s}\t-\t-\t-".format(model_name))
print()
