import os
base_path = './'

path_model = {
    "vanilla": os.path.join(base_path, "pretrained/vanilla.pt"),
    "inputmix": os.path.join(base_path, "pretrained/inputmix.pt"),
    "manifold": os.path.join(base_path, "pretrained/manifold.pt"),
    "cutmix": os.path.join(base_path, "pretrained/cutmix.pt"),
    "puzmix": os.path.join(base_path, "pretrained/puzmix.pt"),
    "comix-blk4": os.path.join(base_path, "pretrained/comix-blk4.pt"),
}

path_dataset = {
    "clean": os.path.join(base_path, "data/clean"),
    "rep": os.path.join(base_path, "data/rep"),
    "noise": os.path.join(base_path, "data/noise"),
}

path_result_format = "./result/resnet50@{}@{}"
