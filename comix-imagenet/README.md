# Fast ImageNet training for 100 epochs with ResNet-50
Some parts of the codes are borrowed from ([link](https://github.com/snu-mllab/PuzzleMix/tree/master/imagenet)). Here, **we use Distributed Data Parallel (DDP)** rather than Data Parallel (DP). Note that Co-Mixup contains cpu workloads, and thus the multi-processing is effective. 

## Requirements
* Python 3.7
* PyTorch 1.7.1
* [Apex](https://github.com/NVIDIA/apex) (to use half precision speedup). 


## Preparing ImageNet Data
1. Download and prepare the ImageNet dataset. You can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh), 
provided by the PyTorch repository, to move the validation subset to the labeled subfolders.
2. Prepare resized versions of the ImageNet dataset by running
```
python resize.py
```

## Reproducing the results
To reproduce the results from the paper, modify ```DATA160``` and ```DATA352``` (in run_fast_ddp.sh) to your own ```data path``` made with `resize.py`, and run
```
sh run_fast_ddp.sh
```
This script runs the main code `main_fast_ddp.py` using the configurations provided in the `configs/` folder. All parameters can be modified by adjusting the configuration files in the `configs/` folder. Output log is saved in `output/` folder.
