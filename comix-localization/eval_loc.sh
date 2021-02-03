device=${1:-0}

echo $device
# ---------------------------- cmds for localization ---------------------------- #
CUDA_VISIBLE_DEVICES=$device python eval.py --model_name vanilla --data_name clean --th 0.25 --tqdm 0
CUDA_VISIBLE_DEVICES=$device python eval.py --model_name inputmix --data_name clean --th 0.25 --tqdm 0
CUDA_VISIBLE_DEVICES=$device python eval.py --model_name manifold --data_name clean --th 0.25 --tqdm 0
CUDA_VISIBLE_DEVICES=$device python eval.py --model_name cutmix --data_name clean --th 0.25 --tqdm 0
CUDA_VISIBLE_DEVICES=$device python eval.py --model_name puzmix --data_name clean --th 0.25 --tqdm 0
CUDA_VISIBLE_DEVICES=$device python eval.py --model_name comix-blk4 --data_name clean --th 0.25 --tqdm 0
