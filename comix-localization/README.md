## Co-Mixup localization and background robustness experiments

## Requirements
* Python 3.7
* PyTorch 1.5.1  

(The code works on other versions, but the evaluation results may vary about <0.1%)

## How to reproduce
1. Download **pre-trained** model and **clean/robust datasets**. 
    * pre-trained models (ResNet-50, 200MB each)
    ```
    sh download_model.sh
    
    => 
    "Vanilla": "./pretrained/vanilla.pt"    
    "Input mix": "./pretrained/inputmix.pt"
    "CutMix": "./pretrained/cutmix.pt"
    "Manifold": "./pretrained/manifold.pt"
    "Puzzle Mix": "./pretrained/puzmix.pt"
    "Co-Mixup": "./pretrained/comix-blk4.pt"
    ```
    
    * datasets (about 10GB each)
    ```
    cd data
    sh download_data.sh all
    
    => 
    "Clean": "./data/clean"
    "Gaussian noise": "./data/noise"    
    "Replacement": "./data/rep"
    ```
    
    You can also download each dataset separately by running 
    ```
    sh download_data.sh clean
    sh download_data.sh noise
    sh download_data.sh rep    
    ```
    
    * If you choose another path to save the models and the datasets, you should modify `config.py` accordingly.
    
3. Run evaluation according to the following commands.

   To reproduce **all experiment** including localization and robustness experiments,
   ```
   sh ./eval_all.sh
   ```
   Note, this will run on background using 3 gpus. You can perform each of experiments separately in foreground by running the following commands. 

   To reproduce **localization**,
   ```
   sh ./eval_loc.sh
   ```

   To reproduce **robustness on replacement dataset**,
   ```
   sh ./eval_robust_rep.sh
   ```

   To reproduce **robustness on noisy dataset**,
   ```
   sh ./eval_robust_noise.sh
   ```

4. Run parsing code using `python parse_result.py` to print the result.
