# AFiRe (AAAI'25)
This repository contains an official implementation of "[AFiRe: Anatomy-Driven Self-Supervised Learning for Fine-Grained Representation in Radiographic Images]" has been accepted by AAAI2025. 

**Our fine-tuning tasks are refer the official code from [here](https://github.com/RL4M/MRM-pytorch).**
----
## How to Perform Pre-train
----
We build our architecture by referring to the released code at [DINO](https://github.com/facebookresearch/dino) and [MAE](https://github.com/facebookresearch/mae).
### Step 1
We use the normal images in  [Johnson et al. MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) for pre-training.
### Step 2
Augmentation images using SLM:
```
python data_prepare.py
```
### Step 3
```
python main_AfiRe.py --nodes 2 --ngpus 8 --arch vit_base --data_path ['/AFiRe/images/inputs', '/AFiRe/images/masks','/AFiRe/images/labels'] --output_dir ./output/pretrain/
```
