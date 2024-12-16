# AFiRe (AAAI'25)
This repository contains an official implementation of "[AFiRe: Anatomy-Driven Self-Supervised Learning for Fine-Grained Representation in Radiographic Images]" accepted by AAAI2025. 

## How to Perform Fine-tune
---
Download AFiRe's [pre-trained weight](https://drive.google.com/file/d/1VeMGrW2m6p-y5z2k55Jd862RtptFoaav/view?usp=sharing).
Load the weights to the ViT-B model:
```
weights_dict = torch.load("./AFiRe_ViT-B.pth", map_location=torch.device('cuda'))['teacher']
for key in list(weights_dict.keys()):
    new_key = key.replace('encoder.', 'module.')
    weights_dict[new_key] = weights_dict.pop(key)
model.load_state_dict(weights_dict, strict=False)
```
Our fine-tuning classification tasks are referred to the official code from [here](https://github.com/funnyzhou/REFERS) and segmentation tasks are referred to [here](https://github.com/SZUHvern/MaCo).
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
We provide some generated samples in the [images.zip](https://github.com/LYH-hh/AFiRe/blob/main/images.zip)
### Step 3
```
python main_AfiRe.py --nodes 2 --ngpus 8 --arch vit_base --data_path ['/AFiRe/images/inputs', '/AFiRe/images/masks','/AFiRe/images/labels'] --output_dir ./output/pretrain/
```
