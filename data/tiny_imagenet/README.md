# Tiny ImageNet Dataset

## Overview
- **Classes:** 200 (subset of ImageNet)
- **Training images:** 100,000 (500 per class)
- **Validation images:** 10,000 (50 per class)
- **Test images:** 10,000 (no labels provided)
- **Image size:** 64×64 RGB
- **File size:** ~400 MB

## Download

Tiny ImageNet is not included in torchvision by default. Download manually:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

Then load with PyTorch using ImageFolder:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
])

train_dataset = datasets.ImageFolder(root='./tiny-imagenet-200/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./tiny-imagenet-200/val', transform=transform)
```

**Note:** The validation folder requires restructuring into class subfolders. Use the `val_annotations.txt` file to organize images by class:

```python
import os
import shutil

val_dir = './tiny-imagenet-200/val'
annotations = os.path.join(val_dir, 'val_annotations.txt')

with open(annotations, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        img_name, class_id = parts[0], parts[1]
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(val_dir, 'images', img_name)
        dst = os.path.join(class_dir, img_name)
        if os.path.exists(src):
            shutil.move(src, dst)
```

## Manual Download
- Source: [Stanford CS231N](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- Original ImageNet: [image-net.org](https://image-net.org/)

## Experiment Parameters
- **Condition 1 (CNN Baseline):** Trained on original 100,000 training images
- **Condition 2 (CNN+GAN):** Trained on 100,000 original + 100,000 DCGAN-generated synthetic images
- **Runs per condition:** 105
- **Total runs:** 210
