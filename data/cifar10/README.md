# CIFAR-10 Dataset

## Overview
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training images:** 50,000
- **Test images:** 10,000
- **Image size:** 32×32 RGB
- **File size:** ~170 MB

## Download

The dataset is automatically downloaded via PyTorch:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

## Manual Download
- Source: [Alex Krizhevsky's CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch mirror: Downloaded automatically by `torchvision.datasets.CIFAR10`

## Experiment Parameters
- **Condition 1 (CNN Baseline):** Trained on original 50,000 training images
- **Condition 2 (CNN+GAN):** Trained on 50,000 original + 50,000 DCGAN-generated synthetic images
- **Runs per condition:** 105
- **Total runs:** 210
