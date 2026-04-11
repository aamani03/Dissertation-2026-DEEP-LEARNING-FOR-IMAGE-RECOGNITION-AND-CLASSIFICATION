# MNIST Dataset

## Overview
- **Classes:** 10 (digits 0–9)
- **Training images:** 60,000
- **Test images:** 10,000
- **Image size:** 28×28 grayscale
- **File size:** ~30 MB

## Download

The dataset is automatically downloaded via PyTorch:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
```

## Manual Download
- Source: [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
- PyTorch mirror: Downloaded automatically by `torchvision.datasets.MNIST`

## Experiment Parameters
- **Condition 1 (CNN Baseline):** Trained on original 60,000 training images
- **Condition 2 (CNN+GAN):** Trained on 60,000 original + 60,000 DCGAN-generated synthetic images
- **Runs per condition:** 105
- **Total runs:** 210
