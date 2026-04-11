# G*Power Statistical Power Analysis

## Parameters

| Parameter | Value |
|-----------|-------|
| Test family | t tests |
| Statistical test | Means: Difference between two independent means (two groups) |
| Type of power analysis | A priori: Compute required sample size |
| Tail(s) | Two |
| Effect size d | 0.5 (medium) |
| α err prob | 0.05 |
| Power (1-β err prob) | 0.95 |
| Allocation ratio N2/N1 | 1 |

## Results

| Output | Value |
|--------|-------|
| Noncentrality parameter δ | 3.6228442 |
| Critical t | 1.9837468 |
| Df | 208 |
| **Sample size group 1** | **105** |
| **Sample size group 2** | **105** |
| **Total sample size** | **210** |
| **Actual power** | **0.9501** |

## Application

Each dataset (MNIST, CIFAR-10, Tiny ImageNet) required 105 runs per condition (CNN baseline and CNN+GAN augmented), totaling 210 runs per dataset and 630 runs across all three datasets.

This design achieved statistical power of 0.95 for detecting medium effects (Cohen's d = 0.5) at α = .05, exceeding the conventional threshold of 0.80 recommended by Cohen (1988).
