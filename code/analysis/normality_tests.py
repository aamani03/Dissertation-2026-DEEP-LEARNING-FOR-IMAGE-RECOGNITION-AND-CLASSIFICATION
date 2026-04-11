"""
Normality Assessment for Dissertation
Deep Learning for Image Recognition and Classification

Generates:
- Shapiro-Wilk test results for all conditions
- Q-Q plots for each dataset/condition
- Histograms with normal curve overlay
- Summary table of normality evidence

Usage: python normality_tests.py
"""

import json
import glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results(dataset):
    """Load all JSON result files for a dataset."""
    pattern = os.path.join(RESULTS_DIR, dataset, '*.json')
    results = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def generate_qq_plots():
    """Generate Q-Q plots for all datasets and conditions."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle('Q-Q Plots: Normality Assessment of F1-Score Distributions',
                 fontsize=14, fontweight='bold', y=0.98)

    datasets = [
        ('mnist', 'MNIST'),
        ('cifar10', 'CIFAR-10'),
        ('tiny_imagenet', 'Tiny ImageNet')
    ]

    for row, (dataset, label) in enumerate(datasets):
        results = load_results(dataset)
        cnn = [r for r in results if r['model_type'] == 'cnn']
        gan = [r for r in results if r['model_type'] == 'cnn_gan']

        cnn_f1 = np.array([r['f1_macro'] for r in cnn])
        gan_f1 = np.array([r['f1_macro'] for r in gan])

        # CNN Q-Q plot
        ax = axes[row, 0]
        stats.probplot(cnn_f1, dist="norm", plot=ax)
        w, p = stats.shapiro(cnn_f1)
        ax.set_title(f'{label} — CNN Baseline\nShapiro-Wilk: W={w:.3f}, p={p:.3f}',
                      fontsize=11)
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')

        # CNN+GAN Q-Q plot
        ax = axes[row, 1]
        stats.probplot(gan_f1, dist="norm", plot=ax)
        w, p = stats.shapiro(gan_f1)
        color = 'red' if p < 0.05 else 'black'
        ax.set_title(f'{label} — CNN+GAN Augmented\nShapiro-Wilk: W={w:.3f}, p={p:.3f}',
                      fontsize=11, color=color)
        ax.get_lines()[0].set_markerfacecolor('darkorange')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGURES_DIR, 'qq_plots_normality.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_histograms():
    """Generate histograms with normal curve overlay."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle('Histograms with Normal Curve: F1-Score Distributions',
                 fontsize=14, fontweight='bold', y=0.98)

    datasets = [
        ('mnist', 'MNIST'),
        ('cifar10', 'CIFAR-10'),
        ('tiny_imagenet', 'Tiny ImageNet')
    ]

    for row, (dataset, label) in enumerate(datasets):
        results = load_results(dataset)
        cnn = [r for r in results if r['model_type'] == 'cnn']
        gan = [r for r in results if r['model_type'] == 'cnn_gan']

        cnn_f1 = np.array([r['f1_macro'] for r in cnn])
        gan_f1 = np.array([r['f1_macro'] for r in gan])

        for col, (data, condition, color) in enumerate([
            (cnn_f1, 'CNN Baseline', 'steelblue'),
            (gan_f1, 'CNN+GAN Augmented', 'darkorange')
        ]):
            ax = axes[row, col]
            n_bins = 20

            # Histogram
            counts, bins, patches = ax.hist(data, bins=n_bins, density=True,
                                             alpha=0.7, color=color, edgecolor='white')

            # Normal curve overlay
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min() - sigma, data.max() + sigma, 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                    label=f'Normal curve\nμ={mu:.4f}, σ={sigma:.4f}')

            w, p = stats.shapiro(data)
            ax.set_title(f'{label} — {condition}\nW={w:.3f}, p={p:.3f}', fontsize=11)
            ax.set_xlabel('F1-Score')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGURES_DIR, 'histograms_normality.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_normality_summary():
    """Print comprehensive normality summary table."""
    print("\n" + "=" * 70)
    print("  NORMALITY ASSESSMENT SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<18} {'Condition':<12} {'W':>7} {'p':>10} {'Skewness':>10} {'Kurtosis':>10} {'Normal?':>9}")
    print("─" * 78)

    datasets = [
        ('mnist', 'MNIST'),
        ('cifar10', 'CIFAR-10'),
        ('tiny_imagenet', 'Tiny ImageNet')
    ]

    for dataset, label in datasets:
        results = load_results(dataset)
        cnn = [r for r in results if r['model_type'] == 'cnn']
        gan = [r for r in results if r['model_type'] == 'cnn_gan']

        cnn_f1 = np.array([r['f1_macro'] for r in cnn])
        gan_f1 = np.array([r['f1_macro'] for r in gan])

        for data, condition in [(cnn_f1, 'CNN'), (gan_f1, 'CNN+GAN')]:
            w, p = stats.shapiro(data)
            skew = stats.skew(data)
            kurt = stats.kurtosis(data)
            normal = '✓ Yes' if p > 0.05 else '✗ No'
            print(f"{label:<18} {condition:<12} {w:>7.3f} {p:>10.4f} {skew:>10.3f} {kurt:>10.3f} {normal:>9}")

    print("\nDecision Rule: If Shapiro-Wilk p > .05, normality assumption is met.")
    print("For Tiny ImageNet CNN+GAN (p < .001), Welch's t-test was used.\n")


def main():
    print("=" * 70)
    print("  NORMALITY TESTS AND VISUALIZATION")
    print("  Deep Learning for Image Recognition and Classification")
    print("=" * 70)

    print("\nGenerating Q-Q plots...")
    generate_qq_plots()

    print("\nGenerating histograms...")
    generate_histograms()

    generate_normality_summary()

    print("=" * 70)
    print("  COMPLETE — Figures saved to figures/ directory")
    print("=" * 70)


if __name__ == '__main__':
    main()
