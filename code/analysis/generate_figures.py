"""
Generate Dissertation Figures
Deep Learning for Image Recognition and Classification

Generates:
- Figure 4.1: Accuracy Comparison
- Figure 4.2: F1-Score Comparison
- Figure 4.3: Precision Comparison
- Figure 4.4: Recall Comparison

Usage: python generate_figures.py
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results(dataset):
    pattern = os.path.join(RESULTS_DIR, dataset, '*.json')
    results = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def get_stats(results, metric):
    cnn = [r[metric] for r in results if r['model_type'] == 'cnn']
    gan = [r[metric] for r in results if r['model_type'] == 'cnn_gan']
    return {
        'cnn_mean': np.mean(cnn),
        'cnn_std': np.std(cnn),
        'gan_mean': np.mean(gan),
        'gan_std': np.std(gan)
    }


def create_comparison_figure(metric, metric_label, filename, fignum, is_percentage=True):
    """Create a grouped bar chart comparing CNN vs CNN+GAN across datasets."""
    datasets = [
        ('mnist', 'MNIST'),
        ('cifar10', 'CIFAR-10'),
        ('tiny_imagenet', 'Tiny ImageNet')
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.35

    cnn_means, gan_means = [], []
    cnn_stds, gan_stds = [], []

    for dataset, _ in datasets:
        results = load_results(dataset)
        s = get_stats(results, metric)
        multiplier = 100 if is_percentage else 1
        cnn_means.append(s['cnn_mean'] * multiplier)
        gan_means.append(s['gan_mean'] * multiplier)
        cnn_stds.append(s['cnn_std'] * multiplier)
        gan_stds.append(s['gan_std'] * multiplier)

    bars1 = ax.bar(x - width/2, cnn_means, width, yerr=cnn_stds,
                    label='CNN Baseline', color='steelblue', capsize=5,
                    edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, gan_means, width, yerr=gan_stds,
                    label='CNN+GAN Augmented', color='darkorange', capsize=5,
                    edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=12)
    ylabel = f'{metric_label} (%)' if is_percentage else metric_label
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Figure {fignum}. {metric_label} Comparison Between CNN and CNN+GAN Models',
                  fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in datasets], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        fmt = f'{height:.2f}' if is_percentage else f'{height:.3f}'
        ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        fmt = f'{height:.2f}' if is_percentage else f'{height:.3f}'
        ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  GENERATING DISSERTATION FIGURES")
    print("=" * 60)

    figures = [
        ('accuracy', 'Accuracy', 'figure_4_1_accuracy.png', '4.1', True),
        ('f1_macro', 'F1-Score', 'figure_4_2_f1score.png', '4.2', False),
        ('precision_macro', 'Precision', 'figure_4_3_precision.png', '4.3', False),
        ('recall_macro', 'Recall', 'figure_4_4_recall.png', '4.4', True),
    ]

    for metric, label, fname, fnum, is_pct in figures:
        print(f"\n  Generating Figure {fnum}: {label} Comparison...")
        create_comparison_figure(metric, label, fname, fnum, is_pct)

    print(f"\n{'='*60}")
    print("  ALL FIGURES GENERATED — saved to figures/ directory")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
