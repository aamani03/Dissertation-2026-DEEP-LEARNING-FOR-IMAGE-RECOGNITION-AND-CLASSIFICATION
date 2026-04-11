"""
Statistical Analysis for Dissertation
Deep Learning for Image Recognition and Classification

Performs:
- Descriptive statistics (mean, SD, CI) for all datasets
- Shapiro-Wilk normality tests
- Levene's test for homogeneity of variance
- Independent samples t-tests / Welch's t-test
- Cohen's d effect sizes
- Comprehensive results table

Usage: python statistical_analysis.py
"""

import json
import glob
import numpy as np
from scipy import stats
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')


def load_results(dataset):
    """Load all JSON result files for a dataset."""
    pattern = os.path.join(RESULTS_DIR, dataset, '*.json')
    results = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def analyze_dataset(dataset_name, results):
    """Run full statistical analysis on one dataset."""
    cnn = [r for r in results if r['model_type'] == 'cnn']
    gan = [r for r in results if r['model_type'] == 'cnn_gan']

    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} DATASET")
    print(f"  CNN: {len(cnn)} runs | CNN+GAN: {len(gan)} runs")
    print(f"{'='*70}")

    for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
        c = np.array([r[metric] for r in cnn])
        g = np.array([r[metric] for r in gan])

        # Descriptive statistics
        n = len(c)
        ci_lo_c = c.mean() - 1.96 * c.std() / np.sqrt(n)
        ci_hi_c = c.mean() + 1.96 * c.std() / np.sqrt(n)
        ci_lo_g = g.mean() - 1.96 * g.std() / np.sqrt(n)
        ci_hi_g = g.mean() + 1.96 * g.std() / np.sqrt(n)

        print(f"\n--- {metric} ---")
        print(f"  CNN:     M = {c.mean():.4f} ({c.mean()*100:.2f}%)  SD = {c.std():.4f}")
        print(f"           95% CI: [{ci_lo_c*100:.2f}%, {ci_hi_c*100:.2f}%]")
        print(f"  CNN+GAN: M = {g.mean():.4f} ({g.mean()*100:.2f}%)  SD = {g.std():.4f}")
        print(f"           95% CI: [{ci_lo_g*100:.2f}%, {ci_hi_g*100:.2f}%]")

    # --- Statistical Assumption Tests (on F1-score) ---
    print(f"\n{'─'*50}")
    print("STATISTICAL ASSUMPTION TESTS (F1-score)")
    print(f"{'─'*50}")

    cnn_f1 = np.array([r['f1_macro'] for r in cnn])
    gan_f1 = np.array([r['f1_macro'] for r in gan])

    # Shapiro-Wilk Normality Test
    w_c, p_c = stats.shapiro(cnn_f1)
    w_g, p_g = stats.shapiro(gan_f1)
    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  CNN:     W = {w_c:.3f}, p = {p_c:.3f}  {'✓ Normal' if p_c > 0.05 else '✗ NOT Normal'}")
    print(f"  CNN+GAN: W = {w_g:.3f}, p = {p_g:.3f}  {'✓ Normal' if p_g > 0.05 else '✗ NOT Normal'}")

    # Levene's Test for Homogeneity of Variance
    lev, p_lev = stats.levene(cnn_f1, gan_f1)
    print(f"\nLevene's Test for Equal Variances:")
    print(f"  F = {lev:.3f}, p = {p_lev:.3f}  {'✓ Equal variances' if p_lev > 0.05 else '✗ Unequal variances'}")

    # --- Inferential Test ---
    print(f"\n{'─'*50}")
    print("INFERENTIAL ANALYSIS (F1-score)")
    print(f"{'─'*50}")

    use_welch = (p_g < 0.05) or (p_lev < 0.05)

    if use_welch:
        t_stat, p_val = stats.ttest_ind(cnn_f1, gan_f1, equal_var=False)
        # Welch-Satterthwaite degrees of freedom
        n1, n2 = len(cnn_f1), len(gan_f1)
        s1, s2 = cnn_f1.std(ddof=1), gan_f1.std(ddof=1)
        num = (s1**2/n1 + s2**2/n2)**2
        den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        df = num / den
        test_name = "Welch's t-test"
        print(f"\n  Using {test_name} (assumptions violated)")
    else:
        t_stat, p_val = stats.ttest_ind(cnn_f1, gan_f1, equal_var=True)
        df = len(cnn_f1) + len(gan_f1) - 2
        test_name = "Independent samples t-test"
        print(f"\n  Using {test_name} (assumptions met)")

    # Cohen's d
    pooled_sd = np.sqrt((cnn_f1.std(ddof=1)**2 + gan_f1.std(ddof=1)**2) / 2)
    d = (gan_f1.mean() - cnn_f1.mean()) / pooled_sd if pooled_sd > 0 else 0

    print(f"  t({df:.1f}) = {t_stat:.2f}")
    print(f"  p = {p_val:.6f}")
    print(f"  Cohen's d = {d:.2f}", end="")
    abs_d = abs(d)
    if abs_d < 0.2:
        print(" (negligible)")
    elif abs_d < 0.5:
        print(" (small)")
    elif abs_d < 0.8:
        print(" (medium)")
    else:
        print(f" (large, {'negative' if d < 0 else 'positive'} direction)")

    # APA formatted result
    print(f"\n  APA Format: t({df:.1f}) = {abs(t_stat):.2f}, p {'< .001' if p_val < 0.001 else f'= {p_val:.3f}'}, d = {abs(d):.2f}")

    # Training time
    cnn_time = np.array([r['train_time_seconds'] for r in cnn])
    gan_time = np.array([r['train_time_seconds'] for r in gan])
    print(f"\n  Training Time:")
    print(f"    CNN:     {cnn_time.mean():.1f}s ± {cnn_time.std():.1f}s")
    print(f"    CNN+GAN: {gan_time.mean():.1f}s ± {gan_time.std():.1f}s")


def main():
    print("=" * 70)
    print("  DISSERTATION STATISTICAL ANALYSIS")
    print("  Deep Learning for Image Recognition and Classification")
    print("=" * 70)

    for dataset in ['mnist', 'cifar10', 'tiny_imagenet']:
        results = load_results(dataset)
        if results:
            analyze_dataset(dataset, results)
        else:
            print(f"\n  WARNING: No results found for {dataset}")

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
