#!/usr/bin/env python3
"""Recreate the four-panel figure from plane_angles_plot_data.mat.

Dependencies: numpy, scipy, matplotlib
Run: python replot_from_mat.py
Optional: --data path/to/data.mat --output-dir path/to/output --max-dimension 64
Only the MAT file is needed; no simulation scripts or CSVs are required.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

matplotlib.use('Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/null_plane_angles_matplotlib')


def plot(dims, summaries, alpha_means, out, max_dimension):
    mask = dims <= max_dimension
    dims, summaries, alpha_means = dims[mask], summaries[mask], alpha_means[mask]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(25, 6.8), sharex=True, sharey=True)
    titles = [r'Smallest principal angle $\theta_{\min}$',
              r'Largest principal angle $\theta_{\max}$', r'Effective overlap angle $\alpha$',
              r'Arithmetic mean $(\theta_{\min}+\theta_{\max})/2$']
    for k, (ax, color, title) in enumerate(
            zip(axes, ['#227c9d', '#bd5b35', '#4c8c65', '#8b6baa'], titles)):
        stats = [
            dict(whislo=s[0, k], q1=s[1, k], med=s[2, k], q3=s[3, k], whishi=s[4, k], fliers=[])
            for s in summaries]
        ax.bxp(stats, positions=dims, widths=.65, showfliers=False, patch_artist=True,
               manage_ticks=False,
               boxprops={'facecolor': color, 'edgecolor': color, 'alpha': .5},
               medianprops={'color': '#172638', 'linewidth': .9},
               whiskerprops={'color': color, 'linewidth': .65},
               capprops={'color': color, 'linewidth': .65})
        ax.set_title(title, loc='left', weight='bold', fontsize=13, pad=13)
        ax.set(xlim=(1, max_dimension + 1), ylim=(-2, 93), xticks=sorted(
            set([d for d in [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if d <= max_dimension] + [
                max_dimension])),
               yticks=np.arange(0, 91, 15), xlabel='Ambient dimension D')
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelleft=True)
        ax.grid(axis='y', alpha=.23)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Angle (degrees)')
    axes[2].plot(dims, alpha_means, color='#742c83', lw=1.5,
                 label=r'Angle of mean overlap: $\arccos\sqrt{\widehat{U}}$')
    axes[2].plot(dims, np.rad2deg(np.arccos(np.sqrt(2 / dims))), color='#111111',
                 lw=1.1, ls='--', label=r'Exact null: $\arccos\sqrt{2/D}$')
    axes[2].legend(loc='lower right', frameon=False, fontsize=9.5)
    fig.suptitle('Random 2D planes: principal angles, overlap and arithmetic mean',
                 x=.055, y=.975, ha='left', weight='bold', fontsize=20)
    fig.text(.055, .91,
             '1,000 isotropic planes per dimension • all 499,500 distinct pairs • '
             'identical angle scales',
             fontsize=12)
    fig.text(.055, .055,
             'Boxes: 25th–75th percentiles  |  Central line: median  |  '
             'Whiskers: 5th–95th percentiles\n'
             r'Per pair: $u=(\cos^2\theta_{\min}+\cos^2\theta_{\max})/2$,  '
             r'$\alpha=\arccos\sqrt{u}$. '
             r'Across pairs: $U=E[u]$. The angle of mean overlap differs from the mean angle.',
             fontsize=11, color='#465260', linespacing=1.9)
    fig.subplots_adjust(left=.055, right=.99, top=.81, bottom=.205, wspace=.13)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'plane_angles_four_panel_from_mat.{ext}', dpi=220)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path(__file__).resolve().with_name('plane_angles_plot_data.mat'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--max-dimension', type=int, default=None)
    args = parser.parse_args()
    data = loadmat(args.data, squeeze_me=True)
    dims = np.asarray(data['dimensions']).reshape(-1)
    summaries = np.asarray(data['angle_quantiles_deg'])
    alpha_means = np.asarray(data['angle_of_empirical_U_deg']).reshape(-1)
    maximum = args.max_dimension if args.max_dimension is not None else int(
        data['plot_max_dimension'])
    if maximum < 2 or maximum > int(dims.max()):
        parser.error(f'--max-dimension must be between 2 and {dims.max()}')
    if summaries.shape != (len(dims), 5, 4):
        raise ValueError('Expected quantiles shape (number of dimensions, 5, 4)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(dims, summaries, alpha_means, args.output_dir, maximum)
    print(f'Figure saved as PNG, PDF and SVG in {args.output_dir}')


if __name__ == '__main__':
    main()
