# -*- coding: utf-8 -*-
"""
Analysis of Claire's BL model Outputs
Goal: create something PDP-adjacent that shows the response
of CF and other stratocumulus variables to changing divergence
across a range of other parameter values
"""
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from string import ascii_lowercase as lowers

numeric_vars = ['D_s_inv', 'D_1e6', 'zi_m', 'zb_m', 'w_sub_hour',
                'CF_pct', 'LWP_g_m2', 'decoupling', 'R_W_m2']

PANELS = [
    ('CF_pct',     'Cloud Fraction [%]'),
    ('LWP_g_m2',   'In-cloud LWP [g m⁻²]'),
    ('decoupling', 'Decoupling parameter $𝒟$'),
    ('R_W_m2',     'Cloud-top cooling ΔR [W m⁻²]'),
]

def load_and_group(data_dir='claire_output'):
    """Load all CSVs, compute w_sub_hour, return dict of {param_name: [df, ...]}."""
    files = sorted(listdir(data_dir))          # sort so groups are contiguous
    groups = {}
    for file in files:
        if not file.endswith('.csv'):
            continue
        data = pd.read_csv(f'{data_dir}/{file}')
        data['w_sub_hour'] = 3600.0 * data['D_s_inv'] * data['zi_m']  # m/hour
        name = data['param_name'].iloc[0]
        groups.setdefault(name, []).append(data[numeric_vars])
    return groups

def marginalise(groups):
    """
    For each param group, average all the per-param-value dataframes
    onto a common w_sub_hour grid using the mean across param values.
    Returns dict of {param_name: averaged_df}.
    """
    averaged = {}
    for name, df_list in groups.items():
        # stack all runs for this param and average by D (same D grid for all)
        combined = pd.concat(df_list)
        averaged[name] = (combined
                          .groupby('D_s_inv')[numeric_vars]
                          .mean()
                          .reset_index(drop=True))
    return averaged

def main():
    groups   = load_and_group('claire_output')
    averaged = marginalise(groups)

    # ── four-panel plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    colors = {'RHft': '#1f77b4', 'SST': '#ff7f0e', 'U': '#2ca02c', 'EIS': '#d62728'}
    labels = {'RHft': 'RH$_{ft}$', 'SST': 'SST', 'U': 'Wind speed $U$', 'EIS': 'EIS'}

    n = 0
    for ax, (col, ylabel) in zip(axes, PANELS):
        for name, df in averaged.items():
            ax.plot(df['w_sub_hour'], df[col],
                    color=colors.get(name, None),
                    label=labels.get(name, name),
                    lw=2)
        ax.set_ylabel(lowers[n] + ') ' + ylabel, fontsize=11)
        ax.grid(alpha=0.3)
        n += 1
    ax.legend(fontsize=9)

    # shared x-label on bottom panels
    for ax in axes[2:]:
        ax.set_xlabel('Subsidence velocity $w_{sub}$ [m hr⁻¹]', fontsize=11)

    plt.tight_layout()
    plt.savefig('pdp_subsidence.png', dpi=150)
    plt.show()
    print('Saved pdp_subsidence.png')

if __name__ == "__main__":
    main()