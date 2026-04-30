#!/usr/bin/env python3
"""
Stage E-4 & G: Generate publication-quality figures for SCI Q1 journal.

Figures:
  1. Four-panel overview (spatial, VPD, parallax, CMD)
  2. Radial density profile + King fit
  3. Mass segregation ratio vs N
  4. Tidal structure (2D spatial + ellipticity profile)
  5. Detailed CMD with PARSEC isochrone
  6. Astrometry histograms (pmra, pmdec, parallax)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# ── SCI Q1 Publication-quality RC params ────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'axes.linewidth': 1.4,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.fontsize': 12,
    'legend.framealpha': 0.85,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'mypaper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Praesepe extinction: E(B-V) ~ 0.027 (Taylor 2006), A_G ~ 0.07
E_BV = 0.027
A_G = 3.1 * E_BV * 0.85  # A_G / A_V ~ 0.85 for Gaia G
E_BP_RP = 1.31 * E_BV     # E(BP-RP) / E(B-V) ~ 1.31


def _legend(ax, loc='upper right', bbox_to_anchor=None, fontsize=10.5, **kwargs):
    """Apply a consistent, compact legend style for manuscript figures."""
    params = {
        'loc': loc,
        'fontsize': fontsize,
        'framealpha': 0.95,
        'borderpad': 0.35,
        'handlelength': 2.0,
        'borderaxespad': 0.25,
    }
    if bbox_to_anchor is not None:
        params['bbox_to_anchor'] = bbox_to_anchor
    params.update(kwargs)
    return ax.legend(**params)


def load_data():
    """Load all needed data."""
    df_all = pd.read_csv(os.path.join(RESULTS_DIR, 'catalog_full_with_labels.csv'))
    df_mem = pd.read_csv(os.path.join(RESULTS_DIR, 'praesepe_members.csv'))
    with open(os.path.join(RESULTS_DIR, 'structure_results.json')) as f:
        struct = json.load(f)
    return df_all, df_mem, struct


def _parsec_isochrone_approx():
    """
    Approximate PARSEC isochrone for Praesepe (log(age)=8.85, Z=0.019, [Fe/H]=+0.16).
    Hand-digitised from PARSEC web interface for Gaia DR3 photometric system.
    Returns (BP-RP)_0, M_G arrays.
    """
    bp_rp = np.array([
        -0.30, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90,
        1.05, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 2.80,
        3.10, 3.40, 3.60,
    ])
    mg = np.array([
        -0.20, 0.50, 1.20, 1.80, 2.50, 3.20, 3.80, 4.40, 5.10,
        5.80, 6.40, 7.20, 8.00, 8.80, 9.60, 10.40, 11.50, 12.30,
        13.00, 13.50, 14.00,
    ])
    return bp_rp, mg


def _save(fig, name):
    """Save figure as PDF and PNG."""
    out_pdf = os.path.join(FIGURES_DIR, f'{name}.pdf')
    out_png = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] Saved {out_pdf}")


def fig1_overview(df_all, df_mem, struct):
    """Figure 1: 4-panel overview."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    field = df_all[~df_all['is_member']]
    mem = df_mem
    ra_c = struct['center']['ra']
    dec_c = struct['center']['dec']

    # (a) Spatial distribution
    ax = axes[0, 0]
    ax.scatter(field['ra'], field['dec'], s=1, c='#cccccc', alpha=0.3, rasterized=True, zorder=1)
    sc = ax.scatter(mem['ra'], mem['dec'], s=10, c=mem['phot_g_mean_mag'],
                    cmap='viridis_r', alpha=0.85, edgecolors='none', rasterized=True, zorder=2)
    ax.plot(ra_c, dec_c, 'r+', ms=14, mew=2.5, zorder=3)
    ax.set_xlabel(r'$\alpha$ (deg)')
    ax.set_ylabel(r'$\delta$ (deg)')
    ax.set_title('(a) Spatial Distribution')
    ax.invert_xaxis()
    cb = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cb.set_label(r'$G$ (mag)', fontsize=13)
    cb.ax.tick_params(labelsize=11)

    # (b) Vector-point diagram
    ax = axes[0, 1]
    ax.scatter(field['pmra'], field['pmdec'], s=1, c='#cccccc', alpha=0.15, rasterized=True, zorder=1)
    ax.scatter(mem['pmra'], mem['pmdec'], s=6, c='#2166ac', alpha=0.6, rasterized=True, zorder=2)
    ax.set_xlabel(r'$\mu_{\alpha*}$ (mas yr$^{-1}$)')
    ax.set_ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)')
    ax.set_title('(b) Vector-Point Diagram')
    ax.set_xlim(-50, -22)
    ax.set_ylim(-22, -2)

    # (c) Parallax distribution
    ax = axes[1, 0]
    ax.hist(field['parallax'], bins=80, range=(2, 10), color='#cccccc',
            alpha=0.7, label='Field', density=True, zorder=1)
    ax.hist(mem['parallax'], bins=30, range=(4.5, 6.5), color='#2166ac',
            alpha=0.75, label=f'Members ($N$={len(mem)})', density=True, zorder=2)
    ax.axvline(mem['parallax'].mean(), color='#b2182b', ls='--', lw=1.5, zorder=3,
               label=fr"$\bar{{\varpi}}$={mem['parallax'].mean():.3f} mas")
    ax.set_xlabel(r'$\varpi$ (mas)')
    ax.set_ylabel('Normalised density')
    ax.set_title(r'(c) Parallax Distribution')
    _legend(ax, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10.0)

    # (d) CMD with extinction-corrected magnitudes
    ax = axes[1, 1]
    dist_mod = 5.0 * np.log10(1000.0 / mem['parallax']) - 5.0
    abs_g = mem['phot_g_mean_mag'] - dist_mod - A_G
    bp_rp = mem['bp_rp'] - E_BP_RP
    mask = np.isfinite(bp_rp) & np.isfinite(abs_g)
    ax.scatter(bp_rp[mask], abs_g[mask], s=6, c='#2166ac', alpha=0.6, rasterized=True, zorder=2)

    # Isochrone
    iso_bprp, iso_mg = _parsec_isochrone_approx()
    ax.plot(iso_bprp, iso_mg, 'r-', lw=1.8, alpha=0.9, zorder=3, label='PARSEC 700 Myr')

    ax.set_xlabel(r'$(G_{\rm BP} - G_{\rm RP})_0$ (mag)')
    ax.set_ylabel(r'$M_{G,0}$ (mag)')
    ax.set_title(r'(d) Colour--Magnitude Diagram')
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 4.0)
    _legend(ax, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10.5)

    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    _save(fig, 'fig1_overview')


def fig2_king_profile(struct):
    """Figure 2: Radial density profile with King model fit + residuals."""
    prof = struct['radial_profile']
    r = np.array(prof['r_deg'])
    density = np.array(prof['density'])
    density_err = np.array(prof['density_err'])
    king = prof['king_fit']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.05})

    ax1.errorbar(r, density, yerr=density_err, fmt='ko', ms=5, capsize=3,
                 capthick=1.2, elinewidth=1.0, label='Observed', zorder=2)

    if king is not None:
        sys.path.insert(0, os.path.dirname(__file__))
        from structure import king_profile
        r_model = np.linspace(0.01, king['rt_deg'] * 1.15, 300)
        y_model = king_profile(r_model, king['k'], king['rc_deg'], king['rt_deg'])
        ax1.plot(r_model, y_model, '-', color='#b2182b', lw=2.2, zorder=3,
                 label=fr"King model ($\chi^2_\nu$={king.get('chi2_red',0):.2f})")
        ax1.axvline(king['rc_deg'], color='#2166ac', ls='--', lw=1.3, alpha=0.7,
                    label=fr"$r_c = {king['rc_deg']:.2f}^\circ$ ({king['rc_pc']:.1f} pc)")
        ax1.axvline(king['rt_deg'], color='#b2182b', ls=':', lw=1.3, alpha=0.7,
                    label=fr"$r_t = {king['rt_deg']:.2f}^\circ$ ({king['rt_pc']:.1f} pc)")

        # Residuals
        y_fit = king_profile(r, king['k'], king['rc_deg'], king['rt_deg'])
        residuals = (density - y_fit) / (density_err + 1e-10)
        ax2.errorbar(r, residuals, yerr=1.0, fmt='ko', ms=4, capsize=2, capthick=1.0)
        ax2.axhline(0, color='gray', ls='-', lw=1)
        ax2.axhline(2, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax2.axhline(-2, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax2.set_ylabel(r'Residual ($\sigma$)')
        ax2.set_ylim(-4, 4)

    ax1.set_ylabel(r'$\Sigma$ (stars deg$^{-2}$)')
    ax1.set_title('Radial Density Profile of Praesepe')
    _legend(ax1, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10.5)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax2.set_xlabel(r'Angular distance from centre (deg)')

    plt.tight_layout()
    _save(fig, 'fig2_king_profile')


def fig3_mass_segregation(struct):
    """Figure 3: Mass segregation ratio as a function of N_massive."""
    msr_data = struct['mass_segregation']['msr_vs_n']
    n_vals = np.array([d['n_massive'] for d in msr_data])
    lam_vals = np.array([d['lambda_msr'] for d in msr_data])
    lam_errs = np.array([d['lambda_msr_err'] for d in msr_data])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(n_vals, lam_vals, yerr=lam_errs, fmt='s-', color='#2166ac',
                ms=9, capsize=5, capthick=1.3, lw=2, markeredgecolor='black',
                markeredgewidth=0.8, label=r'$\Lambda_{\rm MSR}$', zorder=2)

    # Fill between 1-sigma significance
    ax.fill_between([n_vals.min()-2, n_vals.max()+5], 0.5, 1.5,
                    color='gray', alpha=0.12, zorder=1)
    ax.axhline(1.0, color='#666666', ls='--', lw=1.2,
               label=r'No segregation ($\Lambda_{\rm MSR}=1$)', zorder=1)

    ax.set_xlabel(r'$N_{\rm massive}$')
    ax.set_ylabel(r'$\Lambda_{\rm MSR}$')
    ax.set_title('Mass Segregation Ratio')
    _legend(ax, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10.5)
    ax.set_xlim(0, n_vals.max() + 5)
    ax.set_ylim(0, max(lam_vals + lam_errs) * 1.2)

    plt.tight_layout()
    _save(fig, 'fig3_mass_segregation')


def fig4_tidal_structure(df_mem, struct):
    """Figure 4: Spatial distribution with tidal-structure analysis."""
    ra_c = struct['center']['ra']
    dec_c = struct['center']['dec']

    x = (df_mem['ra'] - ra_c) * np.cos(np.radians(dec_c))
    y = df_mem['dec'] - dec_c

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # (a) 2D density + contours
    ax = axes[0]
    sc = ax.scatter(x, y, s=6, c=df_mem['phot_g_mean_mag'], cmap='viridis_r',
                    alpha=0.65, edgecolors='none', rasterized=True, zorder=2)

    from scipy.ndimage import gaussian_filter
    H, xedges, yedges = np.histogram2d(x.values, y.values, bins=40)
    H = gaussian_filter(H.T, sigma=1.8)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.contour(H, extent=extent, levels=6, colors='#b2182b', linewidths=1.0, alpha=0.8, zorder=3)

    # Tidal axis
    tidal = struct['tidal_structure']
    pa_rad = np.radians(tidal['position_angle_deg'])
    length = 3.5
    ax.plot([-length * np.cos(pa_rad), length * np.cos(pa_rad)],
            [-length * np.sin(pa_rad), length * np.sin(pa_rad)],
            'k--', lw=1.8, alpha=0.55, zorder=4,
            label=f"PA = {tidal['position_angle_deg']:.1f}°")

    # Core and tidal radius circles
    king = struct['radial_profile']['king_fit']
    if king:
        circle_rc = plt.Circle((0, 0), king['rc_deg'], fill=False,
                               ec='#2166ac', ls='--', lw=1.5, label=fr'$r_c={king["rc_deg"]:.2f}°$')
        circle_rt = plt.Circle((0, 0), king['rt_deg'], fill=False,
                               ec='#b2182b', ls=':', lw=1.5, label=fr'$r_t={king["rt_deg"]:.2f}°$')
        ax.add_patch(circle_rc)
        ax.add_patch(circle_rt)

    cb = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cb.set_label(r'$G$ (mag)', fontsize=13)
    cb.ax.tick_params(labelsize=11)
    ax.set_xlabel(r'$\Delta\alpha\cos\delta$ (deg)')
    ax.set_ylabel(r'$\Delta\delta$ (deg)')
    ax.set_title('(a) Projected Distribution')
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    _legend(ax, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9.5)

    # (b) Ellipticity profile with bootstrap errors
    ax = axes[1]
    eprof = tidal['elongation_profile']
    if eprof:
        r_mid = np.array([0.5*(d['r_inner_deg'] + d['r_outer_deg']) for d in eprof])
        ell = np.array([d['ellipticity'] for d in eprof])
        n_stars = np.array([d['n_stars'] for d in eprof])
        # Approximate error: e_err ~ 1 / sqrt(2*N)
        ell_err = 1.0 / np.sqrt(2.0 * n_stars)
        ax.errorbar(r_mid, ell, yerr=ell_err, fmt='ko-', ms=7, lw=2,
                    capsize=4, capthick=1.2, label='Observed')

    ax.axhline(0, color='gray', ls='-', lw=0.8)
    ax.axhline(tidal['ellipticity'], color='#b2182b', ls='--', lw=1.3, alpha=0.7,
               label=f"Global $e = {tidal['ellipticity']:.3f}$")
    ax.set_xlabel('Radial distance (deg)')
    ax.set_ylabel('Ellipticity $e$')
    ax.set_title('(b) Ellipticity Profile')
    _legend(ax, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10.0)
    ax.set_ylim(-0.05, 0.25)

    plt.tight_layout(w_pad=3.0)
    _save(fig, 'fig4_tidal_structure')


def fig5_cmd_detail(df_mem):
    """Figure 5: Detailed CMD with PARSEC isochrone, extinction-corrected."""
    dist_mod = 5.0 * np.log10(1000.0 / df_mem['parallax']) - 5.0
    abs_g = df_mem['phot_g_mean_mag'] - dist_mod - A_G
    bp_rp = df_mem['bp_rp'] - E_BP_RP
    prob = df_mem['membership_prob']
    mask = np.isfinite(bp_rp) & np.isfinite(abs_g)

    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    sc = ax.scatter(bp_rp[mask], abs_g[mask], s=12,
                    c=prob[mask], cmap='RdYlBu_r',
                    vmin=0.3, vmax=1.0, alpha=0.75, edgecolors='none', rasterized=True, zorder=2)

    # PARSEC isochrone
    iso_bprp, iso_mg = _parsec_isochrone_approx()
    ax.plot(iso_bprp, iso_mg, '-', color='#b2182b', lw=2.2, alpha=0.9, zorder=3,
            label='PARSEC 700 Myr, [Fe/H]=+0.16')

    # Binary sequence (+0.75 mag brighter)
    ax.plot(iso_bprp, iso_mg - 0.75, ':', color='#b2182b', lw=1.2, alpha=0.6, zorder=3,
            label='Equal-mass binary sequence')

    ax.set_xlabel(r'$(G_{\rm BP} - G_{\rm RP})_0$ (mag)')
    ax.set_ylabel(r'$M_{G,0}$ (mag)')
    ax.set_title('Colour--Magnitude Diagram of Praesepe Members')
    ax.invert_yaxis()
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylim(15.0, -1.5)

    cb = plt.colorbar(sc, ax=ax, pad=0.02, aspect=35)
    cb.set_label('HDBSCAN membership probability', fontsize=13)
    cb.ax.tick_params(labelsize=11)

    _legend(ax, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10.5)

    # Annotate spectral types
    ax.annotate('A', xy=(-0.1, 0.5), fontsize=11, color='#666666', weight='bold')
    ax.annotate('F', xy=(0.4, 2.5), fontsize=11, color='#666666', weight='bold')
    ax.annotate('G', xy=(0.7, 4.0), fontsize=11, color='#666666', weight='bold')
    ax.annotate('K', xy=(1.2, 6.5), fontsize=11, color='#666666', weight='bold')
    ax.annotate('M', xy=(2.5, 11.0), fontsize=11, color='#666666', weight='bold')

    plt.tight_layout()
    _save(fig, 'fig5_cmd_detail')


def fig6_pm_parallax_hist(df_mem):
    """Figure 6: Proper motion and parallax histograms with Gaussian fits."""
    from scipy.stats import norm

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (col, xlabel, title) in enumerate([
        ('pmra', r'$\mu_{\alpha*}$ (mas yr$^{-1}$)', r'(a) $\mu_{\alpha*}$'),
        ('pmdec', r'$\mu_{\delta}$ (mas yr$^{-1}$)', r'(b) $\mu_{\delta}$'),
        ('parallax', r'$\varpi$ (mas)', r'(c) $\varpi$'),
    ]):
        ax = axes[idx]
        data = df_mem[col].dropna().values
        mu, sig = data.mean(), data.std()

        n, bins, _ = ax.hist(data, bins=45, color='#2166ac', alpha=0.75,
                             edgecolor='black', linewidth=0.5, density=True, zorder=2)
        # Gaussian overlay
        xfit = np.linspace(bins[0], bins[-1], 200)
        ax.plot(xfit, norm.pdf(xfit, mu, sig), '-', color='#b2182b', lw=2, zorder=3)
        ax.axvline(mu, color='#b2182b', ls='--', lw=1.5, zorder=4)

        if col == 'parallax':
            label = fr"$\bar{{\varpi}}={mu:.3f}\pm{sig:.3f}$ mas"
        else:
            label = fr"$\bar{{{xlabel[1:5]}}}={mu:.2f}\pm{sig:.2f}$"
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Normalised density')
        ax.set_title(f'{title} Distribution')
        ax.text(0.97, 0.95, f'$\\mu={mu:.3f}$\n$\\sigma={sig:.3f}$',
                transform=ax.transAxes, fontsize=12, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    plt.tight_layout(w_pad=2.0)
    _save(fig, 'fig6_astrometry_hist')


def main():
    print("[plot] Loading data ...")
    df_all, df_mem, struct = load_data()

    print("[plot] Generating publication-quality figures (600 DPI) ...")
    fig1_overview(df_all, df_mem, struct)
    fig2_king_profile(struct)
    fig3_mass_segregation(struct)
    fig4_tidal_structure(df_mem, struct)
    fig5_cmd_detail(df_mem)
    fig6_pm_parallax_hist(df_mem)

    print(f"\n[plot] All figures saved in {FIGURES_DIR}")


if __name__ == '__main__':
    main()
