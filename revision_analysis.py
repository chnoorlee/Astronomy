#!/usr/bin/env python3
"""
Extended analyses:
  (a) Perspective-rotation correction of the proper-motion velocity dispersion.
  (b) Single-star radial-velocity dispersion (RUWE < 1.2, RV error < 1 km/s).
  (c) Five-dimensional (alpha, delta, pmra, pmdec, parallax) membership
      comparison and a posterior spatial filter on the 3D catalogue.
  (d) Monte-Carlo unresolved-binary correction of the mass-function slope,
      mean stellar mass, and total cluster mass.
  (e) Statistical significance of the projected ellipticity (bootstrap +
      circular null Monte-Carlo) and literature context.
  (f) Extended systematic-error budget (distance, extinction, isochrone
      zero point, parallax zero point).

Outputs: results/revision_results.json, several CSV tables, and figures
fig12-fig14 in the paper figures directory.
"""

import json
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'paper', 'mypaper', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PM_RA_CENTER = -36.1
PM_DEC_CENTER = -12.9
PLX_CENTER = 5.37
# Extinction: E(B-V)=0.027 (Taylor 2006). Gaia-band coefficients from
# Wang & Chen (2019), R_V=3.1: A_G/A_V=0.789, A_BP/A_V=1.002, A_RP/A_V=0.589.
E_BV = 0.027
_A_V = 3.1 * E_BV
A_G = 0.789 * _A_V                  # = 0.066 mag
E_BP_RP = (1.002 - 0.589) * _A_V    # = 0.035 mag
RG_EBV = 0.789 * 3.1                # A_G / E(B-V)      = 2.446
RBPRP_EBV = (1.002 - 0.589) * 3.1   # E(BP-RP) / E(B-V) = 1.280
K_PM = 4.74047  # km/s per (mas/yr * kpc)
G_CONST = 4.30091e-3  # kpc (km/s)^2 / Msun
RNG = np.random.default_rng(42)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})


def save_figure(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'))
    plt.close(fig)


def iso_arrays():
    colour = np.array([-0.30, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 2.80, 3.10, 3.40, 3.60])
    mg = np.array([-0.20, 0.50, 1.20, 1.80, 2.50, 3.20, 3.80, 4.40, 5.10, 5.80, 6.40, 7.20, 8.00, 8.80, 9.60, 10.40, 11.50, 12.30, 13.00, 13.50, 14.00])
    mass = np.array([2.60, 2.35, 2.10, 1.85, 1.60, 1.38, 1.20, 1.05, 0.92, 0.82, 0.73, 0.62, 0.52, 0.43, 0.34, 0.27, 0.20, 0.155, 0.120, 0.100, 0.090])
    return colour, mg, mass


def mass_to_mg(mass_vals):
    _, iso_mg, iso_mass = iso_arrays()
    order = np.argsort(iso_mass)
    return np.interp(mass_vals, iso_mass[order], iso_mg[order],
                     left=iso_mg[order][0], right=iso_mg[order][-1])


def mg_to_mass(mg_vals, mass_shift_dex=0.0):
    _, iso_mg, iso_mass = iso_arrays()
    iso_mass_use = iso_mass * 10 ** mass_shift_dex
    order = np.argsort(iso_mg)
    log_mass = np.interp(mg_vals, iso_mg[order], np.log10(iso_mass_use[order]),
                         left=np.nan, right=np.nan)
    return 10 ** log_mass


def tangent_xy(df, ra_c, dec_c):
    x = (df['ra'].to_numpy() - ra_c) * np.cos(np.radians(dec_c))
    y = df['dec'].to_numpy() - dec_c
    return x, y


def angular_radius(df, ra_c, dec_c):
    x, y = tangent_xy(df, ra_c, dec_c)
    return np.sqrt(x * x + y * y)


def king_profile(r, k, rc, rt):
    t1 = 1.0 / np.sqrt(1.0 + (r / rc) ** 2)
    t2 = 1.0 / np.sqrt(1.0 + (rt / rc) ** 2)
    return np.where(r <= rt, k * (t1 - t2) ** 2, 0.0)


def fit_king_radius(radius, area_factor=1.0, n_bins=20):
    r_max = np.percentile(radius, 99)
    edges = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(radius, bins=edges)
    areas = area_factor * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = counts / areas
    err = np.sqrt(np.maximum(counts, 1)) / areas
    p0 = [np.nanmax(density), max(np.median(radius) / 2.0, 0.1), max(r_max * 1.2, 0.5)]
    bounds = ([0.0, 0.01, 0.1], [np.nanmax(density) * 20.0, 8.0, 20.0])
    popt, _ = curve_fit(king_profile, centers, density, p0=p0, sigma=err,
                        bounds=bounds, maxfev=20000)
    return {'rc_deg': float(popt[1]), 'rt_deg': float(popt[2])}


# ---------------------------------------------------------------------------
# (a) Perspective correction
# ---------------------------------------------------------------------------

def triad(ra_rad, dec_rad):
    """Return the (r, p, q) ICRS unit-vector triad for arrays of angles."""
    cosa, sina = np.cos(ra_rad), np.sin(ra_rad)
    cosd, sind = np.cos(dec_rad), np.sin(dec_rad)
    r = np.stack([cosa * cosd, sina * cosd, sind], axis=-1)
    p = np.stack([-sina, cosa, np.zeros_like(sina)], axis=-1)
    q = np.stack([-cosa * sind, -sina * sind, cosd], axis=-1)
    return r, p, q


def perspective_correction(df_mem):
    ra = np.radians(df_mem['ra'].to_numpy())
    dec = np.radians(df_mem['dec'].to_numpy())
    pmra = df_mem['pmra'].to_numpy()
    pmdec = df_mem['pmdec'].to_numpy()
    e_pmra = df_mem['pmra_error'].to_numpy()
    e_pmdec = df_mem['pmdec_error'].to_numpy()

    ra0 = np.radians(np.median(df_mem['ra']))
    dec0 = np.radians(np.median(df_mem['dec']))
    d0_kpc = (1000.0 / df_mem['parallax'].mean()) / 1000.0

    rv = df_mem['radial_velocity'].to_numpy()
    rv_ok = np.isfinite(rv)
    rv_med = np.median(rv[rv_ok])
    rv_mad = 1.4826 * np.median(np.abs(rv[rv_ok] - rv_med))
    vr0 = float(np.mean(rv[rv_ok & (np.abs(rv - rv_med) < 3 * rv_mad)]))

    mu_a0 = float(df_mem['pmra'].mean())
    mu_d0 = float(df_mem['pmdec'].mean())

    # cluster mean space velocity (km/s, ICRS rectangular)
    r0, p0, q0 = triad(np.array([ra0]), np.array([dec0]))
    r0, p0, q0 = r0[0], p0[0], q0[0]
    v_a0 = K_PM * mu_a0 * d0_kpc
    v_d0 = K_PM * mu_d0 * d0_kpc
    V = vr0 * r0 + v_a0 * p0 + v_d0 * q0

    # predicted PM at each star position assuming common space motion at d0
    _, p_i, q_i = triad(ra, dec)
    mu_a_pred = (p_i @ V) / (K_PM * d0_kpc)
    mu_d_pred = (q_i @ V) / (K_PM * d0_kpc)

    # residual internal proper motion after removing the perspective field
    dmu_a = pmra - mu_a_pred
    dmu_d = pmdec - mu_d_pred

    conv = K_PM * d0_kpc

    def err_corr_sigma(res, err):
        s_obs = np.std(res, ddof=1)
        return np.sqrt(max(s_obs ** 2 - np.mean(err ** 2), 0.0))

    # uncorrected (subtract mean PM only) -- reproduces the fiducial method
    s_a_unc = err_corr_sigma(pmra - mu_a0, e_pmra) * conv
    s_d_unc = err_corr_sigma(pmdec - mu_d0, e_pmdec) * conv
    s1d_unc = np.sqrt((s_a_unc ** 2 + s_d_unc ** 2) / 2.0)

    # perspective corrected
    s_a_cor = err_corr_sigma(dmu_a, e_pmra) * conv
    s_d_cor = err_corr_sigma(dmu_d, e_pmdec) * conv
    s1d_cor = np.sqrt((s_a_cor ** 2 + s_d_cor ** 2) / 2.0)

    out = {
        'vr0_kms': vr0, 'd0_pc': d0_kpc * 1000.0,
        'sigma_a_uncorr_kms': float(s_a_unc), 'sigma_d_uncorr_kms': float(s_d_unc),
        'sigma_1d_uncorr_kms': float(s1d_unc),
        'sigma_a_persp_kms': float(s_a_cor), 'sigma_d_persp_kms': float(s_d_cor),
        'sigma_1d_persp_kms': float(s1d_cor),
        'fractional_reduction': float((s1d_unc - s1d_cor) / s1d_unc),
        'perspective_pm_gradient_max_masyr': float(np.max(np.hypot(mu_a_pred - mu_a0, mu_d_pred - mu_d0))),
    }
    return out, (mu_a_pred - mu_a0, mu_d_pred - mu_d0)


# ---------------------------------------------------------------------------
# (b) Single-star RV dispersion
# ---------------------------------------------------------------------------

def rv_singlestar(df_mem):
    rows = []
    base = df_mem.dropna(subset=['radial_velocity']).copy()
    configs = [
        ('all RV members', base),
        ('RUWE<1.2', base[base['ruwe'] < 1.2]),
        ('RUWE<1.2 & e_RV<1', base[(base['ruwe'] < 1.2) & (base['radial_velocity_error'] < 1.0)]),
        ('RUWE<1.2 & e_RV<1.5', base[(base['ruwe'] < 1.2) & (base['radial_velocity_error'] < 1.5)]),
    ]
    for name, sub in configs:
        rv = sub['radial_velocity'].to_numpy()
        erv = sub['radial_velocity_error'].to_numpy()
        if len(rv) < 5:
            rows.append({'config': name, 'n': int(len(rv)), 'sigma_raw': np.nan,
                         'sigma_clip': np.nan, 'sigma_int': np.nan, 'mean_rv': np.nan})
            continue
        med = np.median(rv)
        mad = 1.4826 * np.median(np.abs(rv - med))
        keep = np.abs(rv - med) < 3 * mad
        rvc, ervc = rv[keep], erv[keep]
        s_clip = float(np.std(rvc, ddof=1))
        s_int = float(np.sqrt(max(s_clip ** 2 - np.mean(ervc ** 2), 0.0)))
        rows.append({'config': name, 'n': int(len(rv)), 'n_clip': int(keep.sum()),
                     'sigma_raw': float(np.std(rv, ddof=1)),
                     'sigma_clip': s_clip, 'sigma_int': s_int,
                     'mean_rv': float(np.mean(rvc)),
                     'median_e_rv': float(np.median(erv))})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'rv_singlestar.csv'), index=False)
    return rows


# ---------------------------------------------------------------------------
# (c) 5D membership + spatial posterior filter
# ---------------------------------------------------------------------------

def membership_5d(df_all, df_mem, ra_c, dec_c, struct):
    fiducial_ids = set(df_mem['source_id'].astype(str))
    res = {'have_hdbscan': HAVE_HDBSCAN}

    if HAVE_HDBSCAN:
        feats = df_all[['ra', 'dec', 'pmra', 'pmdec', 'parallax']].to_numpy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(feats)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10,
                                    metric='euclidean', cluster_selection_method='eom')
        labels = clusterer.fit_predict(Xs)
        probs = clusterer.probabilities_
        target = scaler.transform([[ra_c, dec_c, PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER]])[0]
        best_lab, best_d = -1, np.inf
        for lab in np.unique(labels[labels >= 0]):
            d = np.linalg.norm(Xs[labels == lab].mean(axis=0) - target)
            if d < best_d:
                best_d, best_lab = d, lab
        mask5 = (labels == best_lab) & (probs >= 0.3)
        ids5 = set(df_all.loc[mask5, 'source_id'].astype(str))
        inter = len(ids5 & fiducial_ids)
        union = len(ids5 | fiducial_ids)
        res.update({
            'n_5d': int(mask5.sum()),
            'overlap_with_3d': int(inter),
            'jaccard_3d_5d': float(inter / union) if union else 0.0,
            'only_in_3d': int(len(fiducial_ids - ids5)),
            'only_in_5d': int(len(ids5 - fiducial_ids)),
        })
        df_all = df_all.copy()
        df_all['mem5d'] = mask5

    # posterior spatial filter on the 3D fiducial members
    r = angular_radius(df_mem, ra_c, dec_c)
    rt_deg = struct['radial_profile']['king_fit']['rt_deg']
    inside_rt = r <= rt_deg
    res['rt_deg'] = float(rt_deg)
    res['n_3d_total'] = int(len(df_mem))
    res['n_3d_beyond_rt'] = int((~inside_rt).sum())
    res['frac_3d_beyond_rt'] = float((~inside_rt).mean())
    # re-fit structure on the spatially-filtered subset to show stability
    sub = df_mem[inside_rt]
    fit = fit_king_radius(angular_radius(sub, ra_c, dec_c))
    dist_pc = struct['center']['dist_pc']
    deg2pc = np.pi / 180.0 * dist_pc
    res['rc_pc_spatial_filtered'] = float(fit['rc_deg'] * deg2pc)
    res['rt_pc_spatial_filtered'] = float(fit['rt_deg'] * deg2pc)
    res['rc_pc_fiducial'] = float(struct['radial_profile']['king_fit']['rc_pc'])
    res['rt_pc_fiducial'] = float(struct['radial_profile']['king_fit']['rt_pc'])
    return res, df_all


# ---------------------------------------------------------------------------
# (d) Monte-Carlo binary correction of the mass function
# ---------------------------------------------------------------------------

def sample_imf(n, alpha, m_lo=0.10, m_hi=2.60, rng=RNG):
    """Sample n masses from dN/dM propto M^-alpha by inverse transform."""
    u = rng.random(n)
    if abs(alpha - 1.0) < 1e-6:
        return m_lo * (m_hi / m_lo) ** u
    a = 1.0 - alpha
    return (m_lo ** a + u * (m_hi ** a - m_lo ** a)) ** (1.0 / a)


def observe_massfunction(masses_primary, f_bin, alpha_for_q=0.0, rng=RNG,
                         mass_shift_dex=0.0, fit_lo=0.16, fit_hi=1.2):
    """Given true primary masses, build the binary-blended, single-star-inferred
    mass function and return its fitted slope plus mass bookkeeping."""
    n = len(masses_primary)
    is_bin = rng.random(n) < f_bin
    # mass ratio q in (0.1, 1]; alpha_for_q=0 -> uniform
    q = rng.random(n) ** (1.0 / (alpha_for_q + 1.0))
    q = 0.1 + 0.9 * q
    m2 = np.where(is_bin, q * masses_primary, 0.0)
    valid2 = m2 > 0.08
    m2 = np.where(valid2, m2, 0.0)

    mg1 = mass_to_mg(masses_primary)
    flux = 10 ** (-0.4 * mg1)
    flux2 = np.where(m2 > 0, 10 ** (-0.4 * mass_to_mg(np.where(m2 > 0, m2, 0.1))), 0.0)
    mg_sys = -2.5 * np.log10(flux + flux2)

    inferred = mg_to_mass(mg_sys, mass_shift_dex=mass_shift_dex)
    true_total = masses_primary + m2

    good = np.isfinite(inferred) & (inferred > 0)
    bins = np.array([0.09, 0.12, 0.16, 0.22, 0.30, 0.45, 0.65, 0.90, 1.20, 1.70, 2.60])
    counts, edges = np.histogram(inferred[good], bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    dndm = counts / widths
    fmask = (counts > 3) & (centers >= fit_lo) & (centers <= fit_hi)
    if int(fmask.sum()) >= 3:
        coeff = np.polyfit(np.log10(centers[fmask]), np.log10(dndm[fmask]), 1)
        alpha_obs = -float(coeff[0])
    else:
        alpha_obs = np.nan
    mass_bias = float(np.nanmean(inferred[good & is_bin]) / np.nanmean(masses_primary[is_bin])) if is_bin.any() else np.nan
    return {
        'alpha_obs': alpha_obs,
        'sum_inferred': float(np.nansum(inferred[good])),
        'sum_true': float(np.nansum(true_total[good])),
        'mean_inferred': float(np.nanmean(inferred[good])),
        'mean_true': float(np.nanmean(true_total[good])),
        'binary_mass_bias': mass_bias,
    }


def binary_monte_carlo(struct, supp):
    alpha_obs_measured = supp['mass_function_binaries']['present_day_mass_function_alpha_0p16_1p2']
    n_sys = 4000
    f_bin = 0.35  # Khalaj & Baumgardt (2013) intrinsic Praesepe binary fraction
    alpha_grid = np.round(np.arange(1.0, 2.81, 0.1), 2)
    n_real = 12
    table = []
    for at in alpha_grid:
        vals = []
        for _ in range(n_real):
            mp = sample_imf(n_sys, at, rng=RNG)
            vals.append(observe_massfunction(mp, f_bin, rng=RNG)['alpha_obs'])
        table.append({'alpha_true': float(at),
                      'alpha_obs_mean': float(np.nanmean(vals)),
                      'alpha_obs_std': float(np.nanstd(vals))})
    tdf = pd.DataFrame(table)
    tdf.to_csv(os.path.join(RESULTS_DIR, 'binary_mc_massfunction.csv'), index=False)

    # invert the (alpha_true -> alpha_obs) mapping at the measured alpha_obs
    order = np.argsort(tdf['alpha_obs_mean'].to_numpy())
    alpha_true_corr = float(np.interp(alpha_obs_measured,
                                      tdf['alpha_obs_mean'].to_numpy()[order],
                                      tdf['alpha_true'].to_numpy()[order]))

    # equal-mass-sequence fraction produced by an intrinsic f_bin=0.35
    mp = sample_imf(20000, alpha_true_corr, rng=RNG)
    is_bin = RNG.random(len(mp)) < f_bin
    q = 0.1 + 0.9 * RNG.random(len(mp))
    near_equal = is_bin & (q > 0.6)
    eq_seq_fraction = float(near_equal.sum() / len(mp))

    # mass bookkeeping at the corrected slope
    book = observe_massfunction(sample_imf(20000, alpha_true_corr, rng=RNG), f_bin, rng=RNG)
    total_correction = book['sum_true'] / book['sum_inferred']
    fid_mass = float(supp['dynamical_parameters']['photometric_mass_sum_msun'])

    out = {
        'alpha_obs_measured': float(alpha_obs_measured),
        'assumed_binary_fraction': f_bin,
        'alpha_true_corrected': alpha_true_corr,
        'alpha_bias': float(alpha_obs_measured - alpha_true_corr),
        'per_binary_mass_overestimate': book['binary_mass_bias'],
        'predicted_equal_mass_sequence_fraction': eq_seq_fraction,
        'observed_equal_mass_sequence_fraction': float(supp['mass_function_binaries']['cmd_binary_fraction']),
        'total_mass_correction_factor': float(total_correction),
        'corrected_total_mass_msun': float(fid_mass * total_correction),
        'fiducial_total_mass_msun': fid_mass,
        'mc_curve': table,
    }
    return out


# ---------------------------------------------------------------------------
# (e) Ellipticity significance
# ---------------------------------------------------------------------------

def measure_ellipticity(x, y):
    x0, y0 = x - x.mean(), y - y.mean()
    ev = np.linalg.eigvalsh(np.cov(np.vstack([x0, y0])))
    ev.sort()
    return 1.0 - np.sqrt(ev[0] / ev[1])


def ellipticity_significance(df_mem, ra_c, dec_c, n_boot=2000, n_null=2000):
    x, y = tangent_xy(df_mem, ra_c, dec_c)
    e_obs = measure_ellipticity(x, y)
    coords = np.column_stack([x - x.mean(), y - y.mean()])
    r = np.hypot(coords[:, 0], coords[:, 1])

    boot = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(coords), len(coords))
        boot.append(measure_ellipticity(coords[idx, 0], coords[idx, 1]))
    boot = np.array(boot)

    # circular null: keep the radial distribution, randomise azimuth
    null = []
    for _ in range(n_null):
        phi = RNG.uniform(0, 2 * np.pi, len(r))
        null.append(measure_ellipticity(r * np.cos(phi), r * np.sin(phi)))
    null = np.array(null)

    sig_boot = float(e_obs / np.std(boot))
    sig_null = float((e_obs - np.mean(null)) / np.std(null))
    p_null = float(np.mean(null >= e_obs))
    return {
        'e_obs': float(e_obs),
        'e_bootstrap_err': float(np.std(boot)),
        'significance_over_zero_sigma': sig_boot,
        'circular_null_mean_e': float(np.mean(null)),
        'circular_null_std_e': float(np.std(null)),
        'significance_vs_circular_null_sigma': sig_null,
        'p_value_vs_circular_null': p_null,
    }


# ---------------------------------------------------------------------------
# (f) Extended systematic budget
# ---------------------------------------------------------------------------

def mass_function_alpha(df_mem, a_g=A_G, e_bp_rp=E_BP_RP, plx_zp=0.0, mass_shift_dex=0.0,
                        dist_scale=1.0):
    plx = df_mem['parallax'].to_numpy() + plx_zp
    dist_mod = 5.0 * np.log10(1000.0 / plx) - 5.0 + 5.0 * np.log10(dist_scale)
    mg = df_mem['phot_g_mean_mag'].to_numpy() - dist_mod - a_g
    mass = mg_to_mass(mg, mass_shift_dex=mass_shift_dex)
    good = np.isfinite(mass) & (mass > 0)
    bins = np.array([0.09, 0.12, 0.16, 0.22, 0.30, 0.45, 0.65, 0.90, 1.20, 1.70, 2.60])
    counts, edges = np.histogram(mass[good], bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    dndm = counts / widths
    fmask = (counts > 3) & (centers >= 0.16) & (centers <= 1.2)
    alpha = -float(np.polyfit(np.log10(centers[fmask]), np.log10(dndm[fmask]), 1)[0]) if fmask.sum() >= 3 else np.nan
    return alpha, float(np.nansum(mass[good])), int(good.sum())


def extended_systematics(df_mem, struct):
    dist_pc = struct['center']['dist_pc']
    a0, m0, n0 = mass_function_alpha(df_mem)
    rc0 = struct['radial_profile']['king_fit']['rc_pc']
    rt0 = struct['radial_profile']['king_fit']['rt_pc']
    rows = []

    def add(name, alpha, mtot, rc, rt):
        rows.append({'variant': name, 'alpha': alpha, 'total_mass_msun': mtot,
                     'rc_pc': rc, 'rt_pc': rt})

    add('fiducial', a0, m0, rc0, rt0)
    # distance +/- 4.7 pc (affects rc, rt linearly; abs mag via dist scale)
    for dd in (+4.7, -4.7):
        scale = (dist_pc + dd) / dist_pc
        a, m, _ = mass_function_alpha(df_mem, dist_scale=scale)
        add(f'distance {dd:+.1f} pc', a, m, rc0 * scale, rt0 * scale)
    # extinction E(B-V) +/- 0.01 -> A_G, E(BP-RP)
    for de in (+0.01, -0.01):
        a, m, _ = mass_function_alpha(df_mem, a_g=A_G + RG_EBV * de, e_bp_rp=E_BP_RP + RBPRP_EBV * de)
        add(f'E(B-V) {de:+.2f}', a, m, rc0, rt0)
    # isochrone mass scale +/- 0.05 dex (PARSEC vs MIST/BaSTI proxy)
    for ds in (+0.05, -0.05):
        a, m, _ = mass_function_alpha(df_mem, mass_shift_dex=ds)
        add(f'isochrone {ds:+.2f} dex', a, m, rc0, rt0)
    # parallax zero point +/- 0.02 mas
    for dz in (+0.02, -0.02):
        a, m, _ = mass_function_alpha(df_mem, plx_zp=dz)
        add(f'parallax ZP {dz:+.2f} mas', a, m, rc0, rt0)

    tdf = pd.DataFrame(rows)
    tdf.to_csv(os.path.join(RESULTS_DIR, 'systematic_budget_extended.csv'), index=False)

    # summarise half-ranges per source
    def hr(sel, col, base):
        vals = tdf[tdf['variant'].str.startswith(sel)][col].to_numpy()
        return float(0.5 * (np.nanmax(vals) - np.nanmin(vals))) if len(vals) else 0.0

    budget = {
        'alpha_fiducial': a0,
        'total_mass_fiducial': m0,
        'alpha_half_range': {
            'distance': hr('distance', 'alpha', a0),
            'extinction': hr('E(B-V)', 'alpha', a0),
            'isochrone': hr('isochrone', 'alpha', a0),
            'parallax_zp': hr('parallax', 'alpha', a0),
        },
        'total_mass_half_range': {
            'distance': hr('distance', 'total_mass_msun', m0),
            'extinction': hr('E(B-V)', 'total_mass_msun', m0),
            'isochrone': hr('isochrone', 'total_mass_msun', m0),
            'parallax_zp': hr('parallax', 'total_mass_msun', m0),
        },
        'rc_distance_half_range_pc': hr('distance', 'rc_pc', rc0),
        'rt_distance_half_range_pc': hr('distance', 'rt_pc', rt0),
        'rows': rows,
    }
    return budget


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_fig12(df_mem, df_all5, persp, m5d, ra_c, dec_c):
    x, y = tangent_xy(df_mem, ra_c, dec_c)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))

    # (a) 3D vs 5D membership on the sky
    ax = axes[0]
    if 'mem5d' in df_all5.columns:
        ids3 = set(df_mem['source_id'].astype(str))
        ids5 = set(df_all5.loc[df_all5['mem5d'], 'source_id'].astype(str))
        common = ids3 & ids5
        only3 = ids3 - ids5
        only5 = ids5 - ids3
        def xy_for(ids, frame):
            sub = frame[frame['source_id'].astype(str).isin(ids)]
            return tangent_xy(sub, ra_c, dec_c)
        cx, cy = xy_for(common, df_mem)
        o3x, o3y = xy_for(only3, df_mem)
        o5x, o5y = xy_for(only5, df_all5)
        ax.scatter(cx, cy, s=6, c='0.55', alpha=0.6, label=f'3D$\\cap$5D ({len(common)})', rasterized=True)
        ax.scatter(o3x, o3y, s=14, c='#b2182b', marker='x', label=f'3D only ({len(only3)})', rasterized=True)
        ax.scatter(o5x, o5y, s=14, c='#2166ac', marker='+', label=f'5D only ({len(only5)})', rasterized=True)
    else:
        ax.scatter(x, y, s=6, c='0.55')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(m5d['rt_deg'] * np.cos(th), m5d['rt_deg'] * np.sin(th), 'k--', lw=1.2,
            label=f"$r_t$ = {m5d['rt_deg']:.2f}$^\\circ$")
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta\alpha\cos\delta$ (deg)')
    ax.set_ylabel(r'$\Delta\delta$ (deg)')
    ax.set_title('(a) 3D vs 5D membership')
    ax.legend(fontsize=8, loc='upper right')

    # (b) perspective PM correction summary
    ax = axes[1]
    cats = [r'$\sigma_{\mu_{\alpha*}}$', r'$\sigma_{\mu_\delta}$', r'$\sigma_\mathrm{1D}$']
    unc = [persp['sigma_a_uncorr_kms'], persp['sigma_d_uncorr_kms'], persp['sigma_1d_uncorr_kms']]
    cor = [persp['sigma_a_persp_kms'], persp['sigma_d_persp_kms'], persp['sigma_1d_persp_kms']]
    xpos = np.arange(len(cats))
    ax.bar(xpos - 0.18, unc, width=0.36, color='#b2182b', alpha=0.8, label='Uncorrected')
    ax.bar(xpos + 0.18, cor, width=0.36, color='#2166ac', alpha=0.8, label='Perspective-corrected')
    for i, (u, c) in enumerate(zip(unc, cor)):
        ax.text(i - 0.18, u + 0.01, f'{u:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.18, c + 0.01, f'{c:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats)
    ax.set_ylabel(r'Velocity dispersion (km s$^{-1}$)')
    ax.set_title(f"(b) Perspective effect ($\\sigma_\\mathrm{{1D}}$: $-${100*persp['fractional_reduction']:.0f}%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    save_figure(fig, 'fig12_membership5d_perspective')


def make_fig13(binmc):
    tdf = pd.DataFrame(binmc['mc_curve'])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    ax.plot(tdf['alpha_true'], tdf['alpha_obs_mean'], 'o-', color='#2166ac')
    ax.fill_between(tdf['alpha_true'], tdf['alpha_obs_mean'] - tdf['alpha_obs_std'],
                    tdf['alpha_obs_mean'] + tdf['alpha_obs_std'], color='#2166ac', alpha=0.2)
    ax.plot([1, 2.8], [1, 2.8], '0.5', ls=':', label='1:1 (no bias)')
    ax.axhline(binmc['alpha_obs_measured'], color='#b2182b', ls='--',
               label=fr"measured $\alpha_\mathrm{{obs}}={binmc['alpha_obs_measured']:.2f}$")
    ax.axvline(binmc['alpha_true_corrected'], color='#1b7837', ls='--',
               label=fr"corrected $\alpha={binmc['alpha_true_corrected']:.2f}$")
    ax.set_xlabel(r'$\alpha_\mathrm{true}$ (intrinsic single-star slope)')
    ax.set_ylabel(r'$\alpha_\mathrm{obs}$ (single-star inferred)')
    ax.set_title(f"(a) Binary bias MC ($f_\\mathrm{{bin}}={binmc['assumed_binary_fraction']:.2f}$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels = ['Observed\n(equal-mass seq.)', 'Predicted\n($f_\\mathrm{bin}=0.35$)']
    vals = [binmc['observed_equal_mass_sequence_fraction'],
            binmc['predicted_equal_mass_sequence_fraction']]
    ax.bar([0, 1], vals, color=['#2166ac', '#b2182b'], alpha=0.8, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f'{v:.2f}', ha='center', va='bottom')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel('Near-equal-mass sequence fraction')
    ax.set_title('(b) Binary-fraction reconciliation')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    save_figure(fig, 'fig13_binary_correction')


def make_fig14(budget, ell):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    # (a) systematic half-ranges on alpha
    ax = axes[0]
    src = list(budget['alpha_half_range'].keys())
    vals = [budget['alpha_half_range'][s] for s in src]
    ax.barh(src, vals, color='#762a83', alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel(r'Half-range in $\alpha$')
    ax.set_title(f"(a) MF-slope systematics (fid. $\\alpha$={budget['alpha_fiducial']:.2f})")
    ax.grid(alpha=0.25, axis='x')

    # (b) ellipticity bootstrap vs circular null
    ax = axes[1]
    e = ell['e_obs']
    ax.axvspan(0, ell['circular_null_mean_e'] + ell['circular_null_std_e'], color='0.85',
               label='circular null $\\pm1\\sigma$')
    ax.errorbar([e], [1], xerr=[[ell['e_bootstrap_err']], [ell['e_bootstrap_err']]],
                fmt='o', color='#b2182b', capsize=4,
                label=fr"$e={e:.3f}\pm{ell['e_bootstrap_err']:.3f}$")
    ax.axvline(ell['circular_null_mean_e'], color='0.4', ls='--',
               label=fr"null $\langle e\rangle={ell['circular_null_mean_e']:.3f}$")
    ax.set_yticks([])
    ax.set_xlim(0, max(0.16, e + 3 * ell['e_bootstrap_err']))
    ax.set_xlabel('Projected ellipticity $e$')
    ax.set_title(f"(b) $e$ significance: {ell['significance_vs_circular_null_sigma']:.1f}$\\sigma$ vs null")
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    save_figure(fig, 'fig14_systematics_ellipticity')


# ---------------------------------------------------------------------------

def main():
    df_all = pd.read_csv(os.path.join(RESULTS_DIR, 'catalog_full_with_labels.csv'))
    df_mem = pd.read_csv(os.path.join(RESULTS_DIR, 'praesepe_members.csv'))
    with open(os.path.join(RESULTS_DIR, 'structure_results.json')) as f:
        struct = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'supplemental_results.json')) as f:
        supp = json.load(f)
    ra_c = struct['center']['ra']
    dec_c = struct['center']['dec']

    persp, _ = perspective_correction(df_mem)
    rv = rv_singlestar(df_mem)
    m5d, df_all5 = membership_5d(df_all, df_mem, ra_c, dec_c, struct)
    binmc = binary_monte_carlo(struct, supp)
    ell = ellipticity_significance(df_mem, ra_c, dec_c)
    budget = extended_systematics(df_mem, struct)

    make_fig12(df_mem, df_all5, persp, m5d, ra_c, dec_c)
    make_fig13(binmc)
    make_fig14(budget, ell)

    results = {
        'perspective_correction': persp,
        'rv_singlestar': rv,
        'membership_5d': m5d,
        'binary_monte_carlo': binmc,
        'ellipticity_significance': ell,
        'extended_systematics': budget,
    }
    with open(os.path.join(RESULTS_DIR, 'revision_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # concise console summary
    print('\n=== PERSPECTIVE CORRECTION ===')
    print(f"  sigma_1D: {persp['sigma_1d_uncorr_kms']:.3f} -> {persp['sigma_1d_persp_kms']:.3f} km/s "
          f"({100*persp['fractional_reduction']:.1f}% reduction); vr0={persp['vr0_kms']:.2f}")
    print('\n=== SINGLE-STAR RV ===')
    for r in rv:
        print(f"  {r['config']:24s} N={r['n']:3d} sigma_int={r['sigma_int']}")
    print('\n=== 5D MEMBERSHIP / SPATIAL FILTER ===')
    print(f"  {json.dumps(m5d, indent=0)}")
    print('\n=== BINARY MC ===')
    print(f"  alpha_obs={binmc['alpha_obs_measured']:.2f} -> alpha_true={binmc['alpha_true_corrected']:.2f} "
          f"(bias {binmc['alpha_bias']:+.2f}); total-mass x{binmc['total_mass_correction_factor']:.2f}")
    print(f"  predicted eq-mass seq frac={binmc['predicted_equal_mass_sequence_fraction']:.2f} vs observed {binmc['observed_equal_mass_sequence_fraction']:.2f}")
    print('\n=== ELLIPTICITY SIGNIFICANCE ===')
    print(f"  e={ell['e_obs']:.3f}+/-{ell['e_bootstrap_err']:.3f}; vs circular null {ell['significance_vs_circular_null_sigma']:.2f} sigma; p={ell['p_value_vs_circular_null']:.3f}")
    print('\n=== EXTENDED SYSTEMATICS (alpha half-ranges) ===')
    print(f"  {budget['alpha_half_range']}")
    print(f"  total-mass half-ranges: {budget['total_mass_half_range']}")
    print('\nSaved revision_results.json and figures fig12-fig14.')


if __name__ == '__main__':
    main()
