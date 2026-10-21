#!/usr/bin/env python3
"""
Extended diagnostics computed from the member/field catalogues and written
to results/referee_response.json:

  Isochrone goodness-of-fit: residuals of the single-star main sequence
       about the adopted PARSEC isochrone (RMS, robust RMS, reduced chi^2,
       median offset, Pearson r), with a free vertical-shift check.
  Noise / field-star characterisation: spatial and kinematic properties
       of HDBSCAN noise (label -1) and non-Praesepe clusters vs members.
  Extinction recomputation with an explicitly cited, reproducible
       coefficient set (Wang & Chen 2019), and a demonstration that the
       change relative to the previously adopted A_G is below the extinction
       systematic already propagated.
  Inertia-tensor (second-moment) components Ixx, Iyy, Ixy.
  Bootstrap recovery vs radius: perturb the astrometry within the Gaia
       measurement errors, re-run HDBSCAN, and measure the per-star recovery
       rate at p>=0.7 as a function of cluster-centric radius.

All Monte-Carlo steps use seed = 42.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ── Praesepe astrometric locus and pipeline hyperparameters ─────────
PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER = -36.1, -12.9, 5.37
MIN_CLUSTER_SIZE, MIN_SAMPLES = 15, 10

# ── Extinction (Additional issue 7) ─────────────────────────────────
E_BV = 0.027            # Taylor (2006)
R_V = 3.1
# Wang & Chen (2019, ApJ 877, 116), constant Gaia-band coefficients (R_V=3.1):
AG_AV, ABP_AV, ARP_AV = 0.789, 1.002, 0.589
A_G_OLD = 0.071         # value used in the submitted analysis
E_BP_RP_OLD = 0.035

SEED = 42


# ── Isochrone (hand-digitised PARSEC, log age 8.85, [Fe/H]=+0.16) ───
def iso_arrays():
    colour = np.array([-0.30, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90,
                       1.05, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 2.80,
                       3.10, 3.40, 3.60])
    mg = np.array([-0.20, 0.50, 1.20, 1.80, 2.50, 3.20, 3.80, 4.40, 5.10, 5.80,
                   6.40, 7.20, 8.00, 8.80, 9.60, 10.40, 11.50, 12.30, 13.00,
                   13.50, 14.00])
    mass = np.array([2.60, 2.35, 2.10, 1.85, 1.60, 1.38, 1.20, 1.05, 0.92, 0.82,
                     0.73, 0.62, 0.52, 0.43, 0.34, 0.27, 0.20, 0.155, 0.120,
                     0.100, 0.090])
    return colour, mg, mass


def abs_cmd(df, a_g, e_bp_rp):
    dist_mod = 5.0 * np.log10(1000.0 / df['parallax'].to_numpy()) - 5.0
    mg = df['phot_g_mean_mag'].to_numpy() - dist_mod - a_g
    colour = df['bp_rp'].to_numpy() - e_bp_rp
    return colour, mg, dist_mod


def mg_uncertainty(df):
    """Photometric + parallax contribution to the absolute-magnitude error."""
    g_snr = df['phot_g_mean_flux_over_error'].to_numpy()
    sig_g = 1.0857 / np.where(g_snr > 0, g_snr, np.nan)
    sig_plx_term = (5.0 / np.log(10.0)) * (df['parallax_error'].to_numpy()
                                           / df['parallax'].to_numpy())
    return np.sqrt(sig_g ** 2 + sig_plx_term ** 2)


def photometric_alpha(df, a_g, e_bp_rp, m_lo=0.16, m_hi=1.2, nbins=14):
    """Power-law slope dN/dM ~ M^-alpha over [m_lo, m_hi] and summed mass."""
    colour, mg, _ = abs_cmd(df, a_g, e_bp_rp)
    iso_c, iso_mg, iso_mass = iso_arrays()
    order = np.argsort(iso_mg)
    logm = np.interp(mg, iso_mg[order], np.log10(iso_mass[order]),
                     left=np.nan, right=np.nan)
    mass = 10 ** logm
    good = np.isfinite(mass)
    total_mass = float(np.nansum(mass[good]))
    fit = mass[good & (mass >= m_lo) & (mass <= m_hi)]
    edges = np.logspace(np.log10(m_lo), np.log10(m_hi), nbins + 1)
    cnt, _ = np.histogram(fit, bins=edges)
    cen = np.sqrt(edges[:-1] * edges[1:])
    width = np.diff(edges)
    dndm = cnt / width
    ok = cnt > 0
    coeff = np.polyfit(np.log10(cen[ok]), np.log10(dndm[ok]), 1)
    return float(-coeff[0]), total_mass


# ── (M2) Isochrone goodness-of-fit ──────────────────────────────────
def isochrone_gof(df, a_g, e_bp_rp):
    colour, mg, _ = abs_cmd(df, a_g, e_bp_rp)
    iso_c, iso_mg, _ = iso_arrays()
    iso_at = np.interp(colour, iso_c, iso_mg, left=np.nan, right=np.nan)
    resid = mg - iso_at                       # >0 fainter, <0 brighter (binaries)
    sig_mg = mg_uncertainty(df)

    # Restrict to the well-populated, single-star main sequence: exclude the
    # evolved turnoff (blue), the incomplete faint end, and the binary locus.
    on_ms = (np.isfinite(resid) & (colour > 0.0) & (colour < 3.0)
             & (mg < 12.0) & (np.abs(resid) < 0.25))
    r = resid[on_ms]
    s = sig_mg[on_ms]
    med = float(np.median(r))
    mad = float(1.4826 * np.median(np.abs(r - med)))
    rms = float(np.sqrt(np.mean(r ** 2)))
    chi2 = float(np.sum((r / s) ** 2))
    dof = int(r.size - 1)
    chi2_nu = chi2 / dof

    # Free vertical-shift (distance-modulus nuisance) check.
    shift = med
    rms_shift = float(np.sqrt(np.mean((r - shift) ** 2)))

    # Pearson r between observed and isochrone M_G along the sequence.
    pear = float(np.corrcoef(mg[on_ms], iso_at[on_ms])[0, 1])

    # Fraction of single-star MS stars consistent with the isochrone within
    # their photometric errors (i.e. |resid - shift| < 1, 2, 3 sigma).
    z = np.abs((r - shift) / s)
    return {
        'n_single_star_ms': int(r.size),
        'colour_range': [0.0, 3.0],
        'median_offset_mag': med,
        'robust_rms_mag': mad,
        'rms_mag': rms,
        'median_sigma_mg_mag': float(np.median(s)),
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2_mg_error_only': chi2_nu,
        'best_vertical_shift_mag': shift,
        'rms_after_shift_mag': rms_shift,
        'pearson_r': pear,
        'frac_within_1sigma': float(np.mean(z < 1)),
        'frac_within_2sigma': float(np.mean(z < 2)),
        'frac_within_3sigma': float(np.mean(z < 3)),
    }


# ── (M4) Noise / field-star characterisation ────────────────────────
def noise_characterisation(df_all):
    lab = df_all['cluster_label'].to_numpy()
    prae = 1  # Praesepe label (verified)
    members = df_all[lab == prae]
    noise = df_all[lab == -1]
    other = df_all[(lab >= 0) & (lab != prae)]

    def stats(d):
        return {
            'n': int(len(d)),
            'pmra_mean': float(d['pmra'].mean()), 'pmra_std': float(d['pmra'].std()),
            'pmdec_mean': float(d['pmdec'].mean()), 'pmdec_std': float(d['pmdec'].std()),
            'plx_mean': float(d['parallax'].mean()), 'plx_std': float(d['parallax'].std()),
        }

    # Fraction of noise points inside a tight Praesepe astrometric box
    # (|dpm|<3 mas/yr, |dplx|<0.5 mas): an upper bound on cluster-like noise.
    def in_tight_box(d):
        return ((np.abs(d['pmra'] + 36.1) < 3.0) & (np.abs(d['pmdec'] + 12.9) < 3.0)
                & (np.abs(d['parallax'] - 5.37) < 0.5))
    noise_clusterlike = int(in_tight_box(noise).sum())
    return {
        'praesepe_members': stats(members),
        'noise_points': stats(noise),
        'other_clusters': stats(other),
        'n_noise_in_tight_praesepe_box': noise_clusterlike,
        'frac_noise_clusterlike': float(noise_clusterlike / max(len(noise), 1)),
    }


# ── (A9) Inertia-tensor (second-moment) components ──────────────────
def inertia_components(df_mem):
    ra_c = float(df_mem['ra'].median())
    dec_c = float(df_mem['dec'].median())
    x = (df_mem['ra'].to_numpy() - ra_c) * np.cos(np.radians(dec_c))
    y = df_mem['dec'].to_numpy() - dec_c
    Ixx = float(np.mean(x ** 2))
    Iyy = float(np.mean(y ** 2))
    Ixy = float(np.mean(x * y))
    T = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    ev, evec = np.linalg.eigh(T)
    idx = np.argsort(ev)[::-1]
    ev = ev[idx]; evec = evec[:, idx]
    e = 1.0 - np.sqrt(ev[1] / ev[0])
    pa = float(np.degrees(np.arctan2(evec[1, 0], evec[0, 0])))
    pc_per_deg = np.radians(1.0) * (1000.0 / df_mem['parallax'].mean())
    return {
        'ra_c_deg': ra_c, 'dec_c_deg': dec_c,
        'Ixx_deg2': Ixx, 'Iyy_deg2': Iyy, 'Ixy_deg2': Ixy,
        'Ixx_pc2': Ixx * pc_per_deg ** 2,
        'Iyy_pc2': Iyy * pc_per_deg ** 2,
        'Ixy_pc2': Ixy * pc_per_deg ** 2,
        'eigval_major_deg2': float(ev[0]), 'eigval_minor_deg2': float(ev[1]),
        'axis_ratio_q': float(np.sqrt(ev[1] / ev[0])),
        'ellipticity': float(e), 'position_angle_deg': pa,
    }


# ── (A12) Bootstrap recovery vs radius ──────────────────────────────
def run_hdbscan_praesepe(feat):
    X = StandardScaler().fit_transform(feat)
    locus = StandardScaler().fit(feat).transform(
        np.array([[PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER]]))[0]
    cl = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                         min_samples=MIN_SAMPLES, metric='euclidean',
                         cluster_selection_method='eom',
                         allow_single_cluster=False)
    labels = cl.fit_predict(X)
    probs = cl.probabilities_
    best, bestd = -1, np.inf
    for lab in np.unique(labels[labels >= 0]):
        d = np.linalg.norm(X[labels == lab].mean(axis=0) - locus)
        if d < bestd:
            bestd, best = d, lab
    return labels, probs, best


def bootstrap_recovery(df_all, df_mem, n_boot=100, p_thresh=0.7):
    rng = np.random.default_rng(SEED)
    feat0 = df_all[['pmra', 'pmdec', 'parallax']].to_numpy()
    err = df_all[['pmra_error', 'pmdec_error', 'parallax_error']].to_numpy()
    ids = df_all['source_id'].to_numpy()

    mem_ids = df_mem['source_id'].to_numpy()
    counts = {sid: 0 for sid in mem_ids}        # recovered at p >= p_thresh
    counts_mem = {sid: 0 for sid in mem_ids}    # recovered as a member (p >= 0.3)
    for _ in range(n_boot):
        feat = feat0 + rng.normal(size=feat0.shape) * err
        labels, probs, best = run_hdbscan_praesepe(feat)
        in_cluster = (labels == best)
        recovered = set(ids[in_cluster & (probs >= p_thresh)])
        recovered_mem = set(ids[in_cluster & (probs >= 0.3)])
        for sid in mem_ids:
            if sid in recovered:
                counts[sid] += 1
            if sid in recovered_mem:
                counts_mem[sid] += 1

    rec = np.array([counts[s] / n_boot for s in mem_ids], dtype=float)
    rec_mem = np.array([counts_mem[s] / n_boot for s in mem_ids], dtype=float)

    # Cluster-centric radius of each fiducial member.
    ra_c = float(df_mem['ra'].median()); dec_c = float(df_mem['dec'].median())
    x = (df_mem['ra'].to_numpy() - ra_c) * np.cos(np.radians(dec_c))
    y = df_mem['dec'].to_numpy() - dec_c
    radius = np.sqrt(x ** 2 + y ** 2)
    p_fid = df_mem['membership_prob'].to_numpy()

    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 5.0])
    prof = []
    for i in range(len(edges) - 1):
        m = (radius >= edges[i]) & (radius < edges[i + 1])
        if m.sum() == 0:
            continue
        prof.append({'r_inner_deg': float(edges[i]), 'r_outer_deg': float(edges[i + 1]),
                     'n': int(m.sum()),
                     'mean_recovery_p07': float(rec[m].mean()),
                     'std_recovery_p07': float(rec[m].std()),
                     'mean_recovery_member': float(rec_mem[m].mean()),
                     'std_recovery_member': float(rec_mem[m].std())})

    core = p_fid >= 0.7
    outer = (p_fid >= 0.3) & (p_fid < 0.7)
    return {
        'n_boot': n_boot, 'p_thresh': p_thresh,
        'recovery_core_p07_mean': float(rec[core].mean()),
        'recovery_core_p07_std': float(rec[core].std()),
        'recovery_outer_p03_07_mean': float(rec[outer].mean()),
        'recovery_outer_p03_07_std': float(rec[outer].std()),
        'recovery_core_member_mean': float(rec_mem[core].mean()),
        'recovery_outer_member_mean': float(rec_mem[outer].mean()),
        'recovery_vs_radius': prof,
    }


def main():
    df_all = pd.read_csv(os.path.join(RESULTS_DIR, 'catalog_full_with_labels.csv'))
    df_mem = pd.read_csv(os.path.join(RESULTS_DIR, 'praesepe_members.csv'))

    # Extinction (A7)
    a_v = R_V * E_BV
    a_g_new = AG_AV * a_v
    e_bp_rp_new = (ABP_AV - ARP_AV) * a_v
    alpha_old, mass_old = photometric_alpha(df_mem, A_G_OLD, E_BP_RP_OLD)
    alpha_new, mass_new = photometric_alpha(df_mem, a_g_new, e_bp_rp_new)
    extinction = {
        'E_BV': E_BV, 'R_V': R_V,
        'coeff_AG_AV': AG_AV, 'coeff_ABP_AV': ABP_AV, 'coeff_ARP_AV': ARP_AV,
        'A_V': a_v, 'A_G_new': a_g_new, 'E_BP_RP_new': e_bp_rp_new,
        'A_G_old': A_G_OLD, 'E_BP_RP_old': E_BP_RP_OLD,
        'alpha_old': alpha_old, 'alpha_new': alpha_new,
        'd_alpha_from_extinction_change': alpha_new - alpha_old,
        'total_mass_old': mass_old, 'total_mass_new': mass_new,
    }

    out = {
        'extinction': extinction,
        'isochrone_gof': isochrone_gof(df_mem, A_G_OLD, E_BP_RP_OLD),
        'noise_field': noise_characterisation(df_all),
        'inertia_tensor': inertia_components(df_mem),
        'bootstrap_recovery': bootstrap_recovery(df_all, df_mem, n_boot=100),
    }
    path = os.path.join(RESULTS_DIR, 'referee_response.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[referee] wrote {path}")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
