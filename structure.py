#!/usr/bin/env python3
"""
Stage E-3: Structural and dynamical analysis of Praesepe members.

Analyses:
  1. Radial density profile & King model fitting
  2. Mass segregation analysis (using G magnitude as mass proxy)
  3. Tidal tail search via spatial elongation and position-angle analysis
  4. Colour-magnitude diagram properties
  5. Velocity dispersion
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── Praesepe center (updated from membership analysis) ─────────────
RA_CENTER  = 130.05  # will be updated from data
DEC_CENTER = 19.67
DIST_PC    = 186.0   # approximate


def load_members():
    fpath = os.path.join(RESULTS_DIR, 'praesepe_members.csv')
    df = pd.read_csv(fpath)
    print(f"[structure] Loaded {len(df)} members")
    return df


def angular_separation(ra1, dec1, ra2, dec2):
    """Angular separation in degrees (small-angle approximation sufficient here)."""
    dra = (ra1 - ra2) * np.cos(np.radians(dec2))
    ddec = dec1 - dec2
    return np.sqrt(dra**2 + ddec**2)


def update_center(df):
    """Compute median center of members."""
    ra_c = np.median(df['ra'])
    dec_c = np.median(df['dec'])
    print(f"[structure] Updated center: RA={ra_c:.4f}, Dec={dec_c:.4f}")
    return ra_c, dec_c


# ── 1. Radial Density Profile & King Model ──────────────────────────

def king_profile(r, k, rc, rt):
    """
    King (1962) surface density profile:
      f(r) = k * [ 1/sqrt(1+(r/rc)^2) - 1/sqrt(1+(rt/rc)^2) ]^2
    for r <= rt, else 0.
    """
    term1 = 1.0 / np.sqrt(1.0 + (r / rc)**2)
    term2 = 1.0 / np.sqrt(1.0 + (rt / rc)**2)
    val = k * (term1 - term2)**2
    return np.where(r <= rt, val, 0.0)


def radial_density_profile(df, ra_c, dec_c, n_bins=20):
    """Compute radial density profile in concentric annuli."""
    sep = angular_separation(df['ra'].values, df['dec'].values, ra_c, dec_c)

    r_max = np.percentile(sep, 99)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts, _ = np.histogram(sep, bins=bin_edges)
    # Annular area in deg^2
    areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    density = counts / areas  # stars / deg^2
    density_err = np.sqrt(counts) / areas

    # Fit King profile
    try:
        # Initial guesses
        p0 = [density.max(), 0.3, r_max * 0.8]
        bounds = ([0, 0.01, 0.1], [density.max() * 10, 5.0, 15.0])
        popt, pcov = curve_fit(king_profile, bin_centers, density,
                               p0=p0, sigma=density_err + 1e-10,
                               bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        k_fit, rc_fit, rt_fit = popt
        k_err, rc_err, rt_err = perr

        # Convert to physical units
        deg2pc = (np.pi / 180.0) * DIST_PC
        rc_pc = rc_fit * deg2pc
        rt_pc = rt_fit * deg2pc
        rc_pc_err = rc_err * deg2pc
        rt_pc_err = rt_err * deg2pc

        # Goodness of fit: chi-squared
        y_model = king_profile(bin_centers, k_fit, rc_fit, rt_fit)
        residuals = density - y_model
        chi2 = np.sum((residuals / (density_err + 1e-10))**2)
        dof = len(bin_centers) - 3  # 3 free parameters
        chi2_red = chi2 / dof

        king_result = {
            'k': float(k_fit), 'k_err': float(k_err),
            'rc_deg': float(rc_fit), 'rc_err_deg': float(rc_err),
            'rt_deg': float(rt_fit), 'rt_err_deg': float(rt_err),
            'rc_pc': float(rc_pc), 'rt_pc': float(rt_pc),
            'rc_pc_err': float(rc_pc_err), 'rt_pc_err': float(rt_pc_err),
            'concentration': float(np.log10(rt_fit / rc_fit)),
            'chi2': float(chi2), 'chi2_red': float(chi2_red), 'dof': int(dof),
        }
        print(f"[structure] King fit: rc={rc_fit:.3f}±{rc_err:.3f}°, "
              f"rt={rt_fit:.2f}±{rt_err:.2f}°")
        print(f"[structure]   rc={rc_pc:.2f}±{rc_pc_err:.2f} pc, rt={rt_pc:.1f}±{rt_pc_err:.1f} pc, "
              f"c=log(rt/rc)={king_result['concentration']:.2f}")
        print(f"[structure]   χ²/dof = {chi2:.1f}/{dof} = {chi2_red:.2f}")
    except Exception as e:
        print(f"[structure] King fit failed: {e}")
        king_result = None

    profile = {
        'r_deg': bin_centers.tolist(),
        'density': density.tolist(),
        'density_err': density_err.tolist(),
        'counts': counts.tolist(),
        'king_fit': king_result,
    }
    return profile, sep


# ── 2. Mass Segregation ─────────────────────────────────────────────

def mass_segregation_ratio(df, ra_c, dec_c, n_massive=20, n_random=100):
    """
    Λ_MSR (Allison et al. 2009):
    Compare the MST length of the N brightest stars to
    random sets of N stars drawn from all members.

    We use 2D projected positions for MST.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    # Project to tangent plane (small-angle)
    x = (df['ra'].values - ra_c) * np.cos(np.radians(dec_c))
    y = df['dec'].values - dec_c
    coords = np.column_stack([x, y])
    g_mag = df['phot_g_mean_mag'].values

    def mst_length(positions):
        D = squareform(pdist(positions))
        tree = minimum_spanning_tree(D)
        return tree.sum()

    # MST of N brightest
    idx_bright = np.argsort(g_mag)[:n_massive]
    l_massive = mst_length(coords[idx_bright])

    # Random draws
    rng = np.random.default_rng(42)
    l_random_list = []
    for _ in range(n_random):
        idx_rand = rng.choice(len(coords), size=n_massive, replace=False)
        l_random_list.append(mst_length(coords[idx_rand]))

    l_random = np.array(l_random_list)
    l_random_mean = l_random.mean()
    l_random_std = l_random.std()

    msr = l_random_mean / l_massive
    msr_err = l_random_std / l_massive

    print(f"[structure] Mass segregation ratio Λ_MSR = {msr:.2f} ± {msr_err:.2f} "
          f"(N_massive={n_massive})")

    # Also compute MSR for different N_massive values
    msr_vs_n = []
    for n_m in [5, 10, 15, 20, 30, 50]:
        if n_m > len(coords) - 5:
            break
        idx_b = np.argsort(g_mag)[:n_m]
        l_m = mst_length(coords[idx_b])
        l_r_list = []
        for _ in range(n_random):
            idx_r = rng.choice(len(coords), size=n_m, replace=False)
            l_r_list.append(mst_length(coords[idx_r]))
        l_r = np.array(l_r_list)
        msr_vs_n.append({
            'n_massive': n_m,
            'lambda_msr': float(l_r.mean() / l_m),
            'lambda_msr_err': float(l_r.std() / l_m),
        })

    return {'lambda_msr': float(msr), 'lambda_msr_err': float(msr_err),
            'n_massive': n_massive, 'msr_vs_n': msr_vs_n}


# ── 3. Tidal Tail Analysis ──────────────────────────────────────────

def tidal_tail_analysis(df, ra_c, dec_c):
    """
    Analyse spatial elongation and search for tidal features.
    Use inertia tensor / PCA of projected positions.
    """
    x = (df['ra'].values - ra_c) * np.cos(np.radians(dec_c))
    y = df['dec'].values - dec_c

    # Inertia tensor (2D)
    Ixx = np.sum(x**2) / len(x)
    Iyy = np.sum(y**2) / len(y)
    Ixy = np.sum(x * y) / len(x)

    # Eigenvalues and eigenvectors
    T = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eigh(T)
    # Sort by descending eigenvalue
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]

    ellipticity = 1.0 - np.sqrt(eigvals[1] / eigvals[0])
    position_angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    print(f"[structure] Spatial ellipticity: {ellipticity:.3f}")
    print(f"[structure] Position angle of major axis: {position_angle:.1f}°")

    # Radial bins for elongation analysis
    sep = np.sqrt(x**2 + y**2)
    radial_bins = np.linspace(0, np.percentile(sep, 95), 6)
    elongation_profile = []
    for i in range(len(radial_bins) - 1):
        mask = (sep >= radial_bins[i]) & (sep < radial_bins[i+1])
        if mask.sum() < 10:
            continue
        x_bin, y_bin = x[mask], y[mask]
        T_bin = np.array([[np.var(x_bin), np.cov(x_bin, y_bin)[0,1]],
                          [np.cov(x_bin, y_bin)[0,1], np.var(y_bin)]])
        ev = np.linalg.eigvalsh(T_bin)
        ev.sort()
        elong = 1.0 - np.sqrt(ev[0] / max(ev[1], 1e-20))
        elongation_profile.append({
            'r_inner_deg': float(radial_bins[i]),
            'r_outer_deg': float(radial_bins[i+1]),
            'n_stars': int(mask.sum()),
            'ellipticity': float(elong),
        })

    return {
        'ellipticity': float(ellipticity),
        'position_angle_deg': float(position_angle),
        'eigval_major': float(eigvals[0]),
        'eigval_minor': float(eigvals[1]),
        'axis_ratio': float(np.sqrt(eigvals[1] / eigvals[0])),
        'elongation_profile': elongation_profile,
    }


# ── 4. Velocity Dispersion ─────────────────────────────────────────

def velocity_dispersion(df):
    """Compute proper motion velocity dispersion with measurement error correction."""
    pmra_mean = df['pmra'].mean()
    pmdec_mean = df['pmdec'].mean()

    # Observed dispersion (subtract mean motion)
    dpmra = df['pmra'] - pmra_mean
    dpmdec = df['pmdec'] - pmdec_mean

    sigma_pmra_obs = dpmra.std()
    sigma_pmdec_obs = dpmdec.std()

    # Subtract measurement errors in quadrature: σ²_int = σ²_obs - <σ²_err>
    mean_err2_pmra = (df['pmra_error']**2).mean()
    mean_err2_pmdec = (df['pmdec_error']**2).mean()

    sigma_pmra_int2 = max(sigma_pmra_obs**2 - mean_err2_pmra, 0)
    sigma_pmdec_int2 = max(sigma_pmdec_obs**2 - mean_err2_pmdec, 0)
    sigma_pmra_int = np.sqrt(sigma_pmra_int2)
    sigma_pmdec_int = np.sqrt(sigma_pmdec_int2)

    # Convert to km/s using average distance
    plx_mean = df['parallax'].mean()
    dist_pc = 1000.0 / plx_mean
    kms_factor = 4.74047 * dist_pc / 1000.0

    sigma_ra_kms_obs = sigma_pmra_obs * kms_factor
    sigma_dec_kms_obs = sigma_pmdec_obs * kms_factor
    sigma_1d_obs = np.sqrt(sigma_ra_kms_obs**2 + sigma_dec_kms_obs**2) / np.sqrt(2)

    sigma_ra_kms_int = sigma_pmra_int * kms_factor
    sigma_dec_kms_int = sigma_pmdec_int * kms_factor
    sigma_1d_int = np.sqrt(sigma_ra_kms_int**2 + sigma_dec_kms_int**2) / np.sqrt(2)

    # Radial velocity dispersion with 3-sigma clipping
    rv_valid = df.dropna(subset=['radial_velocity']).copy()
    rv_disp_raw = None
    rv_disp_clipped = None
    mean_rv = None
    n_rv = len(rv_valid)
    if n_rv > 10:
        rv = rv_valid['radial_velocity'].values
        rv_med = np.median(rv)
        rv_mad = np.median(np.abs(rv - rv_med)) * 1.4826  # MAD-based sigma
        clip_mask = np.abs(rv - rv_med) < 3.0 * rv_mad
        rv_clipped = rv[clip_mask]
        rv_disp_raw = float(rv.std())
        rv_disp_clipped = float(rv_clipped.std())
        mean_rv = float(rv_clipped.mean())
        n_rv_clipped = len(rv_clipped)
        print(f"[structure] RV raw dispersion: {rv_disp_raw:.2f} km/s ({n_rv} stars)")
        print(f"[structure] RV clipped (3σ) dispersion: {rv_disp_clipped:.2f} km/s ({n_rv_clipped} stars)")

    result = {
        'mean_pmra': float(pmra_mean),
        'mean_pmdec': float(pmdec_mean),
        'sigma_pmra_obs_mas_yr': float(sigma_pmra_obs),
        'sigma_pmdec_obs_mas_yr': float(sigma_pmdec_obs),
        'sigma_pmra_int_mas_yr': float(sigma_pmra_int),
        'sigma_pmdec_int_mas_yr': float(sigma_pmdec_int),
        'mean_pmra_err': float(np.sqrt(mean_err2_pmra)),
        'mean_pmdec_err': float(np.sqrt(mean_err2_pmdec)),
        'sigma_1d_obs_kms': float(sigma_1d_obs),
        'sigma_1d_int_kms': float(sigma_1d_int),
        'dist_pc': float(dist_pc),
        'n_rv_stars': n_rv,
        'rv_dispersion_raw_kms': rv_disp_raw,
        'rv_dispersion_clipped_kms': rv_disp_clipped,
        'mean_rv_kms': mean_rv,
    }
    print(f"[structure] PM observed dispersion: σ_μα*={sigma_pmra_obs:.3f}, σ_μδ={sigma_pmdec_obs:.3f} mas/yr")
    print(f"[structure] PM intrinsic dispersion: σ_μα*={sigma_pmra_int:.3f}, σ_μδ={sigma_pmdec_int:.3f} mas/yr")
    print(f"[structure] 1D velocity dispersion (observed): {sigma_1d_obs:.2f} km/s")
    print(f"[structure] 1D velocity dispersion (intrinsic): {sigma_1d_int:.2f} km/s")

    return result


# ── 5. CMD Statistics ───────────────────────────────────────────────

def cmd_analysis(df):
    """Compute CMD-related statistics."""
    bp_rp = df['bp_rp'].dropna()
    g_mag = df['phot_g_mean_mag']
    plx = df['parallax']

    # Absolute G magnitude
    dist_mod = 5.0 * np.log10(1000.0 / plx) - 5.0
    abs_g = g_mag - dist_mod

    result = {
        'n_cmd_stars': int(len(bp_rp)),
        'bp_rp_range': [float(bp_rp.min()), float(bp_rp.max())],
        'g_mag_range': [float(g_mag.min()), float(g_mag.max())],
        'abs_g_range': [float(abs_g.min()), float(abs_g.max())],
        'median_distance_modulus': float(dist_mod.median()),
    }
    print(f"[structure] CMD: {len(bp_rp)} stars, BP-RP=[{bp_rp.min():.2f}, {bp_rp.max():.2f}]")
    print(f"[structure]   M_G range: [{abs_g.min():.1f}, {abs_g.max():.1f}]")
    return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    df = load_members()

    # Update center
    ra_c, dec_c = update_center(df)
    global DIST_PC
    DIST_PC = 1000.0 / df['parallax'].mean()
    print(f"[structure] Mean distance: {DIST_PC:.1f} pc")

    # Run all analyses
    print("\n=== Radial Density Profile ===")
    profile, sep = radial_density_profile(df, ra_c, dec_c)

    print("\n=== Mass Segregation ===")
    msr = mass_segregation_ratio(df, ra_c, dec_c)

    print("\n=== Tidal Tail Analysis ===")
    tidal = tidal_tail_analysis(df, ra_c, dec_c)

    print("\n=== Velocity Dispersion ===")
    vel = velocity_dispersion(df)

    print("\n=== CMD Analysis ===")
    cmd = cmd_analysis(df)

    # Save all results
    all_results = {
        'center': {'ra': float(ra_c), 'dec': float(dec_c), 'dist_pc': float(DIST_PC)},
        'radial_profile': profile,
        'mass_segregation': msr,
        'tidal_structure': tidal,
        'velocity_dispersion': vel,
        'cmd': cmd,
    }

    out_json = os.path.join(RESULTS_DIR, 'structure_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[structure] All results saved to {out_json}")

    return all_results


if __name__ == '__main__':
    main()
