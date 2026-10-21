import json
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
G_CONST = 4.30091e-3

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


def load_inputs():
    df_all = pd.read_csv(os.path.join(RESULTS_DIR, 'catalog_full_with_labels.csv'))
    df_mem = pd.read_csv(os.path.join(RESULTS_DIR, 'praesepe_members.csv'))
    with open(os.path.join(RESULTS_DIR, 'structure_results.json')) as f:
        struct = json.load(f)
    return df_all, df_mem, struct


def save_figure(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'))
    plt.close(fig)


def tangent_xy(df, ra_c, dec_c):
    x = (df['ra'].to_numpy() - ra_c) * np.cos(np.radians(dec_c))
    y = df['dec'].to_numpy() - dec_c
    return x, y


def angular_radius(df, ra_c, dec_c):
    x, y = tangent_xy(df, ra_c, dec_c)
    return np.sqrt(x * x + y * y)


def absolute_cmd(df):
    dist_mod = 5.0 * np.log10(1000.0 / df['parallax'].to_numpy()) - 5.0
    mg = df['phot_g_mean_mag'].to_numpy() - dist_mod - A_G
    colour = df['bp_rp'].to_numpy() - E_BP_RP
    return colour, mg


def iso_arrays():
    colour = np.array([-0.30, -0.15, 0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20, 2.50, 2.80, 3.10, 3.40, 3.60])
    mg = np.array([-0.20, 0.50, 1.20, 1.80, 2.50, 3.20, 3.80, 4.40, 5.10, 5.80, 6.40, 7.20, 8.00, 8.80, 9.60, 10.40, 11.50, 12.30, 13.00, 13.50, 14.00])
    mass = np.array([2.60, 2.35, 2.10, 1.85, 1.60, 1.38, 1.20, 1.05, 0.92, 0.82, 0.73, 0.62, 0.52, 0.43, 0.34, 0.27, 0.20, 0.155, 0.120, 0.100, 0.090])
    return colour, mg, mass


def cmd_residuals(df):
    colour, mg = absolute_cmd(df)
    iso_c, iso_mg, _ = iso_arrays()
    iso_interp = np.interp(colour, iso_c, iso_mg, left=np.nan, right=np.nan)
    resid = mg - iso_interp
    return colour, mg, resid


def fit_empirical_cmd_sequence(df):
    colour, mg = absolute_cmd(df)
    mask = np.isfinite(colour) & np.isfinite(mg) & (colour > -0.2) & (colour < 3.7)
    c = colour[mask]
    y = mg[mask]
    keep = np.ones_like(c, dtype=bool)
    coeff = None
    for _ in range(4):
        degree = min(5, max(2, keep.sum() - 1))
        coeff = np.polyfit(c[keep], y[keep], degree)
        resid = y - np.polyval(coeff, c)
        med = np.nanmedian(resid[keep])
        mad = 1.4826 * np.nanmedian(np.abs(resid[keep] - med))
        if not np.isfinite(mad) or mad <= 0:
            break
        keep = np.abs(resid - med) < 2.8 * mad
    return coeff


def empirical_cmd_residuals(df, coeff):
    colour, mg = absolute_cmd(df)
    model = np.polyval(coeff, colour)
    resid = mg - model
    mask = np.isfinite(colour) & np.isfinite(mg) & np.isfinite(resid) & (colour > -0.2) & (colour < 3.7)
    resid = np.where(mask, resid, np.nan)
    return colour, mg, resid


def photometric_masses(df):
    _, mg = absolute_cmd(df)
    _, iso_mg, iso_mass = iso_arrays()
    order = np.argsort(iso_mg)
    log_mass = np.interp(mg, iso_mg[order], np.log10(iso_mass[order]), left=np.nan, right=np.nan)
    return 10 ** log_mass


def king_profile(r, k, rc, rt):
    term1 = 1.0 / np.sqrt(1.0 + (r / rc) ** 2)
    term2 = 1.0 / np.sqrt(1.0 + (rt / rc) ** 2)
    return np.where(r <= rt, k * (term1 - term2) ** 2, 0.0)


def fit_king_radius(radius, area_factor=1.0, n_bins=20):
    r_max = np.percentile(radius, 99)
    edges = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(radius, bins=edges)
    areas = area_factor * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = counts / areas
    err = np.sqrt(np.maximum(counts, 1)) / areas
    try:
        p0 = [np.nanmax(density), max(np.median(radius) / 2.0, 0.1), max(r_max * 1.2, 0.5)]
        bounds = ([0.0, 0.01, 0.1], [np.nanmax(density) * 20.0, 8.0, 20.0])
        popt, pcov = curve_fit(king_profile, centers, density, p0=p0, sigma=err, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        model = king_profile(centers, *popt)
        chi2 = np.sum(((density - model) / err) ** 2)
        dof = max(len(centers) - len(popt), 1)
        return {
            'k': float(popt[0]),
            'rc_deg': float(popt[1]),
            'rt_deg': float(popt[2]),
            'k_err': float(perr[0]),
            'rc_err_deg': float(perr[1]),
            'rt_err_deg': float(perr[2]),
            'chi2_red': float(chi2 / dof),
            'r_deg': centers.tolist(),
            'density': density.tolist(),
            'density_err': err.tolist(),
            'counts': counts.tolist(),
        }
    except Exception:
        return None


def method_metrics(name, selected, fiducial_ids, ra_c, dec_c, cmd_coeff):
    ids = set(selected['source_id'].astype(str))
    overlap = len(ids & fiducial_ids)
    union = len(ids | fiducial_ids)
    _, _, resid = empirical_cmd_residuals(selected, cmd_coeff)
    finite = np.isfinite(resid)
    cmd_rms = float(1.4826 * np.nanmedian(np.abs(resid[finite] - np.nanmedian(resid[finite])))) if finite.any() else np.nan
    outlier = float(np.mean(np.abs(resid[finite]) > 0.45)) if finite.any() else np.nan
    r = angular_radius(selected, ra_c, dec_c)
    return {
        'method': name,
        'n_selected': int(len(selected)),
        'overlap_with_hdbscan': int(overlap),
        'jaccard_with_hdbscan': float(overlap / union) if union else 0.0,
        'cmd_rms_mag': cmd_rms,
        'cmd_outlier_fraction': outlier,
        'median_radius_deg': float(np.nanmedian(r)) if len(r) else np.nan,
    }


def algorithm_comparison(df_all, df_mem, ra_c, dec_c, cmd_coeff):
    feature_cols = ['pmra', 'pmdec', 'parallax']
    X = df_all[feature_cols].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    target = scaler.transform([[PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER]])[0]
    fiducial_ids = set(df_mem['source_id'].astype(str))
    selections = {'HDBSCAN': df_mem.copy()}

    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42, reg_covar=1e-5)
    gmm_labels = gmm.fit_predict(Xs)
    gmm_probs = gmm.predict_proba(Xs)
    gmm_dist = np.linalg.norm(gmm.means_ - target, axis=1)
    gmm_lab = int(np.argmin(gmm_dist))
    gmm_mask = (gmm_labels == gmm_lab) & (gmm_probs[:, gmm_lab] >= 0.3)
    selections['GMM'] = df_all[gmm_mask].copy()

    best_db = None
    for eps in np.linspace(0.18, 0.60, 15):
        labels = DBSCAN(eps=float(eps), min_samples=10).fit_predict(Xs)
        labs = [lab for lab in np.unique(labels) if lab >= 0]
        for lab in labs:
            idx = labels == lab
            n = int(idx.sum())
            if n < 100 or n > 1800:
                continue
            dist = float(np.linalg.norm(Xs[idx].mean(axis=0) - target))
            score = dist + abs(n - len(df_mem)) / len(df_mem)
            if best_db is None or score < best_db[0]:
                best_db = (score, eps, lab, idx)
    if best_db is not None:
        selections['DBSCAN'] = df_all[best_db[3]].copy()
        dbscan_eps = float(best_db[1])
    else:
        selections['DBSCAN'] = df_all.iloc[[]].copy()
        dbscan_eps = np.nan

    cut_mask = (
        (np.abs(df_all['pmra'] - PM_RA_CENTER) < 3.0) &
        (np.abs(df_all['pmdec'] - PM_DEC_CENTER) < 3.0) &
        (np.abs(df_all['parallax'] - PLX_CENTER) < 0.5)
    )
    selections['Astrometric cuts'] = df_all[cut_mask].copy()

    rows = [method_metrics(name, selected, fiducial_ids, ra_c, dec_c, cmd_coeff) for name, selected in selections.items()]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, 'algorithm_comparison.csv'), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), sharex=True, sharey=True)
    for ax, (name, selected) in zip(axes.flat, selections.items()):
        ax.scatter(df_all['pmra'], df_all['pmdec'], s=2, c='0.83', alpha=0.35, rasterized=True)
        if len(selected):
            ax.scatter(selected['pmra'], selected['pmdec'], s=7, c=selected['parallax'], cmap='viridis', vmin=4.8, vmax=5.9, alpha=0.85, rasterized=True)
        row = table[table['method'] == name].iloc[0]
        ax.set_title(f"{name}: N={row['n_selected']}, J={row['jaccard_with_hdbscan']:.2f}")
        ax.set_xlim(-45, -28)
        ax.set_ylim(-19, -7)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\mu_{\alpha*}$ (mas yr$^{-1}$)')
    fig.suptitle('Membership selections from alternative astrometric methods', y=0.995)
    fig.tight_layout()
    save_figure(fig, 'fig7_algorithm_comparison')

    return {'dbscan_eps': dbscan_eps, 'metrics': rows}


def probability_and_contamination(df_all, df_mem, ra_c, dec_c, cmd_coeff):
    raw_cluster = df_all[df_all['is_member']].copy()
    thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rows = []
    for p in thresholds:
        sel = raw_cluster[raw_cluster['membership_prob'] >= p].copy()
        r = angular_radius(sel, ra_c, dec_c) if len(sel) else np.array([])
        _, _, resid = empirical_cmd_residuals(sel, cmd_coeff) if len(sel) else (None, None, np.array([]))
        rows.append({
            'probability_threshold': p,
            'n_members': int(len(sel)),
            'relative_recovery_vs_p01': float(len(sel) / max(len(raw_cluster[raw_cluster['membership_prob'] >= 0.1]), 1)),
            'cmd_outlier_fraction': float(np.nanmean(np.abs(resid) > 0.45)) if len(resid) else np.nan,
            'r90_deg': float(np.percentile(r, 90)) if len(r) else np.nan,
        })
    prob_table = pd.DataFrame(rows)
    prob_table.to_csv(os.path.join(RESULTS_DIR, 'probability_threshold_diagnostics.csv'), index=False)

    r_all = angular_radius(df_all, ra_c, dec_c)
    df_tmp = df_all.copy()
    df_tmp['radius_deg'] = r_all
    tight_astrometric = (
        (np.abs(df_tmp['pmra'] - PM_RA_CENTER) < 3.0) &
        (np.abs(df_tmp['pmdec'] - PM_DEC_CENTER) < 3.0) &
        (np.abs(df_tmp['parallax'] - PLX_CENTER) < 0.5)
    )
    bg = df_tmp[(~df_tmp['source_id'].isin(df_mem['source_id'])) & tight_astrometric & (df_tmp['radius_deg'] > 3.0)]
    mem = df_mem.copy()
    mem['radius_deg'] = angular_radius(mem, ra_c, dec_c)
    r_lim = max(np.percentile(mem['radius_deg'], 99), 0.1)
    area_member = np.pi * r_lim ** 2
    area_bg = np.pi * (max(r_all.max(), 3.1) ** 2 - 3.0 ** 2)
    mag_edges = np.array([6, 10, 12, 14, 16, 17, 18, 19, 20], dtype=float)
    mag_rows = []
    raw_cluster = raw_cluster.copy()
    for lo, hi in zip(mag_edges[:-1], mag_edges[1:]):
        n_mem = int(((mem['phot_g_mean_mag'] >= lo) & (mem['phot_g_mean_mag'] < hi)).sum())
        n_raw = int(((raw_cluster['phot_g_mean_mag'] >= lo) & (raw_cluster['phot_g_mean_mag'] < hi)).sum())
        n_bg = int(((bg['phot_g_mean_mag'] >= lo) & (bg['phot_g_mean_mag'] < hi)).sum())
        expected_bg = n_bg * area_member / max(area_bg, 1e-6)
        mag_rows.append({
            'g_min': float(lo),
            'g_max': float(hi),
            'n_members': n_mem,
            'n_raw_hdbscan_cluster': n_raw,
            'relative_completeness': float(n_mem / n_raw) if n_raw else np.nan,
            'expected_field_interlopers': float(expected_bg),
            'contamination_fraction': float(expected_bg / n_mem) if n_mem else np.nan,
        })
    mag_table = pd.DataFrame(mag_rows)
    mag_table.to_csv(os.path.join(RESULTS_DIR, 'completeness_contamination_by_mag.csv'), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].plot(prob_table['probability_threshold'], prob_table['n_members'], 'o-', color='#2166ac')
    axes[0].set_xlabel(r'$p_\mathrm{mem}$ threshold')
    axes[0].set_ylabel(r'$N_\mathrm{members}$')
    axes[0].set_title('(a) Probability threshold')
    centers = 0.5 * (mag_table['g_min'] + mag_table['g_max'])
    axes[1].plot(centers, mag_table['relative_completeness'], 's-', color='#1b7837')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel(r'$G$ (mag)')
    axes[1].set_ylabel('Relative recovery')
    axes[1].set_title('(b) Magnitude completeness proxy')
    axes[2].plot(centers, 100 * mag_table['contamination_fraction'], '^-', color='#b2182b')
    axes[2].set_xlabel(r'$G$ (mag)')
    axes[2].set_ylabel('Estimated contamination (%)')
    axes[2].set_title('(c) Field-interloper estimate')
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, 'fig8_completeness_systematics')

    return {'probability_thresholds': rows, 'magnitude_bins': mag_rows}


def nonspherical_structure(df_mem, struct, ra_c, dec_c):
    x, y = tangent_xy(df_mem, ra_c, dec_c)
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    cov = np.cov(np.vstack([x0, y0]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    q = float(np.sqrt(eigvals[1] / eigvals[0]))
    pa = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    xp = eigvecs[0, 0] * x0 + eigvecs[1, 0] * y0
    yp = eigvecs[0, 1] * x0 + eigvecs[1, 1] * y0
    ell_radius = np.sqrt(xp ** 2 + (yp / max(q, 1e-3)) ** 2)
    ell_fit = fit_king_radius(ell_radius, area_factor=q)
    circ_fit = struct['radial_profile']['king_fit']
    boot = []
    rng = np.random.default_rng(42)
    coords = np.column_stack([x0, y0])
    for _ in range(300):
        idx = rng.choice(len(coords), len(coords), replace=True)
        cov_b = np.cov(coords[idx].T)
        ev, evec = np.linalg.eigh(cov_b)
        ev = ev[np.argsort(ev)[::-1]]
        if ev[0] > 0 and ev[1] > 0:
            boot.append(1.0 - np.sqrt(ev[1] / ev[0]))
    result = {
        'axis_ratio_q': q,
        'ellipticity': 1.0 - q,
        'ellipticity_bootstrap_err': float(np.std(boot)) if boot else np.nan,
        'position_angle_deg': pa,
        'circular_rc_deg': circ_fit['rc_deg'],
        'circular_rt_deg': circ_fit['rt_deg'],
        'elliptical_rc_deg': ell_fit['rc_deg'] if ell_fit else np.nan,
        'elliptical_rt_deg': ell_fit['rt_deg'] if ell_fit else np.nan,
        'elliptical_chi2_red': ell_fit['chi2_red'] if ell_fit else np.nan,
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    h = ax.hist2d(x, y, bins=45, cmap='Greys', cmin=1)
    theta = np.linspace(0, 2 * np.pi, 300)
    for scale, ls, label in [(1.0, '--', r'1$\sigma$ ellipse'), (2.0, ':', r'2$\sigma$ ellipse')]:
        a = scale * np.sqrt(eigvals[0])
        b = scale * np.sqrt(eigvals[1])
        ellipse = np.vstack([a * np.cos(theta), b * np.sin(theta)])
        rot = eigvecs @ ellipse
        ax.plot(rot[0], rot[1], ls=ls, lw=1.5, label=label)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta\alpha\cos\delta$ (deg)')
    ax.set_ylabel(r'$\Delta\delta$ (deg)')
    ax.set_title(fr'Projected elliptical structure: e={1-q:.3f}, PA={pa:.1f}$^\circ$')
    ax.legend(loc='upper right', fontsize=9)
    fig.colorbar(h[3], ax=ax, label='Counts')
    fig.tight_layout()
    save_figure(fig, 'fig9_elliptical_structure')

    return result


def velocity_anisotropy(df_mem, struct, ra_c, dec_c):
    x, y = tangent_xy(df_mem, ra_c, dec_c)
    radius = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    pmx = df_mem['pmra'].to_numpy() - df_mem['pmra'].mean()
    pmy = df_mem['pmdec'].to_numpy() - df_mem['pmdec'].mean()
    dist_pc = struct['center']['dist_pc']
    conv = 4.74047 * dist_pc / 1000.0
    vx = pmx * conv
    vy = pmy * conv
    evx = df_mem['pmra_error'].to_numpy() * conv
    evy = df_mem['pmdec_error'].to_numpy() * conv
    vr = vx * np.cos(phi) + vy * np.sin(phi)
    vt = -vx * np.sin(phi) + vy * np.cos(phi)
    er2 = evx ** 2 * np.cos(phi) ** 2 + evy ** 2 * np.sin(phi) ** 2
    et2 = evx ** 2 * np.sin(phi) ** 2 + evy ** 2 * np.cos(phi) ** 2
    edges = np.quantile(radius, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (radius >= lo) & (radius <= hi)
        if m.sum() < 20:
            continue
        sr_obs = np.std(vr[m], ddof=1)
        st_obs = np.std(vt[m], ddof=1)
        sr = np.sqrt(max(sr_obs ** 2 - np.nanmean(er2[m]), 0.0))
        st = np.sqrt(max(st_obs ** 2 - np.nanmean(et2[m]), 0.0))
        rows.append({
            'r_inner_deg': float(lo),
            'r_outer_deg': float(hi),
            'r_mid_deg': float(0.5 * (lo + hi)),
            'n_stars': int(m.sum()),
            'sigma_radial_kms': float(sr),
            'sigma_tangential_kms': float(st),
            'beta_2d': float(1.0 - st ** 2 / sr ** 2) if sr > 0 else np.nan,
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, 'velocity_anisotropy_profile.csv'), index=False)
    vtot = np.sqrt(vx * vx + vy * vy)
    sig1d = struct['velocity_dispersion']['sigma_1d_int_kms']
    esc = df_mem[(vtot > 3.0 * sig1d)].copy()
    esc['v_residual_kms'] = vtot[vtot > 3.0 * sig1d]
    esc.to_csv(os.path.join(RESULTS_DIR, 'possible_kinematic_escapers.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].plot(table['r_mid_deg'], table['sigma_radial_kms'], 'o-', label=r'$\sigma_R$', color='#2166ac')
    axes[0].plot(table['r_mid_deg'], table['sigma_tangential_kms'], 's-', label=r'$\sigma_T$', color='#b2182b')
    axes[0].set_xlabel('Projected radius (deg)')
    axes[0].set_ylabel(r'Intrinsic dispersion (km s$^{-1}$)')
    axes[0].set_title('(a) Velocity-dispersion profile')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0, color='0.5', lw=1)
    axes[1].plot(table['r_mid_deg'], table['beta_2d'], 'D-', color='#4d9221')
    axes[1].set_xlabel('Projected radius (deg)')
    axes[1].set_ylabel(r'$\beta_\mathrm{2D}=1-\sigma_T^2/\sigma_R^2$')
    axes[1].set_title('(b) Projected anisotropy')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, 'fig10_velocity_anisotropy')

    return {'profile': rows, 'n_possible_escapers': int(len(esc)), 'escape_threshold_kms': float(3.0 * sig1d)}


def mass_function_and_binaries(df_mem, cmd_coeff):
    mass = photometric_masses(df_mem)
    colour, mg, resid = empirical_cmd_residuals(df_mem, cmd_coeff)
    valid = np.isfinite(mass) & (mass > 0)
    bins = np.array([0.09, 0.12, 0.16, 0.22, 0.30, 0.45, 0.65, 0.90, 1.20, 1.70, 2.60])
    counts, edges = np.histogram(mass[valid], bins=bins)
    widths = edges[1:] - edges[:-1]
    centers = np.sqrt(edges[:-1] * edges[1:])
    dndm = counts / widths
    fit_mask = (counts > 3) & (centers >= 0.16) & (centers <= 1.2)
    if fit_mask.sum() >= 3:
        coeff = np.polyfit(np.log10(centers[fit_mask]), np.log10(dndm[fit_mask]), 1)
        alpha = -float(coeff[0])
    else:
        alpha = np.nan
    mf_rows = []
    for lo, hi, c, cen, val in zip(edges[:-1], edges[1:], counts, centers, dndm):
        mf_rows.append({'mass_min': float(lo), 'mass_max': float(hi), 'mass_center': float(cen), 'n_stars': int(c), 'dndm': float(val)})
    pd.DataFrame(mf_rows).to_csv(os.path.join(RESULTS_DIR, 'photometric_mass_function.csv'), index=False)

    binary_mask = np.isfinite(resid) & (colour > 0.4) & (colour < 3.2) & (resid > -0.95) & (resid < -0.35)
    single_mask = np.isfinite(resid) & (colour > 0.4) & (colour < 3.2) & (np.abs(resid) <= 0.25)
    binary_fraction = float(binary_mask.sum() / max(binary_mask.sum() + single_mask.sum(), 1))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    yerr = np.sqrt(np.maximum(counts, 1)) / widths
    axes[0].errorbar(centers, dndm, yerr=yerr, fmt='o', color='#2166ac', capsize=3)
    if np.isfinite(alpha):
        xfit = np.linspace(centers[fit_mask].min(), centers[fit_mask].max(), 100)
        norm = np.nanmedian(dndm[fit_mask] * centers[fit_mask] ** alpha)
        axes[0].plot(xfit, norm * xfit ** (-alpha), '-', color='#b2182b', label=fr'$\alpha={alpha:.2f}$')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'Photometric mass ($M_\odot$)')
    axes[0].set_ylabel(r'$dN/dM$')
    axes[0].set_title('(a) Present-day mass function')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25, which='both')
    axes[1].hist(resid[np.isfinite(resid)], bins=np.linspace(-1.4, 1.2, 45), color='0.75', edgecolor='0.3')
    axes[1].axvspan(-0.95, -0.30, color='#b2182b', alpha=0.18, label='Binary-sequence window')
    axes[1].axvspan(-0.30, 0.30, color='#2166ac', alpha=0.15, label='Single-star window')
    axes[1].axvline(-0.75, color='#b2182b', ls='--', lw=1.4)
    axes[1].set_xlabel(r'$M_G-M_{G,\mathrm{iso}}$ (mag)')
    axes[1].set_ylabel('Number of stars')
    axes[1].set_title(fr'(b) CMD residuals; $f_{{bin}}\approx{binary_fraction:.2f}$')
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, 'fig11_mass_function_binary')

    return {
        'n_mass_estimates': int(valid.sum()),
        'mass_range_msun': [float(np.nanmin(mass[valid])), float(np.nanmax(mass[valid]))],
        'present_day_mass_function_alpha_0p16_1p2': alpha,
        'binary_sequence_candidates': int(binary_mask.sum()),
        'single_sequence_comparison': int(single_mask.sum()),
        'cmd_binary_fraction': binary_fraction,
    }


def dynamical_parameters(df_mem, struct):
    mass = photometric_masses(df_mem)
    total_phot_mass = float(np.nansum(mass))
    dist_pc = struct['center']['dist_pc']
    r = angular_radius(df_mem, struct['center']['ra'], struct['center']['dec']) * np.pi / 180.0 * dist_pc
    r_h = float(np.nanmedian(r))
    n = int(np.isfinite(mass).sum())
    mean_mass = float(np.nanmean(mass))
    sigma = float(struct['velocity_dispersion']['sigma_1d_int_kms'])
    crossing_time = float(r_h / sigma * 0.978) if sigma > 0 else np.nan
    t_rh = float(0.138 * np.sqrt(n) * r_h ** 1.5 / (np.sqrt(G_CONST * mean_mass) * np.log(0.4 * n)) * 0.978) if n > 10 and mean_mass > 0 else np.nan
    galactocentric_radius = 8.2
    circular_speed = 232.0
    omega = circular_speed / (galactocentric_radius * 1000.0)
    jacobi = float((G_CONST * total_phot_mass / (2.0 * omega ** 2)) ** (1.0 / 3.0))
    virial_mass = float(10.0 * r_h * sigma ** 2 / G_CONST) if sigma > 0 else np.nan
    return {
        'photometric_mass_sum_msun': total_phot_mass,
        'half_mass_radius_pc': r_h,
        'mean_photometric_mass_msun': mean_mass,
        'crossing_time_myr': crossing_time,
        'half_mass_relaxation_time_myr': t_rh,
        'age_over_trh_for_700myr': float(700.0 / t_rh) if t_rh > 0 else np.nan,
        'jacobi_radius_pc_approx': jacobi,
        'king_tidal_radius_pc': struct['radial_profile']['king_fit']['rt_pc'],
        'virial_mass_msun': virial_mass,
    }


def systematic_error_budget(df_all, struct, ra_c, dec_c):
    raw = df_all[df_all['is_member']].copy()
    rows = []
    values = []
    for p in [0.2, 0.3, 0.5, 0.7]:
        sel = raw[raw['membership_prob'] >= p].copy()
        if len(sel) < 50:
            continue
        r = angular_radius(sel, ra_c, dec_c)
        fit = fit_king_radius(r)
        x, y = tangent_xy(sel, ra_c, dec_c)
        ev = np.linalg.eigvalsh(np.cov(np.vstack([x - x.mean(), y - y.mean()])))
        ell = 1.0 - np.sqrt(ev.min() / ev.max())
        pmra = sel['pmra'].to_numpy() - sel['pmra'].mean()
        pmdec = sel['pmdec'].to_numpy() - sel['pmdec'].mean()
        conv = 4.74047 * struct['center']['dist_pc'] / 1000.0
        sig_pmra = np.sqrt(max(np.std(pmra, ddof=1) ** 2 - np.mean(sel['pmra_error'].to_numpy() ** 2), 0.0)) * conv
        sig_pmdec = np.sqrt(max(np.std(pmdec, ddof=1) ** 2 - np.mean(sel['pmdec_error'].to_numpy() ** 2), 0.0)) * conv
        sig1d = np.sqrt((sig_pmra ** 2 + sig_pmdec ** 2) / 2.0)
        item = {
            'source': f'p >= {p:.1f}',
            'n_members': int(len(sel)),
            'rc_pc': float(fit['rc_deg'] * np.pi / 180.0 * struct['center']['dist_pc']) if fit else np.nan,
            'rt_pc': float(fit['rt_deg'] * np.pi / 180.0 * struct['center']['dist_pc']) if fit else np.nan,
            'sigma_1d_kms': float(sig1d),
            'ellipticity': float(ell),
        }
        rows.append(item)
        values.append(item)
    val = pd.DataFrame(values)
    fid = val[val['source'] == 'p >= 0.3'].iloc[0]
    budget_rows = []
    for quantity in ['n_members', 'rc_pc', 'rt_pc', 'sigma_1d_kms', 'ellipticity']:
        arr = val[quantity].to_numpy(dtype=float)
        budget_rows.append({
            'quantity': quantity,
            'fiducial': float(fid[quantity]),
            'min_over_tests': float(np.nanmin(arr)),
            'max_over_tests': float(np.nanmax(arr)),
            'systematic_half_range': float(0.5 * (np.nanmax(arr) - np.nanmin(arr))),
        })
    budget = pd.DataFrame(budget_rows)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'systematic_variants.csv'), index=False)
    budget.to_csv(os.path.join(RESULTS_DIR, 'systematic_error_budget.csv'), index=False)
    return {'variants': rows, 'budget': budget_rows}


def main():
    df_all, df_mem, struct = load_inputs()
    ra_c = struct['center']['ra']
    dec_c = struct['center']['dec']
    cmd_coeff = fit_empirical_cmd_sequence(df_mem)
    results = {
        'algorithm_comparison': algorithm_comparison(df_all, df_mem, ra_c, dec_c, cmd_coeff),
        'completeness_contamination': probability_and_contamination(df_all, df_mem, ra_c, dec_c, cmd_coeff),
        'nonspherical_structure': nonspherical_structure(df_mem, struct, ra_c, dec_c),
        'velocity_anisotropy': velocity_anisotropy(df_mem, struct, ra_c, dec_c),
        'mass_function_binaries': mass_function_and_binaries(df_mem, cmd_coeff),
        'dynamical_parameters': dynamical_parameters(df_mem, struct),
        'systematic_error_budget': systematic_error_budget(df_all, struct, ra_c, dec_c),
    }
    with open(os.path.join(RESULTS_DIR, 'supplemental_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
