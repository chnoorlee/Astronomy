#!/usr/bin/env python3
"""
Stage E-2: Membership determination using HDBSCAN clustering.

Method
------
1. Pre-filter: rough proper-motion and parallax box around Praesepe locus.
2. Feature engineering: normalise (μα*, μδ, ϖ) to unit variance.
3. HDBSCAN clustering in 3-D astrometric space.
4. Quality filter: RUWE < 1.4, parallax_over_error > 10, etc.
5. Output: member catalog with membership probability.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Known Praesepe astrometric locus (from literature) ─────────────
PM_RA_CENTER  = -36.1   # mas/yr
PM_DEC_CENTER = -12.9   # mas/yr
PLX_CENTER    =   5.37  # mas

# Pre-filter box (generous)
PM_HALF_WIDTH = 15.0    # mas/yr
PLX_HALF_WIDTH = 3.0    # mas

# Quality filters
RUWE_MAX = 1.4
PLX_OVER_ERR_MIN = 10


def load_data():
    fpath = os.path.join(DATA_DIR, 'gaia_dr3_praesepe_field.csv')
    print(f"[membership] Loading {fpath}")
    df = pd.read_csv(fpath)
    print(f"[membership] Loaded {len(df)} sources")
    return df


def quality_filter(df):
    """Apply astrometric quality cuts."""
    n0 = len(df)
    mask = (
        (df['ruwe'] < RUWE_MAX) &
        (df['parallax_over_error'] > PLX_OVER_ERR_MIN) &
        (df['phot_g_mean_flux_over_error'] > 50) &
        np.isfinite(df['pmra']) &
        np.isfinite(df['pmdec']) &
        np.isfinite(df['parallax'])
    )
    df_clean = df[mask].copy().reset_index(drop=True)
    print(f"[membership] Quality filter: {n0} → {len(df_clean)} sources")
    return df_clean


def prefilter_box(df):
    """Rough astrometric box around the Praesepe locus."""
    n0 = len(df)
    mask = (
        (np.abs(df['pmra']  - PM_RA_CENTER)  < PM_HALF_WIDTH) &
        (np.abs(df['pmdec'] - PM_DEC_CENTER) < PM_HALF_WIDTH) &
        (np.abs(df['parallax'] - PLX_CENTER)  < PLX_HALF_WIDTH)
    )
    df_box = df[mask].copy().reset_index(drop=True)
    print(f"[membership] Pre-filter box: {n0} → {len(df_box)} sources")
    return df_box


def run_hdbscan(df, min_cluster_size=15, min_samples=10):
    """
    HDBSCAN clustering in (pmra, pmdec, parallax) space.
    Returns df with 'cluster_label' and 'membership_prob' columns.
    """
    features = df[['pmra', 'pmdec', 'parallax']].values

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    print(f"[membership] Running HDBSCAN (min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}) on {len(X)} sources ...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(X)
    probs  = clusterer.probabilities_

    df = df.copy()
    df['cluster_label']  = labels
    df['membership_prob'] = probs

    unique_labels = np.unique(labels[labels >= 0])
    print(f"[membership] Found {len(unique_labels)} cluster(s); "
          f"{np.sum(labels == -1)} noise points")

    # Identify Praesepe cluster: closest to known locus in scaled space
    praesepe_label = -1
    best_dist = np.inf
    praesepe_center_scaled = scaler.transform(
        np.array([[PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER]])
    )[0]

    for lab in unique_labels:
        members = X[labels == lab]
        centroid = members.mean(axis=0)
        dist = np.linalg.norm(centroid - praesepe_center_scaled)
        if dist < best_dist:
            best_dist = dist
            praesepe_label = lab

    print(f"[membership] Praesepe cluster label = {praesepe_label}")

    df['is_member'] = (df['cluster_label'] == praesepe_label)
    members = df[df['is_member']]
    print(f"[membership] Praesepe members: {len(members)}")
    print(f"[membership]   pmra  = {members['pmra'].mean():.2f} ± {members['pmra'].std():.2f}")
    print(f"[membership]   pmdec = {members['pmdec'].mean():.2f} ± {members['pmdec'].std():.2f}")
    print(f"[membership]   plx   = {members['parallax'].mean():.3f} ± {members['parallax'].std():.3f}")

    return df, praesepe_label


def apply_probability_cut(df, prob_threshold=0.7):
    """Keep only high-probability members."""
    members = df[df['is_member']].copy()
    n0 = len(members)
    high_prob = members[members['membership_prob'] >= prob_threshold]
    print(f"[membership] Probability cut (≥{prob_threshold}): {n0} → {len(high_prob)} members")
    return high_prob


def save_results(df_all, df_members, praesepe_label):
    """Save full catalog and member catalog."""
    out_all = os.path.join(RESULTS_DIR, 'catalog_full_with_labels.csv')
    out_mem = os.path.join(RESULTS_DIR, 'praesepe_members.csv')

    df_all.to_csv(out_all, index=False)
    df_members.to_csv(out_mem, index=False)

    print(f"[membership] Saved full catalog : {out_all} ({len(df_all)} rows)")
    print(f"[membership] Saved member catalog: {out_mem} ({len(df_members)} rows)")

    # Summary statistics as JSON
    import json
    summary = {
        'n_total_sources': len(df_all),
        'n_members_raw': int(df_all['is_member'].sum()),
        'n_members_high_prob': len(df_members),
        'praesepe_label': int(praesepe_label),
        'mean_pmra': float(df_members['pmra'].mean()),
        'std_pmra': float(df_members['pmra'].std()),
        'mean_pmdec': float(df_members['pmdec'].mean()),
        'std_pmdec': float(df_members['pmdec'].std()),
        'mean_parallax': float(df_members['parallax'].mean()),
        'std_parallax': float(df_members['parallax'].std()),
        'mean_distance_pc': float(1000.0 / df_members['parallax'].mean()),
        'g_mag_range': [float(df_members['phot_g_mean_mag'].min()),
                        float(df_members['phot_g_mean_mag'].max())],
    }
    out_json = os.path.join(RESULTS_DIR, 'membership_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[membership] Saved summary: {out_json}")

    return summary


def main():
    df = load_data()
    df = quality_filter(df)
    df_box = prefilter_box(df)
    df_labeled, prae_label = run_hdbscan(df_box)
    df_members = apply_probability_cut(df_labeled, prob_threshold=0.3)
    summary = save_results(df_labeled, df_members, prae_label)

    print("\n[membership] === SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return df_labeled, df_members


if __name__ == '__main__':
    main()
