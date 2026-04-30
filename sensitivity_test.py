#!/usr/bin/env python3
"""
Run actual HDBSCAN parameter sensitivity test.
Varies min_cluster_size and min_samples, reports member counts.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan

sys.path.insert(0, os.path.dirname(__file__))
from membership import load_data, quality_filter, prefilter_box, PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def run_sensitivity():
    df = load_data()
    df = quality_filter(df)
    df_box = prefilter_box(df)

    features = df_box[['pmra', 'pmdec', 'parallax']].values
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    praesepe_center_scaled = scaler.transform(
        np.array([[PM_RA_CENTER, PM_DEC_CENTER, PLX_CENTER]])
    )[0]

    param_grid = [
        (10, 5), (10, 10), (15, 5), (15, 10), (15, 15),
        (20, 10), (20, 15), (25, 10), (30, 15),
    ]

    results = []
    for mcs, ms in param_grid:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            metric='euclidean', cluster_selection_method='eom',
            allow_single_cluster=False,
        )
        labels = clusterer.fit_predict(X)
        probs = clusterer.probabilities_

        # Find Praesepe cluster
        unique_labels = np.unique(labels[labels >= 0])
        best_label, best_dist = -1, np.inf
        for lab in unique_labels:
            centroid = X[labels == lab].mean(axis=0)
            d = np.linalg.norm(centroid - praesepe_center_scaled)
            if d < best_dist:
                best_dist, best_label = d, lab

        if best_label >= 0:
            prae_mask = labels == best_label
            n_raw = int(prae_mask.sum())
            n_p03 = int(((prae_mask) & (probs >= 0.3)).sum())
            n_p05 = int(((prae_mask) & (probs >= 0.5)).sum())
            n_p07 = int(((prae_mask) & (probs >= 0.7)).sum())
            n_clusters = len(unique_labels)

            # Compute mean astrometry for Praesepe members (p>=0.3)
            idx = np.where((prae_mask) & (probs >= 0.3))[0]
            mean_plx = float(df_box.iloc[idx]['parallax'].mean())
            mean_pmra = float(df_box.iloc[idx]['pmra'].mean())
        else:
            n_raw = n_p03 = n_p05 = n_p07 = 0
            n_clusters = len(unique_labels)
            mean_plx = mean_pmra = np.nan

        row = {
            'min_cluster_size': mcs, 'min_samples': ms,
            'n_clusters': n_clusters, 'n_raw': n_raw,
            'n_p03': n_p03, 'n_p05': n_p05, 'n_p07': n_p07,
            'mean_plx': round(mean_plx, 3), 'mean_pmra': round(mean_pmra, 2),
        }
        results.append(row)
        print(f"  mcs={mcs:2d}, ms={ms:2d} → {n_clusters} clusters, "
              f"N_raw={n_raw}, N(p≥0.3)={n_p03}, N(p≥0.5)={n_p05}, N(p≥0.7)={n_p07}")

    out = os.path.join(RESULTS_DIR, 'hdbscan_sensitivity.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    return results

if __name__ == '__main__':
    run_sensitivity()
