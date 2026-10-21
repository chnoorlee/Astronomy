#!/usr/bin/env python3
"""
Build the machine-readable member catalogue for publication / Zenodo.

Following the AAS data-editor guidance, we do NOT replicate the Gaia DR3
database. The published table contains only:
  - the Gaia DR3 source_id (full long integer, never truncated),
  - the sky positions (ra, dec), and
  - the quantities we calculate (membership probability, cluster-centric
    radius, photometric mass),
each at a precision consistent with its uncertainty. Raw Gaia columns
(parallax, proper motions, photometry, RUWE, ...) are dropped; readers can
recover them at full precision by joining on source_id.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Extinction (Wang & Chen 2019, R_V=3.1) -- consistent with the analysis.
E_BV = 0.027
_A_V = 3.1 * E_BV
A_G = 0.789 * _A_V
E_BP_RP = (1.002 - 0.589) * _A_V


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


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'praesepe_members.csv'))

    # Verify source_id integrity (full 64-bit integers, never floats).
    assert np.issubdtype(df['source_id'].dtype, np.integer), 'source_id not integer!'
    assert (df['source_id'] > 1e17).all(), 'suspiciously short source_id'

    # Calculated columns -------------------------------------------------
    ra_c = float(df['ra'].median())
    dec_c = float(df['dec'].median())
    x = (df['ra'].to_numpy() - ra_c) * np.cos(np.radians(dec_c))
    y = df['dec'].to_numpy() - dec_c
    r_deg = np.sqrt(x ** 2 + y ** 2)

    dist_mod = 5.0 * np.log10(1000.0 / df['parallax'].to_numpy()) - 5.0
    mg = df['phot_g_mean_mag'].to_numpy() - dist_mod - A_G
    _, iso_mg, iso_mass = iso_arrays()
    order = np.argsort(iso_mg)
    logm = np.interp(mg, iso_mg[order], np.log10(iso_mass[order]),
                     left=np.nan, right=np.nan)
    mass = 10 ** logm

    out = pd.DataFrame({
        'source_id': df['source_id'].astype('int64'),         # full long int
        'ra_deg': df['ra'].round(6),                           # ~ mas
        'dec_deg': df['dec'].round(6),
        'membership_prob': df['membership_prob'].round(3),     # 3 dp is ample
        'r_cluster_deg': np.round(r_deg, 4),                   # calculated
        'mass_phot_msun': np.round(mass, 3),                   # calculated
    })
    out = out.sort_values('source_id').reset_index(drop=True)

    path = os.path.join(RESULTS_DIR, 'praesepe_members_published.csv')
    out.to_csv(path, index=False)
    print(f'[publish] wrote {path}: {len(out)} rows, {out.shape[1]} columns')
    print(f'[publish] columns: {list(out.columns)}')
    print(f'[publish] center (ra,dec) = ({ra_c:.5f}, {dec_c:.5f})')
    print(out.head(4).to_string(index=False))
    print(f'[publish] n with finite mass: {np.isfinite(out["mass_phot_msun"]).sum()}')


if __name__ == '__main__':
    main()
