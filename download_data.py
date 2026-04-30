#!/usr/bin/env python3
"""
Stage E-1: Download Gaia DR3 data for the Praesepe (M44 / NGC 2632) field.

Praesepe parameters (literature):
  - Center: RA=130.05°, Dec=+19.67° (J2000)
  - Distance: ~186 pc  =>  parallax ~5.37 mas
  - Proper motion: μα*≈−36.1 mas/yr, μδ≈−12.9 mas/yr
  - Tidal radius: ~12 pc (~3.7°)

We query a 10° cone to capture potential tidal-tail members, with a
generous parallax cut (2–10 mas) and G<20.5.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, 'gaia_dr3_praesepe_field.csv')

# ── Praesepe field parameters ──────────────────────────────────────
RA_CENTER  = 130.05   # deg
DEC_CENTER = 19.67    # deg
SEARCH_RADIUS = 10.0  # deg  (large to catch tidal tails)
PARALLAX_MIN = 2.0    # mas
PARALLAX_MAX = 10.0   # mas
G_MAG_LIMIT  = 20.5

ADQL_QUERY = f"""
SELECT
    source_id, ra, ra_error, dec, dec_error,
    parallax, parallax_error, parallax_over_error,
    pmra, pmra_error, pmdec, pmdec_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    bp_rp, bp_g, g_rp,
    phot_g_mean_flux_over_error,
    radial_velocity, radial_velocity_error,
    ruwe,
    astrometric_excess_noise, astrometric_excess_noise_sig,
    l, b
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {RA_CENTER}, {DEC_CENTER}, {SEARCH_RADIUS}))
  AND parallax BETWEEN {PARALLAX_MIN} AND {PARALLAX_MAX}
  AND phot_g_mean_mag < {G_MAG_LIMIT}
  AND parallax_over_error > 5
"""


def download_gaia_data():
    """Submit async TAP query to Gaia archive and save result."""
    from astroquery.gaia import Gaia

    print(f"[download] Querying Gaia DR3 archive ...")
    print(f"[download]   Center : ({RA_CENTER}, {DEC_CENTER})")
    print(f"[download]   Radius : {SEARCH_RADIUS}°")
    print(f"[download]   Parallax: {PARALLAX_MIN}–{PARALLAX_MAX} mas")
    print(f"[download]   G < {G_MAG_LIMIT}")
    t0 = time.time()

    job = Gaia.launch_job_async(ADQL_QUERY)
    results = job.get_results()

    elapsed = time.time() - t0
    print(f"[download] Received {len(results)} rows in {elapsed:.1f}s")

    # Convert to pandas and save
    df = results.to_pandas()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[download] Saved to {OUTPUT_FILE}")
    print(f"[download] Columns: {list(df.columns)}")
    print(f"[download] File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")

    return df


if __name__ == '__main__':
    if os.path.exists(OUTPUT_FILE) and '--force' not in sys.argv:
        print(f"[download] Data file already exists: {OUTPUT_FILE}")
        print(f"[download] Use --force to re-download.")
        df = pd.read_csv(OUTPUT_FILE)
        print(f"[download] Loaded {len(df)} rows.")
    else:
        df = download_gaia_data()

    # Quick sanity check
    print(f"\n[download] === Quick Stats ===")
    print(f"  RA  range : {df['ra'].min():.2f} – {df['ra'].max():.2f}")
    print(f"  Dec range : {df['dec'].min():.2f} – {df['dec'].max():.2f}")
    print(f"  Parallax  : {df['parallax'].median():.2f} ± {df['parallax'].std():.2f} mas (median ± std)")
    print(f"  G mag     : {df['phot_g_mean_mag'].min():.1f} – {df['phot_g_mean_mag'].max():.1f}")
