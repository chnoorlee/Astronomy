#!/usr/bin/env python3
"""
Stage E-1 (v2): Download Gaia DR3 data for Praesepe using synchronous query.
Uses a 5-degree cone (sufficient for core + tidal structure) with tight parallax cuts.
"""

import os
import sys
import time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, 'gaia_dr3_praesepe_field.csv')

RA_CENTER  = 130.05
DEC_CENTER = 19.67
SEARCH_RADIUS = 5.0   # reduced from 10 to 5 degrees
PARALLAX_MIN = 3.0
PARALLAX_MAX = 8.0
G_MAG_LIMIT  = 20.0

ADQL_QUERY = f"""
SELECT TOP 200000
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

def main():
    if os.path.exists(OUTPUT_FILE) and '--force' not in sys.argv:
        import pandas as pd
        df = pd.read_csv(OUTPUT_FILE)
        print(f"[download] Already exists: {len(df)} rows")
        return df

    from astroquery.gaia import Gaia

    print(f"[download] Querying Gaia DR3 (sync, 5° cone) ...")
    t0 = time.time()

    job = Gaia.launch_job(ADQL_QUERY)
    results = job.get_results()

    elapsed = time.time() - t0
    print(f"[download] Received {len(results)} rows in {elapsed:.1f}s")

    import pandas as pd
    df = results.to_pandas()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[download] Saved to {OUTPUT_FILE}")
    print(f"[download] File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")

    # Quick stats
    print(f"\n[download] === Quick Stats ===")
    print(f"  RA  range : {df['ra'].min():.2f} – {df['ra'].max():.2f}")
    print(f"  Dec range : {df['dec'].min():.2f} – {df['dec'].max():.2f}")
    print(f"  Parallax  : {df['parallax'].median():.2f} ± {df['parallax'].std():.2f} mas")
    print(f"  G mag     : {df['phot_g_mean_mag'].min():.1f} – {df['phot_g_mean_mag'].max():.1f}")
    return df

if __name__ == '__main__':
    main()
