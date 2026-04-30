#!/usr/bin/env python3
"""
Main pipeline: Download → Membership → Structure → Figures
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("  Praesepe (M44) Study Pipeline — Gaia DR3 + HDBSCAN")
print("=" * 60)

# Stage 1: Download
print("\n>>> STAGE 1: Data Download")
import download_data
download_data.main() if hasattr(download_data, 'main') else None

# Stage 2: Membership
print("\n>>> STAGE 2: Membership Determination")
import membership
membership.main()

# Stage 3: Structure
print("\n>>> STAGE 3: Structural Analysis")
import structure
structure.main()

# Stage 4: Figures
print("\n>>> STAGE 4: Generate Figures")
import plot_figures
plot_figures.main()

print("\n" + "=" * 60)
print("  Pipeline complete.")
print("=" * 60)
