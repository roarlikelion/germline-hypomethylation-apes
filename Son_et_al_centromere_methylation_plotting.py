#!/usr/bin/env python3

"""
Son_et_al_centromere_methylation_plotting.py

Plot DNA methylation profiles across pecentromeric/satellite regions for
human blood/B-cell and sperm Oxford Nanopore methylation data.

For each region defined in `censat_ranges_f`, the script:

    1. Extracts blood/B-cell and sperm methylation measurements.
    2. Sorts measurements by genomic position.
    3. Smooths methylation levels using a 1D Gaussian filter.
    4. Adds centromeric/satellite annotations as background regions.
    5. Plots raw and smoothed methylation levels.
    6. Saves one PDF plot per genomic region.

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script expects three pandas DataFrames to already be loaded:

    human_meth
    censat_ranges_f
    censat


1. human_meth
-------------
Contains genomic methylation measurements.

Required columns:

    chr
        Chromosome name.

    stt
        Genomic position/start coordinate.

    bloodONT_methlev
        Blood/B-cell methylation level measured using Oxford Nanopore data.

    spermONT_methlev
        Sperm methylation level measured using Oxford Nanopore data.

Example:

    chr     stt       bloodONT_methlev    spermONT_methlev
    chr1    120000    0.82                 0.65
    chr1    120500    0.79                 0.68


2. censat_ranges_f
------------------
Defines the genomic regions to plot.

Required columns:

    chr
        Chromosome name.

    censat_start
        Start coordinate of the region.

    censat_end
        End coordinate of the region.

Example:

    chr     censat_start    censat_end
    chr1    120000000       126000000


3. censat
---------
Contains centromeric and satellite annotations.

Required columns:

    chr
        Chromosome name.

    stt
        Feature start coordinate.

    end
        Feature end coordinate.

    censat_group
        Centromeric/satellite feature category.

Possible feature categories include:

    nonCenSat
    ct
    censat
    inactive_hor
    active_hor
    dhor
    hsat2
    hsat3
    gsat
    bsat
    mon
    hsat1B
    hsat1A
    rDNA


Example:

    chr     stt         end         censat_group
    chr1    121000000   122000000   inactive_hor
    chr1    122000000   123500000   active_hor


---------------------------------------------------------------------------
SMOOTHING
---------------------------------------------------------------------------

Methylation values are smoothed independently for each tissue using:

    scipy.ndimage.gaussian_filter1d

with:

    sigma = 20

The Gaussian filter is applied after sorting methylation measurements
by genomic position.


---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Plots are written to:

    plots/

One PDF is generated for each region in `censat_ranges_f`.

Filename format:

    <chr>_<start>_<end>.pdf

Example:

    chr1_120000000_126000000.pdf


Each plot contains:

    - Raw blood/B-cell methylation values
    - Smoothed blood/B-cell methylation profile
    - Raw sperm methylation values
    - Smoothed sperm methylation profile
    - Centromeric/satellite annotations as background regions


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Required Python packages:

    pandas
    matplotlib
    scipy
    numpy

Install with:

    pip install pandas matplotlib scipy numpy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d


# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------

# Use Arial if it is installed on the system.
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# Embed editable TrueType fonts in PDF/PS output.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# -------------------------------------------------------------------------
# Centromeric/satellite feature colors
# -------------------------------------------------------------------------

feature_colors = {
    "nonCenSat": "#FFFFFF",
    "ct": "#FFFFFF",
    "censat": "#a6a6a6",
    "inactive_hor": "#a6a6a6",
    "active_hor": "#666666",
    "dhor": "#a6a6a6",
    "hsat2": "#a6a6a6",
    "hsat3": "#a6a6a6",
    "gsat": "#a6a6a6",
    "bsat": "#a6a6a6",
    "mon": "#a6a6a6",
    "hsat1B": "#a6a6a6",
    "hsat1A": "#a6a6a6",
    "rDNA": "#a6a6a6",
}


# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


# -------------------------------------------------------------------------
# Generate one methylation plot for each centromeric/satellite region
# -------------------------------------------------------------------------

for _, row in censat_ranges_f.iterrows():

    chr_name = row["chr"]
    start = row["censat_start"]
    end = row["censat_end"]


    # ---------------------------------------------------------------------
    # Extract blood/B-cell methylation measurements
    # ---------------------------------------------------------------------

    region_df_bcell = human_meth[
        (human_meth["chr"] == chr_name)
        & (human_meth["stt"] >= start)
        & (human_meth["stt"] <= end)
        & (human_meth["bloodONT_methlev"].notna())
    ].copy()


    # ---------------------------------------------------------------------
    # Extract sperm methylation measurements
    # ---------------------------------------------------------------------

    region_df_sperm = human_meth[
        (human_meth["chr"] == chr_name)
        & (human_meth["stt"] >= start)
        & (human_meth["stt"] <= end)
        & (human_meth["spermONT_methlev"].notna())
    ].copy()


    # Skip regions without methylation data in either tissue.
    if region_df_bcell.empty and region_df_sperm.empty:
        print(
            f"Skipping {chr_name}, "
            "no methylation data found in range."
        )
        continue


    # ---------------------------------------------------------------------
    # Smooth blood/B-cell methylation
    # ---------------------------------------------------------------------

    if not region_df_bcell.empty:

        # Sort methylation measurements by genomic position.
        region_df_bcell = region_df_bcell.sort_values("stt")

        # Apply Gaussian smoothing.
        region_df_bcell["smoothed_bloodONT_methlev"] = (
            gaussian_filter1d(
                region_df_bcell["bloodONT_methlev"].to_numpy(),
                sigma=20,
            )
        )


    # ---------------------------------------------------------------------
    # Smooth sperm methylation
    # ---------------------------------------------------------------------

    if not region_df_sperm.empty:

        # Sort methylation measurements by genomic position.
        region_df_sperm = region_df_sperm.sort_values("stt")

        # Apply Gaussian smoothing.
        region_df_sperm["smoothed_spermONT_methlev"] = (
            gaussian_filter1d(
                region_df_sperm["spermONT_methlev"].to_numpy(),
                sigma=20,
            )
        )


    # ---------------------------------------------------------------------
    # Extract centromeric/satellite annotations overlapping the region
    # ---------------------------------------------------------------------

    censat_filtered = censat[
        (censat["chr"] == chr_name)
        & (censat["end"] >= start)
        & (censat["stt"] <= end)
    ].copy()


    # ---------------------------------------------------------------------
    # Create plot
    # ---------------------------------------------------------------------

    fig, ax1 = plt.subplots(
        1,
        1,
        figsize=(17, 5),
        sharex=True,
    )


    # Add centromeric/satellite annotations as background spans.
    for _, frow in censat_filtered.iterrows():

        color = feature_colors.get(
            frow["censat_group"],
            "#a6a6a6",
        )

        ax1.axvspan(
            frow["stt"],
            frow["end"],
            color=color,
            alpha=1,
        )


    # ---------------------------------------------------------------------
    # Plot blood/B-cell methylation
    # ---------------------------------------------------------------------

    if not region_df_bcell.empty:

        # Smoothed methylation profile.
        ax1.plot(
            region_df_bcell["stt"],
            region_df_bcell["smoothed_bloodONT_methlev"],
            label="BCell (smoothed)",
            linewidth=0.7,
            color="#d52854",
        )

        # Raw methylation measurements.
        ax1.scatter(
            region_df_bcell["stt"],
            region_df_bcell["bloodONT_methlev"],
            label="BCell (raw)",
            s=4,
            color="#e16887",
            alpha=0.5,
            rasterized=True,
        )


    # ---------------------------------------------------------------------
    # Plot sperm methylation
    # ---------------------------------------------------------------------

    if not region_df_sperm.empty:

        # Smoothed methylation profile.
        ax1.plot(
            region_df_sperm["stt"],
            region_df_sperm["smoothed_spermONT_methlev"],
            label="spermONT (smoothed)",
            linewidth=0.7,
            color="#00B894",
        )

        # Raw methylation measurements.
        ax1.scatter(
            region_df_sperm["stt"],
            region_df_sperm["spermONT_methlev"],
            label="spermONT (raw)",
            s=4,
            color="#66d4be",
            alpha=0.5,
            rasterized=True,
        )


    # ---------------------------------------------------------------------
    # Figure formatting
    # ---------------------------------------------------------------------

    ax1.set_ylabel("Methylation")
    ax1.set_ylim(0, 1)

    ax1.set_title(
        f"DNA Methylation in {chr_name}:{start}-{end}"
    )

    ax1.legend(
        frameon=False,
        ncol=2,
    )

    plt.tight_layout()


    # ---------------------------------------------------------------------
    # Save PDF
    # ---------------------------------------------------------------------

    pdf_filename = (
        f"{chr_name}_{start}_{end}.pdf"
    )

    plt.savefig(
        os.path.join(
            output_dir,
            pdf_filename,
        )
    )

    plt.close()