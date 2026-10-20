#!/usr/bin/env python3

"""
Son_et_al_CTtransition_methylation_threshold_plot.py

Plot sperm and blood DNA methylation profiles relative to centromere-
transition (CT) boundaries for each chromosome arm and estimate the
distance over which sperm methylation remains significantly lower than
blood methylation.

The analysis:

    1. Tests sperm vs blood methylation in overlapping 10-kb windows.
    2. Uses a one-sided Welch's t-test:
           sperm methylation < blood methylation
    3. Applies Benjamini-Hochberg FDR correction across all tested windows.
    4. Defines a significance boundary using consecutive non-significant
       windows.
    5. Calculates mean ± standard deviation methylation in 5-kb bins.
    6. Generates one panel per chromosome arm.
    7. Saves window-level statistical results as a TSV file.

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script expects a pandas DataFrame named:

    merged_df

Required columns:

1. Chromosome arm
-----------------

chr

Expected naming convention:

    chr1_p
    chr1_q
    chr2_p
    chr2_q
    ...
    chr22_p
    chr22_q
    chrX_p
    chrX_q
    chrY_p
    chrY_q


2. Distance from CT transition
------------------------------

distance_from_closestCTtransition

Distance in base pairs from the closest CT transition.

Negative values:
    positions before the CT transition

Positive values:
    positions after the CT transition


3. DNA methylation
------------------

spermONT_methlev

    Sperm DNA methylation level from Oxford Nanopore data.

bloodONT_methlev

    Blood DNA methylation level from Oxford Nanopore data.


Example input:

    chr      distance_from_closestCTtransition    spermONT_methlev    bloodONT_methlev
    chr1_p   -5000                                0.61                0.78
    chr1_p    5000                                0.52                0.75
    chr1_p   10000                                0.48                0.73


The DataFrame can be loaded before running this analysis, for example:

    merged_df = pd.read_csv("CT_transition_methylation.tsv", sep="\\t")


---------------------------------------------------------------------------
SIGNIFICANCE ANALYSIS
---------------------------------------------------------------------------

Distance range:

    0 to 400,000 bp

Window size:

    10,000 bp

Window overlap:

    5,000 bp

Window step:

    5,000 bp

Minimum observations per group:

    5

Statistical test:

    Welch's independent two-sample t-test

Alternative hypothesis:

    spermONT_methlev < bloodONT_methlev

Multiple-testing correction:

    Benjamini-Hochberg FDR across all valid chromosome/window tests

Significance threshold:

    q < 0.05


---------------------------------------------------------------------------
SIGNIFICANCE BOUNDARY
---------------------------------------------------------------------------

For each chromosome arm, sufficiently powered windows are examined
from the CT transition outward.

Significant window:
    q < 0.05

Non-significant window:
    q >= 0.05

Underpowered windows:
    skipped entirely and do not count as failures

The scan stops after:

    2 consecutive sufficiently powered non-significant windows

The significance boundary is defined as the midpoint between:

    center of the last significant window

and:

    center of the first non-significant window in the terminating run

If significance continues through all tested windows, the center of the
last significant window is used.

If no significant window is detected, no red threshold line is plotted.


---------------------------------------------------------------------------
METHYLATION PROFILE
---------------------------------------------------------------------------

Plotting range:

    -50,000 to +350,000 bp

Methylation values are summarized in:

    5,000-bp bins

For each bin, the plot shows:

    - mean sperm methylation
    - ±1 standard deviation for sperm
    - mean blood methylation
    - ±1 standard deviation for blood
    - raw methylation measurements


---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Figure:

    fig4_threshold.pdf

Window-level statistical results:

    fig4_threshold_window_statistics.tsv


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Required Python packages:

    numpy
    pandas
    matplotlib
    scipy
    statsmodels

Install with:

    pip install numpy pandas matplotlib scipy statsmodels
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------

# Arial will be used if installed on the system.
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# Embed editable TrueType fonts in PDF/PS output.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================================
# Significance-analysis parameters
# =========================================================================

# Sliding-window size used for statistical testing.
sig_bin_size = 10000

# Overlap between adjacent statistical windows.
sig_overlap = 5000

# Distance between starts of consecutive windows.
sig_step = sig_bin_size - sig_overlap

# Test significance from the CT transition to +400 kb.
sig_range = (0, 400000)

# FDR significance threshold.
q_threshold = 0.05

# Stop after this many consecutive sufficiently powered
# non-significant windows.
max_consecutive_fail = 2

# Minimum number of measurements required in each group.
min_n = 5


# =========================================================================
# Main methylation-profile parameters
# =========================================================================

# Bin size used to calculate mean methylation profiles.
main_bin_size = 5000

# Genomic range displayed in each panel.
main_range = (-50000, 350000)


# =========================================================================
# Generate statistical and plotting bins
# =========================================================================

sig_starts = np.arange(
    sig_range[0],
    sig_range[1] - sig_bin_size + sig_step,
    sig_step,
)

main_bins = np.arange(
    main_range[0],
    main_range[1] + main_bin_size,
    main_bin_size,
)


# =========================================================================
# Define chromosome-arm order
# =========================================================================

chromosomes = [
    f"chr{i}_{arm}"
    for i in range(1, 23)
    for arm in ["p", "q"]
] + [
    "chrX_p",
    "chrX_q",
    "chrY_p",
    "chrY_q",
]


# Keep only chromosome arms present in the input data.
chromosomes = [
    chrom
    for chrom in chromosomes
    if chrom in merged_df["chr"].dropna().unique()
]


# =========================================================================
# 1. Calculate sliding-window P-values
# =========================================================================

all_window_results = []


for chrom in chromosomes:

    # Select current chromosome arm.
    chr_df = merged_df[
        merged_df["chr"] == chrom
    ].copy()


    # Restrict significance testing to 0-400 kb and rows with both
    # sperm and blood methylation measurements.
    df_sub = chr_df[
        (
            chr_df["distance_from_closestCTtransition"]
            >= sig_range[0]
        )
        & (
            chr_df["distance_from_closestCTtransition"]
            <= sig_range[1]
        )
    ].dropna(
        subset=[
            "spermONT_methlev",
            "bloodONT_methlev",
        ]
    ).copy()


    # Test each overlapping distance window.
    for start in sig_starts:

        end = start + sig_bin_size


        # Select measurements inside the current window.
        g = df_sub[
            (
                df_sub["distance_from_closestCTtransition"]
                >= start
            )
            & (
                df_sub["distance_from_closestCTtransition"]
                < end
            )
        ]


        sperm_vals = g[
            "spermONT_methlev"
        ].dropna()

        blood_vals = g[
            "bloodONT_methlev"
        ].dropna()


        # Require at least min_n measurements in each group.
        enough_n = (
            len(sperm_vals) >= min_n
            and len(blood_vals) >= min_n
        )


        if enough_n:

            # One-sided Welch's t-test.
            #
            # Alternative hypothesis:
            #
            #     sperm methylation < blood methylation
            _, p = ttest_ind(
                sperm_vals,
                blood_vals,
                equal_var=False,
                alternative="less",
            )

        else:

            p = np.nan


        all_window_results.append(
            {
                "chr": chrom,
                "bin_start": start,
                "bin_end": end,
                "bin_center": (start + end) / 2,
                "n_sperm": len(sperm_vals),
                "n_blood": len(blood_vals),
                "enough_n": enough_n,
                "pvalue": p,
            }
        )


# Convert window-level results to a DataFrame.
pvals_df = (
    pd.DataFrame(
        all_window_results
    )
    .sort_values(
        [
            "chr",
            "bin_center",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================================
# 2. Benjamini-Hochberg FDR correction
# =========================================================================

pvals_df["qvalue"] = np.nan

valid = pvals_df[
    "pvalue"
].notna()


# Apply BH-FDR to all valid windows across chromosome arms.
if valid.sum() > 0:

    _, qvals, _, _ = multipletests(
        pvals_df.loc[
            valid,
            "pvalue",
        ],
        method="fdr_bh",
    )

    pvals_df.loc[
        valid,
        "qvalue",
    ] = qvals


# Save window-level statistical results.
pvals_df.to_csv(
    "fig4_threshold_window_statistics.tsv",
    sep="\t",
    index=False,
)


# =========================================================================
# 3. Generate chromosome-arm plots
# =========================================================================

ncols = 6

nrows = math.ceil(
    len(chromosomes) / ncols
)


fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(28, 4 * nrows),
    sharex=True,
    sharey=True,
)


axes = np.array(
    axes
).reshape(-1)


for ax, chrom in zip(
    axes,
    chromosomes,
):

    chr_df = merged_df[
        merged_df["chr"] == chrom
    ].copy()


    chr_pvals = (
        pvals_df[
            pvals_df["chr"] == chrom
        ]
        .sort_values(
            "bin_center"
        )
        .reset_index(
            drop=True
        )
    )


    # =====================================================================
    # Determine significance boundary
    # =====================================================================

    fail_count = 0

    # Center of the most recent significant window.
    last_sig_center = None

    # First failure in the terminating non-significant run.
    first_fail_center = None

    break_center = None


    for _, row in chr_pvals.iterrows():

        q = row[
            "qvalue"
        ]

        center = row[
            "bin_center"
        ]

        enough_n = row[
            "enough_n"
        ]


        # Underpowered windows are ignored completely.
        if not enough_n:
            continue


        # Determine whether the window passes the FDR threshold.
        is_sig = (
            pd.notna(q)
            and q < q_threshold
        )


        if is_sig:

            # Record the most recent significant window.
            last_sig_center = center

            # Reset the failure counter.
            fail_count = 0
            first_fail_center = None


        else:

            fail_count += 1


            # Record the first failure in the current run.
            if fail_count == 1:
                first_fail_center = center


            # Stop after the required number of consecutive
            # non-significant tested windows.
            if (
                fail_count
                >= max_consecutive_fail
            ):

                break_center = (
                    first_fail_center
                )

                break


    # =====================================================================
    # Calculate significance midpoint
    # =====================================================================

    if q_threshold == 0:

        sig_midpoint = 0


    elif last_sig_center is not None:

        if break_center is not None:

            sig_midpoint = (
                last_sig_center
                + break_center
            ) / 2

        else:

            sig_midpoint = (
                last_sig_center
            )


    else:

        # No significant sufficiently powered windows.
        sig_midpoint = np.nan


    # =====================================================================
    # Calculate binned methylation profiles
    # =====================================================================

    chr_df["main_bin"] = pd.cut(
        chr_df[
            "distance_from_closestCTtransition"
        ],
        main_bins,
        include_lowest=True,
    )


    grouped = chr_df.groupby(
        "main_bin",
        observed=False,
    )


    # Sperm mean and standard deviation.
    means_sperm = grouped[
        "spermONT_methlev"
    ].mean()

    std_sperm = grouped[
        "spermONT_methlev"
    ].std()


    # Blood mean and standard deviation.
    means_bloodONT = grouped[
        "bloodONT_methlev"
    ].mean()

    std_bloodONT = grouped[
        "bloodONT_methlev"
    ].std()


    # Midpoint of each 5-kb plotting bin.
    bin_centers = np.array(
        [
            interval.mid
            for interval in means_sperm.index
        ]
    )


    # =====================================================================
    # Plot mean ± standard deviation
    # =====================================================================

    # Sperm variability.
    ax.fill_between(
        bin_centers,
        means_sperm.values
        - std_sperm.values,
        means_sperm.values
        + std_sperm.values,
        color="#00B894",
        alpha=0.3,
    )


    # Blood variability.
    ax.fill_between(
        bin_centers,
        means_bloodONT.values
        - std_bloodONT.values,
        means_bloodONT.values
        + std_bloodONT.values,
        color="#d52854",
        alpha=0.3,
    )


    # =====================================================================
    # Plot raw methylation measurements
    # =====================================================================

    ax.scatter(
        chr_df[
            "distance_from_closestCTtransition"
        ],
        chr_df[
            "bloodONT_methlev"
        ],
        alpha=0.5,
        s=1,
        color="#e16887",
        rasterized=True,
    )


    ax.scatter(
        chr_df[
            "distance_from_closestCTtransition"
        ],
        chr_df[
            "spermONT_methlev"
        ],
        alpha=0.5,
        s=1,
        color="#66d4be",
        rasterized=True,
    )


    # =====================================================================
    # Plot mean methylation profiles
    # =====================================================================

    ax.plot(
        bin_centers,
        means_sperm.values,
        color="#00B894",
        linewidth=2,
        label="spermONT Mean",
    )


    ax.plot(
        bin_centers,
        means_bloodONT.values,
        color="#d52854",
        linewidth=2,
        label="bloodONT Mean",
    )


    # CT transition position.
    ax.axvline(
        0,
        color="black",
        linewidth=1.5,
    )


    # =====================================================================
    # Panel formatting
    # =====================================================================

    ax.set_title(
        str(chrom)
    )

    ax.set_xlim(
        main_range
    )

    ax.set_ylim(
        -0.1,
        1.1,
    )

    ax.grid(
        True,
        alpha=0.3,
    )


    # =====================================================================
    # Plot significance boundary
    # =====================================================================

    if pd.notna(
        sig_midpoint
    ):

        ax.axvline(
            sig_midpoint,
            color="red",
            linestyle="--",
            linewidth=1.5,
        )


        y_top = ax.get_ylim()[1]


        ax.text(
            sig_midpoint,
            y_top,
            f"{sig_midpoint / 1000:.1f} kb",
            color="red",
            ha="center",
            va="bottom",
            fontsize=14,
            rotation=90,
        )


# =========================================================================
# Turn off unused panels
# =========================================================================

for ax in axes[
    len(chromosomes):
]:
    ax.axis(
        "off"
    )


# =========================================================================
# Shared axis labels
# =========================================================================

for ax in axes[
    -ncols:
]:
    ax.set_xlabel(
        "Distance from closest CT transition"
    )


for i in range(
    0,
    len(axes),
    ncols,
):

    if i < len(chromosomes):

        axes[i].set_ylabel(
            "Methylation Level"
        )


# =========================================================================
# Shared legend
# =========================================================================

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95,
    ]
)


# =========================================================================
# Save figure
# =========================================================================

plt.savefig(
    "fig4_threshold.pdf",
    bbox_inches="tight",
    format="pdf",
    transparent=True,
)


plt.show()