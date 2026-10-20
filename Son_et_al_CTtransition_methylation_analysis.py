#!/usr/bin/env python3

"""
Son_et_al_CTtransition_methylation_analysis.py

Analyze DNA methylation levels relative to the nearest centromere-transition
(CT) boundary.

The script performs two analyses:

1. Methylation profile around CT transitions
   - Uses positions from -200 kb to +200 kb relative to the closest CT transition.
   - Groups methylation measurements into 1-kb bins.
   - Calculates the mean methylation level and standard deviation for
     sperm and B-cell samples.
   - Plots raw methylation measurements, binned means, and ±1 standard
     deviation around each mean.

2. Sperm vs B-cell methylation significance
   - Uses positions from 0 to +200 kb relative to the CT transition.
   - Groups methylation measurements into 5-kb bins.
   - Performs Welch's independent two-sample t-test between sperm and
     B-cell methylation levels within each bin.
   - Plots log10(P-value) across distance from the CT transition.

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script expects a pandas DataFrame named:

    merged_df

with at least the following columns:

    distance_from_closestCTtransition
        Distance in base pairs from each genomic window/measurement to
        the closest CT transition (Satellite - nonSatellite transition region).

        Negative values:
            positions before the CT transition

        Positive values:
            positions after the CT transition

    sperm2_methlev
        DNA methylation level for sperm.

    bcell2_methlev
        DNA methylation level for B cells.

Example input structure:

    distance_from_closestCTtransition    sperm2_methlev    bcell2_methlev
    -15000                               0.72              0.81
    -10000                               0.69              0.78
    5000                                 0.55              0.71
    10000                                0.51              0.68

The DataFrame can be loaded before running the analysis, for example:

    merged_df = pd.read_csv("CT_transition_methylation.tsv", sep="\\t")


---------------------------------------------------------------------------
ANALYSIS 1: METHYLATION PROFILE
---------------------------------------------------------------------------

Range:

    -200,000 bp to +200,000 bp

Bin size:

    1,000 bp

For each bin, the script calculates:

    mean sperm methylation
    standard deviation of sperm methylation

    mean B-cell methylation
    standard deviation of B-cell methylation


---------------------------------------------------------------------------
ANALYSIS 2: STATISTICAL TEST
---------------------------------------------------------------------------

Range:

    0 to +200,000 bp

Bin size:

    5,000 bp

For each bin, sperm and B-cell methylation levels are compared using:

    scipy.stats.ttest_ind(..., equal_var=False)

This corresponds to Welch's independent two-sample t-test.

The plotted values are:

    log10(P-value)

Values below -10 are capped at -10 for visualization.

A reference line is shown at:

    P = 1e-6


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Required Python packages:

    pandas
    numpy
    matplotlib
    scipy

Install with:

    pip install pandas numpy matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind


# -------------------------------------------------------------------------
# Plot settings
# -------------------------------------------------------------------------

# Arial will be used if it is installed on the system.
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# Embed editable TrueType fonts in PDF/PS output.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================================
# Analysis 1: Methylation profile around CT transitions
# =========================================================================

# Use 1-kb bins across ±200 kb from the closest CT transition.
bin_size_2 = 1000
range_2 = (-200000, 200000)

bins_2 = np.arange(
    range_2[0],
    range_2[1] + bin_size_2,
    bin_size_2
)


# Assign each methylation measurement to a distance bin.
merged_df["bin_2"] = pd.cut(
    merged_df["distance_from_closestCTtransition"],
    bins_2
)


# -------------------------------------------------------------------------
# Calculate methylation statistics for each distance bin
# -------------------------------------------------------------------------

grouped = merged_df.groupby("bin_2")

# Mean methylation level per bin.
mean_sperm_2 = grouped["sperm2_methlev"].mean()
mean_bcell_2 = grouped["bcell2_methlev"].mean()

# Standard deviation per bin.
std_sperm_2 = grouped["sperm2_methlev"].std()
std_bcell_2 = grouped["bcell2_methlev"].std()


# Use the midpoint of each interval as the x-coordinate.
bin_centers_2 = [
    interval.mid
    for interval in mean_sperm_2.index
]


# -------------------------------------------------------------------------
# Plot methylation profiles
# -------------------------------------------------------------------------

plt.figure(figsize=(9, 3))


# Sperm mean ± standard deviation.
plt.fill_between(
    bin_centers_2,
    mean_sperm_2.values - std_sperm_2.values,
    mean_sperm_2.values + std_sperm_2.values,
    color="#006883",
    alpha=0.3,
    label="_nolegend_",
)


# B-cell mean ± standard deviation.
plt.fill_between(
    bin_centers_2,
    mean_bcell_2.values - std_bcell_2.values,
    mean_bcell_2.values + std_bcell_2.values,
    color="#53061a",
    alpha=0.3,
    label="_nolegend_",
)


# -------------------------------------------------------------------------
# Plot raw methylation measurements
# -------------------------------------------------------------------------

plt.scatter(
    merged_df["distance_from_closestCTtransition"],
    merged_df["bcell2_methlev"],
    alpha=0.1,
    s=0.5,
    label="_nolegend_",
    color="#d11141",
    rasterized=True,
)

plt.scatter(
    merged_df["distance_from_closestCTtransition"],
    merged_df["sperm2_methlev"],
    alpha=0.1,
    s=0.5,
    label="_nolegend_",
    color="#00aedb",
    rasterized=True,
)


# -------------------------------------------------------------------------
# Plot binned mean methylation levels
# -------------------------------------------------------------------------

plt.plot(
    bin_centers_2,
    mean_sperm_2.values,
    color="#006883",
    linewidth=3,
    label="Sperm2 Mean",
)

plt.plot(
    bin_centers_2,
    mean_bcell_2.values,
    color="#53061a",
    linewidth=3,
    label="Bcell2 Mean",
)


# -------------------------------------------------------------------------
# Figure formatting
# -------------------------------------------------------------------------

plt.xlabel("Distance from closest CT transition")
plt.ylabel("Methylation Level")

plt.xlim(range_2)

plt.title(
    "Distance vs Methylation Level (±200 kb)"
)

plt.grid(True)
plt.legend()

plt.tight_layout()


# Uncomment to save the figure.
#
# plt.savefig(
#     "transition.pdf",
#     bbox_inches="tight",
#     format="pdf",
#     transparent=True,
# )

plt.show()


# =========================================================================
# Analysis 2: Statistical significance across 5-kb bins
# =========================================================================

# Analyze the region from the CT transition to +200 kb.
bin_size = 5000
range_test = (0, 200000)

bins = np.arange(
    range_test[0],
    range_test[1] + bin_size,
    bin_size
)


# -------------------------------------------------------------------------
# Subset measurements within the selected distance range
# -------------------------------------------------------------------------

df_sub = merged_df[
    (merged_df["distance_from_closestCTtransition"] >= range_test[0])
    & (merged_df["distance_from_closestCTtransition"] <= range_test[1])
].copy()


# Assign measurements to 5-kb bins.
df_sub["bin"] = pd.cut(
    df_sub["distance_from_closestCTtransition"],
    bins
)


# -------------------------------------------------------------------------
# Compare sperm and B-cell methylation within each bin
# -------------------------------------------------------------------------

results = []

for b, g in df_sub.groupby("bin"):

    sperm_vals = g["sperm2_methlev"].dropna()
    bcell_vals = g["bcell2_methlev"].dropna()

    # Perform Welch's t-test when both groups contain
    # at least two methylation measurements.
    if len(sperm_vals) > 1 and len(bcell_vals) > 1:

        _, p = ttest_ind(
            sperm_vals,
            bcell_vals,
            equal_var=False,
        )

    else:
        p = np.nan


    results.append(
        {
            "bin_center": b.mid,
            "pvalue": p,
        }
    )


# Convert statistical results to a DataFrame.
pvals_df = pd.DataFrame(results)


# -------------------------------------------------------------------------
# Transform P-values for plotting
# -------------------------------------------------------------------------

# Convert P-values to log10 scale.
yvals = np.log10(
    pvals_df["pvalue"]
)

# Cap values below -10 for visualization.
yvals = np.clip(
    yvals,
    -10,
    None,
)


# -------------------------------------------------------------------------
# Plot statistical significance
# -------------------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(
    pvals_df["bin_center"],
    yvals,
    "-o",
    markersize=5,
    color="black",
)


# Reference threshold: P = 1e-6.
plt.axhline(
    np.log10(0.000001),
    color="red",
    linestyle="--",
    linewidth=1,
)


# -------------------------------------------------------------------------
# Figure formatting
# -------------------------------------------------------------------------

plt.xlabel("Distance (bp)")
plt.ylabel("log10(P-value)")

plt.title(
    "Significance Across 5 kb Bins (Sperm vs Bcell)"
)

plt.ylim(-10.5, 0)

plt.grid(alpha=0.4)

plt.tight_layout()


# Uncomment to save the figure.
#
# plt.savefig(
#     "transition_significance.pdf",
#     bbox_inches="tight",
#     format="pdf",
#     transparent=True,
# )

plt.show()