#!/usr/bin/env python3

"""
Son_et_al_CTtransition_pairwise_significance_distance.py

Estimate the distance from centromere-transition (CT) boundaries over which
DNA methylation levels remain significantly different between pairs of
samples/tissues.

The analysis uses overlapping genomic-distance windows, one-sided Welch's
t-tests, Benjamini-Hochberg FDR correction, and a consecutive-failure rule
to estimate the extent of significant methylation differences.

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script expects a pandas DataFrame named:

    merged_df

Each row should represent a methylation measurement or genomic window.

Required columns:

1. Chromosome/arm
-----------------

chr

Expected chromosome naming convention:

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

Distance in base pairs from the closest centromere-transition boundary.

This analysis uses:

    0 to 400,000 bp


3. Methylation-level columns
----------------------------

Required columns depend on the pairwise comparisons below.

Current comparisons use:

    sperm2_methlev
    bcell2_methlev
    colon_methlev
    hair_methlev
    spermONT_methlev
    bloodONT_methlev


Example input structure:

    chr      distance_from_closestCTtransition   sperm2_methlev   bcell2_methlev
    chr1_p   5000                                0.42             0.68
    chr1_p   7500                                0.45             0.71
    chr1_p   12000                               0.48             0.69


The DataFrame can be loaded before running this analysis, for example:

    merged_df = pd.read_csv("CT_transition_methylation.tsv", sep="\\t")


---------------------------------------------------------------------------
WINDOW PARAMETERS
---------------------------------------------------------------------------

Window size:

    10,000 bp

Window overlap:

    5,000 bp

Step size:

    5,000 bp

Distance range:

    0 to 400,000 bp

Therefore, consecutive windows are:

    0-10 kb
    5-15 kb
    10-20 kb
    15-25 kb
    ...


---------------------------------------------------------------------------
STATISTICAL TEST
---------------------------------------------------------------------------

For each chromosome arm, comparison, and distance window, methylation
levels are compared using a one-sided Welch's t-test:

    scipy.stats.ttest_ind(
        group1,
        group2,
        equal_var=False,
        alternative="less"
    )

The alternative hypothesis is:

    group1 methylation < group2 methylation

A window must contain at least:

    5 measurements in group1
    5 measurements in group2

to be tested.

Windows with fewer observations are treated as underpowered and are
excluded from the significance/failure counting procedure.


---------------------------------------------------------------------------
MULTIPLE-TESTING CORRECTION
---------------------------------------------------------------------------

For each pairwise comparison, all valid window P-values across chromosome
arms are corrected together using:

    Benjamini-Hochberg FDR

Significance threshold:

    q < 0.05


---------------------------------------------------------------------------
SIGNIFICANCE-DISTANCE RULE
---------------------------------------------------------------------------

For each chromosome arm, windows are examined in increasing distance from
the CT transition.

A significant window:

    q < 0.05

resets the consecutive-failure counter.

A sufficiently powered but non-significant window increments the failure
counter.

Underpowered windows are skipped and do not count as significant windows
or failures.

The scan stops after:

    2 consecutive sufficiently powered non-significant windows

The estimated significance boundary is the midpoint between:

    center of the last significant window

and:

    center of the first non-significant window in the terminating run

If no significant window is observed, the reported distance is:

    0


---------------------------------------------------------------------------
PAIRWISE COMPARISONS
---------------------------------------------------------------------------

Current comparisons:

    sperm2_methlev vs bcell2_methlev
    sperm2_methlev vs colon_methlev
    sperm2_methlev vs hair_methlev
    spermONT_methlev vs bloodONT_methlev


---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Two tab-delimited files are generated:

1. Base-pair units:

    chromosome_sig_midpoints_all_pairwise_bp_strict.tsv

2. Kilobase units:

    chromosome_sig_midpoints_all_pairwise_kb_strict.tsv

Rows represent chromosome arms.

Columns represent pairwise methylation comparisons.

Values represent the estimated distance from the CT transition over which
a significant methylation difference is detected.


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Required Python packages:

    numpy
    pandas
    scipy
    statsmodels

"""

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# =========================================================================
# Analysis parameters
# =========================================================================

# Size of each significance-testing window in base pairs.
sig_bin_size = 10000

# Overlap between adjacent windows.
sig_overlap = 5000

# Distance between the starts of consecutive windows.
sig_step = sig_bin_size - sig_overlap

# Analyze positions from the CT transition to +400 kb.
sig_range = (0, 400000)

# FDR significance threshold.
q_threshold = 0.05

# Stop after this many consecutive sufficiently powered
# non-significant windows.
default_max_consecutive_fail = 2

# Minimum number of observations required in each group.
min_n = 5


# Generate starting positions for overlapping windows.
sig_starts = np.arange(
    sig_range[0],
    sig_range[1] - sig_bin_size + sig_step,
    sig_step,
)


# =========================================================================
# Comparison-specific failure thresholds
# =========================================================================

# Individual comparisons can use different consecutive-failure thresholds.
# Currently, the spermONT vs bloodONT comparison uses the default value.
max_fail_override = {
    (
        "spermONT_methlev",
        "bloodONT_methlev",
    ): default_max_consecutive_fail
}


# =========================================================================
# Pairwise methylation comparisons
# =========================================================================

# IMPORTANT:
#
# The one-sided Welch's t-test below uses:
#
#     alternative="less"
#
# Therefore each comparison tests:
#
#     group1 methylation < group2 methylation

pairwise_comparisons = [
    (
        "sperm2_methlev",
        "bcell2_methlev",
    ),
    (
        "sperm2_methlev",
        "colon_methlev",
    ),
    (
        "sperm2_methlev",
        "hair_methlev",
    ),

    # Additional comparisons can be enabled if needed:
    #
    # (
    #     "brainOLIGO_methlev",
    #     "bcell2_methlev",
    # ),
    # (
    #     "brainOLIGO_methlev",
    #     "colon_methlev",
    # ),
    # (
    #     "brainOLIGO_methlev",
    #     "hair_methlev",
    # ),
    # (
    #     "testis_methlev",
    #     "bcell2_methlev",
    # ),
    # (
    #     "testis_methlev",
    #     "colon_methlev",
    # ),
    # (
    #     "testis_methlev",
    #     "hair_methlev",
    # ),

    (
        "spermONT_methlev",
        "bloodONT_methlev",
    ),
]


# =========================================================================
# Define chromosome-arm order
# =========================================================================

# Natural chromosome-arm ordering.
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
# Run pairwise significance analyses
# =========================================================================

all_sig_tables = []


for group1, group2 in pairwise_comparisons:

    print(
        f"Processing {group1} vs {group2}"
    )


    # Get comparison-specific failure threshold if defined.
    max_consecutive_fail = max_fail_override.get(
        (group1, group2),
        default_max_consecutive_fail,
    )


    # Store results for every genomic window in this comparison.
    all_window_results = []


    # =====================================================================
    # Calculate window-level P-values
    # =====================================================================

    for chrom in chromosomes:

        # Select current chromosome arm.
        chr_df = merged_df[
            merged_df["chr"] == chrom
        ].copy()


        # Restrict analysis to the selected distance range and rows
        # containing methylation values for both groups.
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
                group1,
                group2,
            ]
        ).copy()


        # Test each overlapping distance window.
        for start in sig_starts:

            end = start + sig_bin_size


            # Select observations inside the current window.
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


            vals1 = g[
                group1
            ].dropna()

            vals2 = g[
                group2
            ].dropna()


            # Require at least min_n observations in both groups.
            enough_n = (
                len(vals1) >= min_n
                and len(vals2) >= min_n
            )


            if enough_n:

                # One-sided Welch's independent two-sample t-test.
                #
                # Alternative hypothesis:
                #
                #     group1 < group2

                _, p = ttest_ind(
                    vals1,
                    vals2,
                    equal_var=False,
                    alternative="less",
                )

            else:

                p = np.nan


            # Store window-level results.
            all_window_results.append(
                {
                    "chr": chrom,
                    "group1": group1,
                    "group2": group2,
                    "comparison":
                        f"{group1}_vs_{group2}",

                    "bin_start":
                        start,

                    "bin_end":
                        end,

                    "bin_center":
                        (start + end) / 2,

                    "n_group1":
                        len(vals1),

                    "n_group2":
                        len(vals2),

                    "enough_n":
                        enough_n,

                    "pvalue":
                        p,
                }
            )


    # Convert all window-level results for this comparison to a DataFrame.
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


    # =====================================================================
    # Benjamini-Hochberg FDR correction
    # =====================================================================

    pvals_df[
        "qvalue"
    ] = np.nan


    # Only valid P-values are included in multiple-testing correction.
    valid = pvals_df[
        "pvalue"
    ].notna()


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


    # =====================================================================
    # Determine significance distance for each chromosome arm
    # =====================================================================

    sig_results = []


    for chrom in chromosomes:

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


        # Number of consecutive sufficiently powered
        # non-significant windows.
        fail_count = 0

        # Center of the most recent significant window.
        last_sig_center = None

        # First non-significant window in the terminating failure run.
        break_center = None

        first_fail_center = None


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


            # -------------------------------------------------------------
            # Skip underpowered windows
            # -------------------------------------------------------------
            #
            # Underpowered windows:
            #
            #     - are not considered significant
            #     - do not count as failures
            #     - do not reset the failure counter
            #

            if not enough_n:
                continue


            # Determine whether the current window is significant.
            is_sig = (
                pd.notna(q)
                and q < q_threshold
            )


            if is_sig:

                # Update the last significant position.
                last_sig_center = center

                # Reset consecutive-failure tracking.
                fail_count = 0
                first_fail_center = None


            else:

                # Current sufficiently powered window is non-significant.
                fail_count += 1


                # Record the first failure in the current run.
                if fail_count == 1:
                    first_fail_center = center


                # Stop once the required number of consecutive
                # non-significant windows is reached.
                if (
                    fail_count
                    >= max_consecutive_fail
                ):

                    break_center = (
                        first_fail_center
                    )

                    break


        # =================================================================
        # Calculate estimated significance boundary
        # =================================================================

        if last_sig_center is not None:

            if break_center is not None:

                # Midpoint between the center of the last significant
                # window and the first window in the terminating
                # non-significant run.
                sig_midpoint = (
                    last_sig_center
                    + break_center
                ) / 2

            else:

                # Significance persists through all tested windows.
                sig_midpoint = (
                    last_sig_center
                )


        else:

            # No significant sufficiently powered window was observed.
            sig_midpoint = np.nan


        # Store chromosome-level significance distance.
        sig_results.append(
            {
                "chr":
                    chrom,

                "group1":
                    group1,

                "group2":
                    group2,

                "comparison":
                    f"{group1}_vs_{group2}",

                "sig_midpoint_bp":
                    (
                        sig_midpoint
                        if pd.notna(sig_midpoint)
                        else 0
                    ),

                "sig_midpoint_kb":
                    (
                        sig_midpoint / 1000
                        if pd.notna(sig_midpoint)
                        else 0
                    ),
            }
        )


    # Store results for this pairwise comparison.
    sig_table = pd.DataFrame(
        sig_results
    )

    all_sig_tables.append(
        sig_table
    )


# =========================================================================
# Combine all pairwise comparisons
# =========================================================================

sig_table_all = pd.concat(
    all_sig_tables,
    ignore_index=True,
)


# =========================================================================
# Create base-pair output table
# =========================================================================

sig_table_bp = (
    sig_table_all
    .pivot(
        index="chr",
        columns="comparison",
        values="sig_midpoint_bp",
    )
    .reindex(
        chromosomes
    )
    .reset_index()
)


# =========================================================================
# Create kilobase output table
# =========================================================================

sig_table_kb = (
    sig_table_all
    .pivot(
        index="chr",
        columns="comparison",
        values="sig_midpoint_kb",
    )
    .reindex(
        chromosomes
    )
    .reset_index()
)


# Remove the pandas column-index name created by pivot().
sig_table_bp.columns.name = None
sig_table_kb.columns.name = None


# =========================================================================
# Save results
# =========================================================================

sig_table_bp.to_csv(
    "chromosome_sig_midpoints_all_pairwise_bp.tsv",
    sep="\t",
    index=False,
)


sig_table_kb.to_csv(
    "chromosome_sig_midpoints_all_pairwise_kb.tsv",
    sep="\t",
    index=False,
)


print(
    "Significance-distance tables saved."
)