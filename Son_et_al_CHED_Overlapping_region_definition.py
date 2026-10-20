#!/usr/bin/env python3

"""
Son_et_al_CHED_Overlapping_region_definition.py

Define CHED genomic regions using overlapping chromosome-specific
methylation significance boundaries.

This script uses two CHED definitions:

Approach 2
    A genomic window is classified as CHED only when it falls within
    the significance boundary for all four methylation comparisons:

        sperm2 vs bcell2
        sperm2 vs colon
        sperm2 vs hair
        spermONT vs bloodONT

Approach 3
    A genomic window is classified as CHED when it falls within the
    spermONT vs bloodONT significance boundary.

Adjacent CHED-positive windows are merged into continuous genomic regions.

---------------------------------------------------------------------------
INPUT 1
---------------------------------------------------------------------------

all_windows.tsv

Required columns:

    chr
        Chromosome-arm identifier, for example:
            chr1_p
            chr1_q

    stt
        Window start coordinate.

    end
        Window end coordinate.

    distance_from_closestCTtransition (closest Satellite_nonSatellite boundary)
        Distance in base pairs from the closest CT transition.

Example:

    chr      stt       end       distance_from_closestCTtransition
    chr1_p   100000    100499    5000
    chr1_p   100500    100999    5500


---------------------------------------------------------------------------
INPUT 2
---------------------------------------------------------------------------

chromosome_sig_midpoints_all_pairwise_bp_merged.tsv

Contains the chromosome-specific distances from centromere-transition
(Satellite-to-non-Satellite) boundaries over which DNA methylation levels
remain significantly different between pairs of samples/tissues.

Required columns:

    chr
        Chromosome-arm identifier.

    sperm2_bp_vs_bcell2_bp
        Significance boundary for sperm2 vs bcell2 methylation.

    sperm2_bp_vs_colon_bp
        Significance boundary for sperm2 vs colon methylation.

    sperm2_bp_vs_hair_bp
        Significance boundary for sperm2 vs hair methylation.

    spermONT_bp_vs_bloodONT_bp
        Significance boundary for spermONT vs bloodONT methylation.

All significance-boundary values are reported in base pairs from the
centromere-transition boundary.

---------------------------------------------------------------------------
APPROACH 2
---------------------------------------------------------------------------

A window is considered CHED-supported when:

    distance_from_closestCTtransition

falls between 0 and the significance boundary for ALL four comparisons.

Because four comparisons are tested:

    number of "yes" values >= 4

means all four comparisons must support the window.


---------------------------------------------------------------------------
APPROACH 3
---------------------------------------------------------------------------

A window is considered CHED-supported when:

    0 <= distance_from_closestCTtransition
        <= spermONT_bp_vs_bloodONT_bp


---------------------------------------------------------------------------
MERGING WINDOWS
---------------------------------------------------------------------------

Before merging, chromosome-arm suffixes are removed:

    chr1_p -> chr1
    chr1_q -> chr1

CHED-positive windows are merged when they overlap or are directly adjacent.

The current setting:

    max_gap = 0

allows no additional gap between windows.


---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Approach 2:

    TableS9_CHED_supportedall.tsv

Approach 3:

    TableS9_CHED_supportedONT.tsv

Output columns:

    chr
    stt
    end


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Required Python packages:

    pandas
    numpy


"""

import numpy as np
import pandas as pd


# =========================================================================
# Load input data
# =========================================================================

merged_df = pd.read_csv(
    "all_windows.tsv",
    sep="\t",
)

sig = pd.read_csv(
    "chromosome_sig_midpoints_all_pairwise_bp_merged.tsv",
    sep="\t",
)


# =========================================================================
# Standardize significance-boundary column names
# =========================================================================

# Example:
#
# sperm2_methlev_vs_bcell2_methlev
#
# becomes:
#
# sperm2_bp_vs_bcell2_bp
#
sig.columns = sig.columns.str.replace(
    "_methlev",
    "_bp",
    regex=False,
)


# =========================================================================
# Merge significance boundaries with genomic windows
# =========================================================================

sat_filt = merged_df.merge(
    sig[
        [
            "chr",
            "sperm2_bp_vs_bcell2_bp",
            "sperm2_bp_vs_colon_bp",
            "sperm2_bp_vs_hair_bp",
            "spermONT_bp_vs_bloodONT_bp",
        ]
    ],
    on="chr",
    how="left",
)


# =========================================================================
# Determine whether each window falls within each significance boundary
# =========================================================================

# sperm2 vs bcell2
sat_filt["if_sperm2_bp_vs_bcell2_bp"] = np.where(
    sat_filt["distance_from_closestCTtransition"].between(
        0,
        sat_filt["sperm2_bp_vs_bcell2_bp"],
    ),
    "yes",
    "no",
)


# sperm2 vs colon
sat_filt["if_sperm2_bp_vs_colon_bp"] = np.where(
    sat_filt["distance_from_closestCTtransition"].between(
        0,
        sat_filt["sperm2_bp_vs_colon_bp"],
    ),
    "yes",
    "no",
)


# sperm2 vs hair
sat_filt["if_sperm2_bp_vs_hair_bp"] = np.where(
    sat_filt["distance_from_closestCTtransition"].between(
        0,
        sat_filt["sperm2_bp_vs_hair_bp"],
    ),
    "yes",
    "no",
)


# spermONT vs bloodONT
sat_filt["if_spermONT_bp_vs_bloodONT_bp"] = np.where(
    sat_filt["distance_from_closestCTtransition"].between(
        0,
        sat_filt["spermONT_bp_vs_bloodONT_bp"],
    ),
    "yes",
    "no",
)


# =========================================================================
# Approach 2: supported by all four methylation comparisons
# =========================================================================

approach2_cols = [
    "if_sperm2_bp_vs_bcell2_bp",
    "if_sperm2_bp_vs_colon_bp",
    "if_sperm2_bp_vs_hair_bp",
    "if_spermONT_bp_vs_bloodONT_bp",
]


# Require all four comparisons to support the window.
sat_filt["if_approach2_CHED"] = np.where(
    sat_filt[approach2_cols].eq("yes").sum(axis=1) >= 4,
    "yes",
    "no",
)


# =========================================================================
# Approach 3: spermONT vs bloodONT support
# =========================================================================

sat_filt["if_approach3_CHED"] = (
    sat_filt["if_spermONT_bp_vs_bloodONT_bp"]
)


# =========================================================================
# Function to merge adjacent CHED-positive windows
# =========================================================================

def merge_ched_windows(data, flag_column, output_file, max_gap=0):
    """
    Merge adjacent or overlapping CHED-positive genomic windows.

    Parameters
    ----------
    data : pandas.DataFrame
        Input genomic-window table.

    flag_column : str
        Column containing "yes"/"no" CHED classifications.

    output_file : str
        Output TSV filename.

    max_gap : int
        Maximum allowed gap between consecutive windows.
        Default = 0.
    """

    ched = data.loc[
        data[flag_column].eq("yes"),
        [
            "chr",
            "stt",
            "end",
        ],
    ].copy()


    # Remove chromosome-arm suffix:
    #
    # chr1_p -> chr1
    # chr1_q -> chr1
    #
    ched["chr"] = ched["chr"].str.replace(
        r"_(p|q)$",
        "",
        regex=True,
    )


    # Sort windows before merging.
    ched = (
        ched.sort_values(
            [
                "chr",
                "stt",
                "end",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # Start a new block when the chromosome changes or when
    # consecutive windows are separated by more than max_gap.
    new_block = (
        ched["chr"].ne(
            ched["chr"].shift()
        )
        |
        ched["stt"].gt(
            ched["end"].shift().fillna(-1)
            + max_gap
            + 1
        )
    )


    group = new_block.cumsum()


    # Merge consecutive CHED-positive windows.
    merged = (
        ched.groupby(
            group,
            as_index=False,
        )
        .agg(
            chr=("chr", "first"),
            stt=("stt", "min"),
            end=("end", "max"),
        )
    )


    # Save merged genomic regions.
    merged.to_csv(
        output_file,
        sep="\t",
        index=False,
    )


# =========================================================================
# Save Approach 2 CHED regions
# =========================================================================

merge_ched_windows(
    data=sat_filt,
    flag_column="if_approach2_CHED",
    output_file="TableS9_CHED_supportedall.tsv",
    max_gap=0,
)


# =========================================================================
# Save Approach 3 CHED regions
# =========================================================================

merge_ched_windows(
    data=sat_filt,
    flag_column="if_approach3_CHED",
    output_file="TableS9_CHED_supportedONT.tsv",
    max_gap=0,
)


print("CHED region tables saved:")
print("  TableS9_CHED_supportedall.tsv")
print("  TableS9_CHED_supportedONT.tsv")