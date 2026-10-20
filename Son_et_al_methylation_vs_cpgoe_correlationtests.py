#!/usr/bin/env python3

"""
Son_et_al_methylation_vs_cpgoe_correlationtests.py

Calculate correlations between CpG observed/expected (CpG O/E) ratios
and DNA methylation levels across chimpanzee tissues and genomic features.

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script expects a pandas DataFrame called:

    chimpanzee_repeat

The DataFrame should contain one row per genomic window.

Required columns include:

1. CpG O/E and sequence composition
-----------------------------------
CpG_OE
    CpG observed/expected ratio for the genomic window.

GC_content
    GC content of the genomic window.

GC_Skew
    GC skew of the genomic window.


2. Tissue methylation levels
----------------------------
lymphoblastoidONT_methlev
sperm_methlev
wholeblood_methlev
brainNEU_methlev
brainOLIGO_methlev

These columns should contain the mean methylation level for each window.


3. Tissue-specific CpG counts
-----------------------------
lymphoblastoidONT_cpgcounts
sperm_cpgcounts
wholeblood_cpgcounts
brainNEU_cpgcounts
brainOLIGO_cpgcounts

These columns contain the number of CpGs contributing to the methylation
estimate for each tissue.

Only windows with >2 CpGs are included for the corresponding tissue.


4. Genomic annotation columns
-----------------------------
feature_allrepeat_exclude
feature_gene
feature_intron
feature_repeat
feature_censat

These columns are used to define genomic feature groups such as:

    No repeats
    Any repeats
    protein-coding promoters
    protein-coding gene bodies
    satellites
    LINE/L1
    simple repeats
    SVA
    human satellites
    higher-order repeats
    monomeric repeats
    subterminal satellites


Example input structure
-----------------------

chr     stt   end   CpG_OE   GC_content   GC_Skew   sperm_methlev   sperm_cpgcounts
chr1    1     500   0.62     0.43         0.05      0.78            12
chr1    501   1000  0.71     0.47        -0.02      0.65            8


The DataFrame can be loaded before running this analysis, for example:

    chimpanzee_repeat = pd.read_csv("chimpanzee_windows.tsv", sep="\\t")


---------------------------------------------------------------------------
ANALYSIS
---------------------------------------------------------------------------

For each tissue, correlations are calculated between:

    CpG_OE and methylation level

The following tests are performed:

    - Spearman correlation
    - Pearson correlation
    - Kendall's tau
    - Partial correlation controlling for GC content
    - Partial correlation controlling for GC skew
    - Partial correlation controlling for both GC content and GC skew

Rows with missing CpG_OE, GC_content, GC_Skew, or methylation values
are excluded.

Feature-specific tests are performed only when >20 windows are available.


---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Output file:

    0_chimpanzee_11tissues_partial_correlations_filtered_CpG_OE_by_tissue.csv

The output includes:

    Tissue
    Feature
    total_window_counts
    Percentage of Total
    Spearman Correlation (P-Value)
    Pearson Correlation (P-Value)
    Kendall's Tau (P-Value)
    Partial Correlation (GC%) (P-Value)
    Partial Correlation (GC Skew) (P-Value)
    Partial Correlation (GC% & GC Skew) (P-Value)


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Install required Python packages with:

    pip install pandas scipy pingouin
"""

import pandas as pd
from pingouin import partial_corr
from scipy.stats import spearmanr, pearsonr, kendalltau


# -------------------------------------------------------------------------
# Load input data
# -------------------------------------------------------------------------

# Uncomment and modify this line if running the script independently:
#
# chimpanzee_repeat = pd.read_csv(
#     "chimpanzee_windows.tsv",
#     sep="\t"
# )


# -------------------------------------------------------------------------
# Define tissue methylation columns and corresponding CpG-count columns
# -------------------------------------------------------------------------

tissue_columns = [
    "lymphoblastoidONT_methlev",
    "sperm_methlev",
    "wholeblood_methlev",
    "brainNEU_methlev",
    "brainOLIGO_methlev",
]

cpg_count_columns = [
    "lymphoblastoidONT_cpgcounts",
    "sperm_cpgcounts",
    "wholeblood_cpgcounts",
    "brainNEU_cpgcounts",
    "brainOLIGO_cpgcounts",
]


# Store all correlation results.
results = []


# -------------------------------------------------------------------------
# Define genomic feature categories
# -------------------------------------------------------------------------

# Windows without or with annotated repeat elements.
chimpanzee_repeat.loc[:, "No repeats"] = (
    chimpanzee_repeat["feature_allrepeat_exclude"].isna()
)

chimpanzee_repeat.loc[:, "Any repeats"] = (
    chimpanzee_repeat["feature_allrepeat_exclude"].notna()
)


# Protein-coding gene annotations.
chimpanzee_repeat.loc[:, "protein_coding_promoter"] = (
    chimpanzee_repeat["feature_gene"]
    .str.contains("protein_coding_promoter", na=False)
)

chimpanzee_repeat.loc[:, "protein_coding_genebody"] = (
    chimpanzee_repeat["feature_gene"]
    .str.contains("protein_coding_genebody", na=False)
)


# Intron and exon annotations.
# These features are defined here but are not currently included
# in feature_columns below.

chimpanzee_repeat.loc[:, "intron"] = (
    chimpanzee_repeat["feature_intron"]
    .str.contains("intron", na=False)
)

chimpanzee_repeat.loc[:, "exon UTR"] = (
    chimpanzee_repeat["feature_intron"]
    .str.contains("exon_UTR", na=False)
)

chimpanzee_repeat.loc[:, "exon CDS"] = (
    chimpanzee_repeat["feature_intron"]
    .str.contains("exon_CDS", na=False)
)


# RepeatMasker-based repeat classes.
chimpanzee_repeat.loc[:, "satellite"] = (
    chimpanzee_repeat["feature_repeat"]
    .str.contains("Satellite", na=False, case=False)
)

chimpanzee_repeat.loc[:, "LINE/L1"] = (
    chimpanzee_repeat["feature_repeat"]
    .str.contains("LINE/L1", na=False)
)

chimpanzee_repeat.loc[:, "Simple repeat"] = (
    chimpanzee_repeat["feature_repeat"]
    .str.contains("Simple_repeat", na=False)
)

chimpanzee_repeat.loc[:, "Retroposon/SVA"] = (
    chimpanzee_repeat["feature_repeat"]
    .str.contains("Retroposon/SVA", na=False)
)


# Centromeric and satellite sequence annotations.
chimpanzee_repeat.loc[:, "HSat"] = (
    chimpanzee_repeat["feature_censat"]
    .str.contains(r"hsat1|hsat2|hsat3", case=False, na=False)
)

chimpanzee_repeat.loc[:, "hor"] = (
    chimpanzee_repeat["feature_censat"]
    .str.contains("hor", case=False, na=False)
)

chimpanzee_repeat.loc[:, "mon"] = (
    chimpanzee_repeat["feature_censat"]
    .str.contains("mon", case=False, na=False)
)

chimpanzee_repeat.loc[:, "Subterminal SAT"] = (
    chimpanzee_repeat["feature_censat"]
    .str.contains("subTerm", na=False)
)


# Genomic feature groups included in the correlation analysis.
feature_columns = [
    "No repeats",
    "Any repeats",
    "protein_coding_promoter",
    "protein_coding_genebody",

    # Optional features:
    # "intron",
    # "exon UTR",
    # "exon CDS",
    # "SINE/Alu",

    "satellite",
    "LINE/L1",
    "Simple repeat",
    "Retroposon/SVA",
    "HSat",
    "hor",
    "mon",
    "Subterminal SAT",
]


# -------------------------------------------------------------------------
# Run correlation analyses for each tissue
# -------------------------------------------------------------------------

for tissue, cpg_count in zip(tissue_columns, cpg_count_columns):

    # Keep windows supported by at least 3 CpGs for the current tissue.
    #
    # Rows missing variables required for correlation analysis are removed.
    filtered_data = chimpanzee_repeat[
        chimpanzee_repeat[cpg_count] > 2
    ].dropna(
        subset=[
            "CpG_OE",
            "GC_content",
            "GC_Skew",
            tissue,
        ]
    ).copy()

    total_rows = len(filtered_data)

    # Skip tissue if no windows remain after filtering.
    if total_rows == 0:
        continue


    # ---------------------------------------------------------------------
    # Overall correlation analysis
    # ---------------------------------------------------------------------

    # Partial correlation controlling for GC content.
    partial_corr_GCper = partial_corr(
        data=filtered_data,
        x="CpG_OE",
        y=tissue,
        covar="GC_content",
    )

    # Partial correlation controlling for GC skew.
    partial_corr_GCskew = partial_corr(
        data=filtered_data,
        x="CpG_OE",
        y=tissue,
        covar="GC_Skew",
    )

    # Partial correlation controlling for both GC content and GC skew.
    partial_corr_GC_both = partial_corr(
        data=filtered_data,
        x="CpG_OE",
        y=tissue,
        covar=["GC_Skew", "GC_content"],
    )

    # Standard correlation tests.
    spearman_corr, spearman_pval = spearmanr(
        filtered_data["CpG_OE"],
        filtered_data[tissue],
    )

    pearson_corr, pearson_pval = pearsonr(
        filtered_data["CpG_OE"],
        filtered_data[tissue],
    )

    kendall_corr, kendall_pval = kendalltau(
        filtered_data["CpG_OE"],
        filtered_data[tissue],
    )


    # Store overall tissue-level results.
    results.append(
        {
            "Tissue": tissue,
            "Feature": "Overall",
            "total_window_counts": total_rows,
            "Percentage of Total": "100%",

            "Spearman Correlation (P-Value)":
                f"{round(spearman_corr, 3)} "
                f"({round(spearman_pval, 3)})",

            "Pearson Correlation (P-Value)":
                f"{round(pearson_corr, 3)} "
                f"({round(pearson_pval, 3)})",

            "Kendall's Tau (P-Value)":
                f"{round(kendall_corr, 3)} "
                f"({round(kendall_pval, 3)})",

            "Partial Correlation (GC%) (P-Value)":
                f"{round(partial_corr_GCper['r'][0], 3)} "
                f"({round(partial_corr_GCper['p-val'][0], 3)})",

            "Partial Correlation (GC Skew) (P-Value)":
                f"{round(partial_corr_GCskew['r'][0], 3)} "
                f"({round(partial_corr_GCskew['p-val'][0], 3)})",

            "Partial Correlation (GC% & GC Skew) (P-Value)":
                f"{round(partial_corr_GC_both['r'][0], 3)} "
                f"({round(partial_corr_GC_both['p-val'][0], 3)})",
        }
    )


    # ---------------------------------------------------------------------
    # Feature-specific correlation analyses
    # ---------------------------------------------------------------------

    for feature in feature_columns:

        # Select windows belonging to the current genomic feature.
        feature_data = filtered_data[filtered_data[feature]]

        feature_count = len(feature_data)

        # Fraction of filtered tissue windows belonging to this feature.
        percentage_of_total = (
            feature_count / total_rows
        ) * 100


        # Require more than 20 windows for feature-specific statistics.
        if feature_count > 20:

            partial_corr_GCper = partial_corr(
                data=feature_data,
                x="CpG_OE",
                y=tissue,
                covar="GC_content",
            )

            partial_corr_GCskew = partial_corr(
                data=feature_data,
                x="CpG_OE",
                y=tissue,
                covar="GC_Skew",
            )

            partial_corr_GC_both = partial_corr(
                data=feature_data,
                x="CpG_OE",
                y=tissue,
                covar=["GC_Skew", "GC_content"],
            )

            spearman_corr, spearman_pval = spearmanr(
                feature_data["CpG_OE"],
                feature_data[tissue],
            )

            pearson_corr, pearson_pval = pearsonr(
                feature_data["CpG_OE"],
                feature_data[tissue],
            )

            kendall_corr, kendall_pval = kendalltau(
                feature_data["CpG_OE"],
                feature_data[tissue],
            )


            # Format correlation coefficient and P-value as:
            # correlation (P-value)

            spearman_value = (
                f"{round(spearman_corr, 3)} "
                f"({round(spearman_pval, 3)})"
            )

            pearson_value = (
                f"{round(pearson_corr, 3)} "
                f"({round(pearson_pval, 3)})"
            )

            kendall_value = (
                f"{round(kendall_corr, 3)} "
                f"({round(kendall_pval, 3)})"
            )

            partial_value = (
                f"{round(partial_corr_GCper['r'][0], 3)} "
                f"({round(partial_corr_GCper['p-val'][0], 3)})"
            )

            partial_value2 = (
                f"{round(partial_corr_GCskew['r'][0], 3)} "
                f"({round(partial_corr_GCskew['p-val'][0], 3)})"
            )

            partial_value_both = (
                f"{round(partial_corr_GC_both['r'][0], 3)} "
                f"({round(partial_corr_GC_both['p-val'][0], 3)})"
            )

        else:

            # Do not calculate correlations when there are <=20 windows.
            spearman_value = "Insufficient data"
            pearson_value = "Insufficient data"
            kendall_value = "Insufficient data"
            partial_value = "Insufficient data"
            partial_value2 = "Insufficient data"
            partial_value_both = "Insufficient data"


        # Store feature-specific results.
        results.append(
            {
                "Tissue": tissue,
                "Feature": feature,
                "total_window_counts": feature_count,

                "Percentage of Total":
                    f"{round(percentage_of_total, 2)}%",

                "Spearman Correlation (P-Value)":
                    spearman_value,

                "Pearson Correlation (P-Value)":
                    pearson_value,

                "Kendall's Tau (P-Value)":
                    kendall_value,

                "Partial Correlation (GC%) (P-Value)":
                    partial_value,

                "Partial Correlation (GC Skew) (P-Value)":
                    partial_value2,

                "Partial Correlation (GC% & GC Skew) (P-Value)":
                    partial_value_both,
            }
        )


# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "0_chimpanzee_tissues_partial_correlations_filtered_CpG_OE_by_tissue.csv",
    index=False,
)


# Display results when running interactively, for example in Jupyter.
results_df