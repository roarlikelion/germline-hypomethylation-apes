#!/usr/bin/env python3

"""
Son_et_al_CHED_gene_tissue_expression_rank_analysis.py

Compare tissue-expression rankings between genes associated with CHED
regions and genes not associated with CHED regions.

The analysis:

    1. Identifies genes expressed in >9 tissues at expression >0.1.
    2. Ranks each gene's expression across tissues.
    3. Separates genes into CHED and non-CHED groups.
    4. Compares rank distributions using the Kolmogorov-Smirnov test.
    5. Performs permutation tests on mean tissue ranks.
    6. Applies Benjamini-Hochberg FDR correction across tissues.
    7. Generates:
         - ECDF plots for selected tissues
         - Tissue-wide rank-difference bar plot
         - TSV file containing tissue-wide statistical results

---------------------------------------------------------------------------
INPUT
---------------------------------------------------------------------------

This script expects a pandas DataFrame named:

    df

Each row should represent one gene.

Required columns:

1. Tissue-expression columns
----------------------------

For example, GTEx v10 bulk-tissue RNA-seq median TPM values
(GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm).

Required tissue columns are listed in `tissue_columns` below.


2. CHED association
-------------------

if_affectedbyCHED

Expected values:

    yes
        Gene is associated with/affected by CHED.

    no
        Gene is not associated with CHED.


Example input structure:

    gene    Testis    Liver    Whole_Blood    ...    if_affectedbyCHED
    GENE1   5.2       0.3      0.8            ...    yes
    GENE2   1.1       2.4      1.5            ...    no


The DataFrame can be loaded before this analysis, for example:

    df = pd.read_csv("gene_expression.tsv", sep="\\t")


---------------------------------------------------------------------------
EXPRESSION FILTER
---------------------------------------------------------------------------

A gene is classified as `generally_expressed` when:

    expression TPM > 0.1 in more than 9 tissues

Equivalent code:

    df[tissue_columns].gt(0.1).sum(axis=1) > 9

Only these genes are retained for downstream analysis.


---------------------------------------------------------------------------
TISSUE RANKING
---------------------------------------------------------------------------

For each gene, expression values are ranked across tissues using:

    ascending=False
    method="max"

Therefore:

    rank 1 = highest-expression tissue for that gene

Lower ranks indicate relatively higher expression within a gene.


---------------------------------------------------------------------------
STATISTICAL TESTS
---------------------------------------------------------------------------

Kolmogorov-Smirnov test:
    Compares the complete rank distributions between CHED and
    non-CHED genes.

Permutation test:
    Compares mean tissue ranks between CHED and non-CHED genes.

Statistic:

    mean(CHED rank) - mean(non-CHED rank)

Interpretation:

    negative value
        CHED genes have lower ranks, corresponding to relatively higher
        expression in that tissue.

    positive value
        CHED genes have higher ranks, corresponding to relatively lower
        expression in that tissue.

Permutation tests use:

    10,000 permutations

Multiple-testing correction:

    Benjamini-Hochberg FDR across tissues


---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------

Figures:

    ECDF_selectedtissues_100kb.pdf

    fig5_tissues_enrichements_100kb.pdf

Statistical results:

    Son_et_al_CHED_gene_tissue_expression_rank_analysis.tsv


---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------

Required Python packages:

    pandas
    numpy
    matplotlib
    seaborn
    scipy
    statsmodels

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from scipy.stats import ks_2samp
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


# -------------------------------------------------------------------------
# Tissue-expression columns
# -------------------------------------------------------------------------

tissue_columns = [
    "Adipose_Subcutaneous",
    "Adipose_Visceral_Omentum",
    "Adrenal_Gland",
    "Artery_Aorta",
    "Artery_Coronary",
    "Artery_Tibial",
    "Bladder",
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum",
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue",
    "Cells_Cultured_fibroblasts",
    "Cells_EBV-transformed_lymphocytes",
    "Cervix_Ectocervix",
    "Cervix_Endocervix",
    "Colon_Sigmoid",
    "Colon_Transverse",
    "Colon_Transverse_Mixed_Cell",
    "Colon_Transverse_Mucosa",
    "Colon_Transverse_Muscularis",
    "Esophagus_Gastroesophageal_Junction",
    "Esophagus_Mucosa",
    "Esophagus_Muscularis",
    "Fallopian_Tube",
    "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle",
    "Kidney_Cortex",
    "Kidney_Medulla",
    "Liver",
    "Liver_Hepatocyte",
    "Liver_Mixed_Cell",
    "Liver_Portal_Tract",
    "Lung",
    "Minor_Salivary_Gland",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Ovary",
    "Pancreas",
    "Pancreas_Acini",
    "Pancreas_Islets",
    "Pancreas_Mixed_Cell",
    "Pituitary",
    "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum",
    "Small_Intestine_Terminal_Ileum_Lymphode_Aggregate",
    "Small_Intestine_Terminal_Ileum_Mixed_Cell",
    "Spleen",
    "Stomach",
    "Stomach_Mixed_Cell",
    "Stomach_Mucosa",
    "Stomach_Muscularis",
    "Testis",
    "Thyroid",
    "Uterus",
    "Vagina",
    "Whole_Blood",
]


# =========================================================================
# Identify generally expressed genes
# =========================================================================

# A gene is considered generally expressed when its expression is >0.1
# in more than 9 tissues.
df["generally_expressed"] = (
    df[tissue_columns]
    .gt(0.1)
    .sum(axis=1)
    > 9
)


# =========================================================================
# Calculate within-gene tissue-expression ranks
# =========================================================================

# Rank tissue expression separately for each gene.
#
# ascending=False:
#     highest expression receives the lowest rank.
#
# Therefore:
#     rank 1 = highest-expression tissue for that gene.

for tissue in tissue_columns:
    df[f"{tissue}_ranking"] = (
        df[tissue_columns]
        .rank(
            axis=1,
            method="max",
            ascending=False,
        )[tissue]
    )


# Retain genes passing the generally-expressed filter.
df_nonzero = df[
    df["generally_expressed"] == True
].copy()


# =========================================================================
# Define CHED and non-CHED gene groups
# =========================================================================

group1 = df_nonzero[
    df_nonzero["if_affectedbyCHED"] == "yes"
].copy()

group2 = df_nonzero[
    df_nonzero["if_affectedbyCHED"] == "no"
].copy()


group1["group"] = "CHED Genes"
group2["group"] = "nonCHED Genes"


if group1.empty or group2.empty:
    raise ValueError(
        "One of the groups is empty. "
        "Check values in 'if_affectedbyCHED'."
    )


# =========================================================================
# Selected-tissue ECDF analysis
# =========================================================================

ranking_cols = [
    "Testis_ranking",
    "Brain_Hypothalamus_ranking",
    "Brain_Hippocampus_ranking",
    "Whole_Blood_ranking",
    "Liver_Mixed_Cell_ranking",
    "Stomach_Muscularis_ranking",
]


# Combine CHED and non-CHED genes.
df_plot = pd.concat(
    [group1, group2],
    ignore_index=True,
)


# Retain only group and selected ranking columns.
cols_needed = ["group"] + ranking_cols

df_plot = df_plot.loc[
    :,
    cols_needed,
]


# Convert wide ranking table into long format.
df_long = df_plot.melt(
    id_vars="group",
    value_vars=ranking_cols,
    var_name="tissue",
    value_name="ranking",
).dropna(
    subset=["ranking"]
)


# Clean tissue names for plotting.
df_long["tissue"] = (
    df_long["tissue"]
    .str.replace(
        "_ranking",
        "",
        regex=False,
    )
    .str.replace(
        "_",
        " ",
    )
)


# -------------------------------------------------------------------------
# Convert P-values to significance symbols
# -------------------------------------------------------------------------

def p_to_stars(p):
    """Convert a P-value to a significance label."""

    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# -------------------------------------------------------------------------
# KS test for selected tissues
# -------------------------------------------------------------------------

ks_results = []

for tissue in df_long["tissue"].unique():

    CHED_vals = df_long.loc[
        (df_long["tissue"] == tissue)
        & (df_long["group"] == "CHED Genes"),
        "ranking",
    ].dropna()

    nonCHED_vals = df_long.loc[
        (df_long["tissue"] == tissue)
        & (df_long["group"] == "nonCHED Genes"),
        "ranking",
    ].dropna()


    # Two-sample Kolmogorov-Smirnov test.
    ks_stat, ks_p = ks_2samp(
        CHED_vals,
        nonCHED_vals,
    )


    # Compare median ranks to describe direction.
    CHED_median = CHED_vals.median()
    nonCHED_median = nonCHED_vals.median()


    if CHED_median < nonCHED_median:
        direction = "CHED < nonCHED"

    elif CHED_median > nonCHED_median:
        direction = "CHED > nonCHED"

    else:
        direction = "CHED = nonCHED"


    ks_results.append(
        {
            "tissue": tissue,
            "CHED_median": CHED_median,
            "nonCHED_median": nonCHED_median,
            "direction": direction,
            "KS_statistic": ks_stat,
            "KS_p_value": ks_p,
            "KS_significance": p_to_stars(ks_p),
            "n_CHED": len(CHED_vals),
            "n_nonCHED": len(nonCHED_vals),
        }
    )


ks_results_df = pd.DataFrame(
    ks_results
)


print(
    "\nKolmogorov-Smirnov test results:"
)

print(
    ks_results_df.to_string(
        index=False
    )
)


# Information displayed on individual ECDF panels.
plot_info = {
    row["tissue"]: {
        "direction": row["direction"],
        "ks_p": row["KS_p_value"],
        "ks_sig": row["KS_significance"],
    }
    for _, row in ks_results_df.iterrows()
}


# -------------------------------------------------------------------------
# ECDF plots for selected tissues
# -------------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="talk",
)


g = sns.FacetGrid(
    df_long,
    col="tissue",
    col_wrap=6,
    hue="group",
    palette={
        "CHED Genes": "#00A79D",
        "nonCHED Genes": "black",
    },
    sharex=False,
    sharey=True,
    height=5,
    aspect=0.8,
)


g.map_dataframe(
    sns.ecdfplot,
    x="ranking",
    linewidth=2,
)


g.set_axis_labels(
    "Ranking",
    "ECDF",
)

g.set_titles(
    "{col_name}",
    size=14,
)


# Add KS test information to each tissue panel.
for ax in g.axes.flat:

    tissue = ax.get_title()

    if tissue in plot_info:

        info = plot_info[tissue]

        ax.text(
            0.05,
            0.95,
            (
                f"{info['direction']}\n"
                f"KS p = {info['ks_p']:.2e} "
                f"{info['ks_sig']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )


g.add_legend(
    title="Group"
)

g._legend.set_bbox_to_anchor(
    (1.0, 0.0)
)

g._legend.set_loc(
    "lower right"
)


plt.tight_layout()


plt.savefig(
    "ECDF_selectedtissues.pdf",
    bbox_inches="tight",
    format="pdf",
    transparent=True,
)


plt.show()


# =========================================================================
# Tissue-wide statistical analysis
# =========================================================================

rank_cols = (
    df_nonzero.columns[
        df_nonzero.columns.str.endswith(
            "_ranking"
        )
    ]
    .tolist()
)


if len(rank_cols) == 0:
    raise ValueError(
        "No columns ending with '_ranking' were found."
    )


# -------------------------------------------------------------------------
# Permutation test for difference in mean ranks
# -------------------------------------------------------------------------

def permutation_test_mean(
    a,
    b,
    n_perm=10000,
    rng=None,
):
    """
    Perform a two-sided permutation test for the difference in mean ranks.

    Statistic:

        mean(CHED ranks) - mean(non-CHED ranks)

    Parameters
    ----------
    a : array-like
        CHED gene ranks.

    b : array-like
        non-CHED gene ranks.

    n_perm : int
        Number of permutations.

    rng : numpy random generator
        Random-number generator.

    Returns
    -------
    observed_difference : float
        Difference between group means.

    p_value : float
        Two-sided permutation P-value.
    """

    if rng is None:
        rng = np.random.default_rng(
            123
        )


    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )


    # Remove non-finite values.
    a = a[
        np.isfinite(a)
    ]

    b = b[
        np.isfinite(b)
    ]


    n_a = a.size
    n_b = b.size


    if n_a == 0 or n_b == 0:
        return np.nan, np.nan


    # Observed difference in mean tissue rank.
    obs = (
        np.mean(a)
        - np.mean(b)
    )


    combined = np.concatenate(
        [a, b]
    )

    perm_stats = np.empty(
        n_perm
    )


    for i in range(n_perm):

        permuted = rng.permutation(
            combined
        )

        perm_a = permuted[:n_a]
        perm_b = permuted[n_a:]

        perm_stats[i] = (
            np.mean(perm_a)
            - np.mean(perm_b)
        )


    # Two-sided permutation P-value.
    pval = (
        np.sum(
            np.abs(perm_stats)
            >= np.abs(obs)
        )
        + 1
    ) / (
        n_perm + 1
    )


    return obs, pval


# -------------------------------------------------------------------------
# Calculate summary statistics for every tissue
# -------------------------------------------------------------------------

summary_rows = []

# Fixed seed for reproducibility.
rng = np.random.default_rng(
    123
)


for c in rank_cols:

    tissue = c.replace(
        "_ranking",
        "",
    )


    CHED_vals = group1[
        c
    ].to_numpy(
        dtype=float
    )

    nonCHED_vals = group2[
        c
    ].to_numpy(
        dtype=float
    )


    CHED_vals = CHED_vals[
        np.isfinite(
            CHED_vals
        )
    ]

    nonCHED_vals = nonCHED_vals[
        np.isfinite(
            nonCHED_vals
        )
    ]


    if (
        CHED_vals.size > 0
        and nonCHED_vals.size > 0
    ):

        # Mean tissue ranks.
        CHED_mean = np.mean(
            CHED_vals
        )

        nonCHED_mean = np.mean(
            nonCHED_vals
        )

        delta_mean_rank = (
            CHED_mean
            - nonCHED_mean
        )


        # Median tissue ranks.
        CHED_median = np.median(
            CHED_vals
        )

        nonCHED_median = np.median(
            nonCHED_vals
        )

        delta_median_rank = (
            CHED_median
            - nonCHED_median
        )


        # Two-sample KS test.
        ks_res = ks_2samp(
            CHED_vals,
            nonCHED_vals,
            alternative="two-sided",
            method="auto",
        )

        ks_statistic = (
            ks_res.statistic
        )

        ks_pvalue = (
            ks_res.pvalue
        )


        # Permutation test for difference in mean ranks.
        (
            perm_delta_mean,
            perm_pvalue_mean,
        ) = permutation_test_mean(
            CHED_vals,
            nonCHED_vals,
            n_perm=10000,
            rng=rng,
        )


    else:

        CHED_mean = np.nan
        nonCHED_mean = np.nan
        delta_mean_rank = np.nan

        CHED_median = np.nan
        nonCHED_median = np.nan
        delta_median_rank = np.nan

        ks_statistic = np.nan
        ks_pvalue = np.nan

        perm_delta_mean = np.nan
        perm_pvalue_mean = np.nan


    summary_rows.append(
        {
            "tissue": tissue,

            "CHED_mean":
                CHED_mean,

            "nonCHED_mean":
                nonCHED_mean,

            "delta_mean_rank":
                delta_mean_rank,

            "CHED_median":
                CHED_median,

            "nonCHED_median":
                nonCHED_median,

            "delta_median_rank":
                delta_median_rank,

            "ks_statistic":
                ks_statistic,

            "ks_pvalue":
                ks_pvalue,

            "perm_delta_mean":
                perm_delta_mean,

            "perm_pvalue_mean":
                perm_pvalue_mean,

            "n_CHED":
                CHED_vals.size,

            "n_nonCHED":
                nonCHED_vals.size,
        }
    )


out = pd.DataFrame(
    summary_rows
)


# =========================================================================
# Benjamini-Hochberg multiple-testing correction
# =========================================================================

out["ks_padj_bh"] = np.nan
out["perm_padj_mean_bh"] = np.nan


mask_ks = out[
    "ks_pvalue"
].notna()

mask_perm = out[
    "perm_pvalue_mean"
].notna()


# BH-FDR correction for KS tests.
out.loc[
    mask_ks,
    "ks_padj_bh",
] = multipletests(
    out.loc[
        mask_ks,
        "ks_pvalue",
    ],
    method="fdr_bh",
)[1]


# BH-FDR correction for permutation tests.
out.loc[
    mask_perm,
    "perm_padj_mean_bh",
] = multipletests(
    out.loc[
        mask_perm,
        "perm_pvalue_mean",
    ],
    method="fdr_bh",
)[1]


# Sort tissues by difference in mean rank.
out = (
    out.sort_values(
        "delta_mean_rank",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)


# -------------------------------------------------------------------------
# Plotting statistic
# -------------------------------------------------------------------------

# NOTE:
# `signed_area_ecdf` is retained for compatibility with the original
# analysis, but it is not an ECDF area.
#
# It represents:
#
#     mean(CHED rank) - mean(non-CHED rank)

out["signed_area_ecdf"] = (
    out["delta_mean_rank"]
)


# =========================================================================
# Save tissue-wide statistical results
# =========================================================================

out.to_csv(
    "Son_et_al_CHED_gene_tissue_expression_rank_analysis.tsv",
    sep="\t",
    index=False,
)


# =========================================================================
# Tissue-wide rank-difference plot
# =========================================================================

plot_df = (
    out.sort_values(
        "signed_area_ecdf",
        ascending=True,
    )
    .copy()
)


# -------------------------------------------------------------------------
# Assign bar colors based on BH-adjusted permutation P-value
# -------------------------------------------------------------------------

plot_df["bar_color"] = (
    "lightgray"
)


plot_df.loc[
    plot_df["perm_padj_mean_bh"] < 0.05,
    "bar_color",
] = "#fdb863"


plot_df.loc[
    plot_df["perm_padj_mean_bh"] < 0.01,
    "bar_color",
] = "#e66101"


plot_df.loc[
    plot_df["perm_padj_mean_bh"] < 0.001,
    "bar_color",
] = "#b2182b"


plot_df.loc[
    plot_df["perm_padj_mean_bh"] < 0.0001,
    "bar_color",
] = "#67001f"


# -------------------------------------------------------------------------
# Draw horizontal bar plot
# -------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 15)
)


ypos = np.arange(
    len(plot_df)
)


ax.barh(
    ypos,
    plot_df["signed_area_ecdf"],
    color=plot_df["bar_color"],
    edgecolor="black",
    linewidth=0.3,
)


# Zero represents no difference in mean tissue rank.
ax.axvline(
    0,
    color="black",
    linewidth=1,
)


# -------------------------------------------------------------------------
# Axis labels and formatting
# -------------------------------------------------------------------------

ax.set_yticks(
    ypos
)

ax.set_yticklabels(
    plot_df["tissue"]
)

ax.set_ylabel(
    "Tissue"
)

ax.set_xlabel(
    "Difference in mean tissue rank\n"
    "(CHED − nonCHED Genes)"
)


ax.invert_yaxis()


# Place tissue labels on the right side.
ax.yaxis.set_ticks_position(
    "right"
)

ax.yaxis.set_label_position(
    "right"
)

ax.tick_params(
    axis="y",
    right=True,
    left=False,
    labelright=True,
    labelleft=False,
)


# Keep x-axis labels at the bottom.
ax.tick_params(
    axis="x",
    bottom=True,
    top=False,
    labelbottom=True,
    labeltop=False,
)


# -------------------------------------------------------------------------
# X-axis limits
# -------------------------------------------------------------------------

xmin = plot_df[
    "signed_area_ecdf"
].min()

xmax = plot_df[
    "signed_area_ecdf"
].max()


xrange_ = (
    xmax - xmin
    if xmax != xmin
    else 1
)


ax.set_xlim(
    xmin - 0.05 * xrange_,
    xmax + 0.05 * xrange_,
)


ax.grid(
    True,
    axis="both",
    linestyle="-",
    linewidth=1,
    alpha=0.7,
)


# -------------------------------------------------------------------------
# Highlight brain and testis tissue labels
# -------------------------------------------------------------------------

for lbl in ax.get_yticklabels():

    txt = lbl.get_text().lower()

    if "brain" in txt:
        lbl.set_color(
            "red"
        )

    elif "testis" in txt:
        lbl.set_color(
            "blue"
        )


# -------------------------------------------------------------------------
# Significance legend
# -------------------------------------------------------------------------

legend_handles = [
    Patch(
        facecolor="lightgray",
        edgecolor="black",
        label="padj ≥ 0.05",
    ),

    Patch(
        facecolor="#fdb863",
        edgecolor="black",
        label="padj < 0.05",
    ),

    Patch(
        facecolor="#e66101",
        edgecolor="black",
        label="padj < 0.01",
    ),

    Patch(
        facecolor="#b2182b",
        edgecolor="black",
        label="padj < 0.001",
    ),

    Patch(
        facecolor="#67001f",
        edgecolor="black",
        label="padj < 0.0001",
    ),
]


ax.legend(
    handles=legend_handles,
    title="Permutation BH-adjusted p-value",
    loc="lower left",
    frameon=False,
)


plt.tight_layout()


plt.savefig(
    "fig5_tissues_enrichements.pdf",
    bbox_inches="tight",
    format="pdf",
    transparent=True,
)


plt.show()


# Display final statistical results when running interactively.
out