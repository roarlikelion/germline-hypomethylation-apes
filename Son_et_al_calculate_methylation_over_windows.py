#!/usr/bin/env python3

"""
Son_et_al_calculate_methylation_over_windows.py

Calculate mean CpG methylation levels and CpG counts for genomic windows.

Usage
-----
python Son_et_al_calculate_methylation_over_windows.py input_meth.tsv input_windows.tsv output.tsv

Input 1: input_meth.tsv
-----------------------
Tab-delimited methylation file with no header:

    chr    stt    end    cov    lev

Example:

    chr1    150    151    20    0.75

Input 2: input_windows.tsv
--------------------------
Tab-delimited window file containing five columns:

    chr    stt    end    window_number    CpG_OE

Example:

    chr1    1    500    1    0.62

Output: output.tsv
------------------
Tab-delimited file with no header. The original window columns are retained,
with two additional columns appended:

    chr    stt    end    window_number    CpG_OE
    mean_methylation_level    counts_cpg

For each genomic window, CpG records are selected when they fall completely
within the window:

    CpG stt >= window stt
    CpG end <= window end

The script calculates:
    - mean_methylation_level: mean of the "lev" values within the window
    - counts_cpg: number of CpG records within the window

Chromosomes/contigs are processed in parallel.
"""

import pandas as pd
import sys
from multiprocessing import Pool


# Command-line arguments
meth_file = sys.argv[1]
window_file = sys.argv[2]
output_file = sys.argv[3]


# Read methylation data.
# The methylation input file does not contain a header.
meth = pd.read_csv(
    meth_file,
    sep="\t",
    names=["chr", "stt", "end", "cov", "lev"]
)


# Read genomic windows.
windows = pd.read_csv(window_file, sep="\t")

# Standardize window column names.
windows.columns = ["chr", "stt", "end", "window_number", "CpG_OE"]


# Identify chromosomes/contigs to process independently.
chr_values = windows["chr"].unique()


def process_chromosome(chr_value):
    """
    Calculate mean methylation level and CpG count for all windows
    on a single chromosome/contig.
    """

    # Methylation records for the current chromosome.
    meth_subset = meth[meth["chr"] == chr_value].copy()

    # Windows for the current chromosome.
    window_subset = windows[windows["chr"] == chr_value].copy()

    mean_methylation_levels = []
    counts_cpg = []

    for index, window_row in window_subset.iterrows():

        window_stt = window_row["stt"]
        window_end = window_row["end"]

        # Select CpGs that fall completely within the genomic window.
        cpgs_in_window = meth_subset[
            (meth_subset["stt"] >= window_stt)
            & (meth_subset["end"] <= window_end)
        ]

        # Mean methylation level of CpGs within the window.
        mean_methylation = cpgs_in_window["lev"].mean()

        # Number of CpGs within the window.
        cpgs = len(cpgs_in_window)

        mean_methylation_levels.append(mean_methylation)
        counts_cpg.append(cpgs)

    # Append calculated values to the window table.
    window_subset.loc[:, "mean_methylation_level"] = mean_methylation_levels
    window_subset.loc[:, "counts_cpg"] = counts_cpg

    return window_subset


# Process chromosomes/contigs in parallel.
with Pool(processes=len(chr_values)) as pool:
    results = pool.map(process_chromosome, chr_values)


# Combine chromosome-level results into one table.
final_result = pd.concat(results, ignore_index=True, axis=0)


# Write output as a tab-delimited file without a header.
final_result.to_csv(
    output_file,
    sep="\t",
    index=False,
    header=None
)


print(f"Mean methylation levels and CpG counts saved to {output_file}")