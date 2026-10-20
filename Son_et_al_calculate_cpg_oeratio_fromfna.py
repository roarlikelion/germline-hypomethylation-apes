#!/usr/bin/env python3

"""
Son_et_al_calculate_cpg_oeratio_fromfna.py

Calculate CpG observed/expected (O/E) ratios across non-overlapping
500-bp windows of a genome FASTA file.

Usage
-----
python Son_et_al_calculate_cpg_oeratio_fromfna.py genome.fasta windows.bed cpg_oe.tsv

Arguments
---------
genome.fasta
    Input genome in standard FASTA format. Each FASTA record is treated
    as a chromosome or contig.

windows.bed
    Output tab-delimited file containing genomic windows:
        chr    window_stt    window_end    window_number

    Coordinates are 1-based and inclusive.

cpg_oe.tsv
    Output tab-delimited file containing:
        chr    window_stt    window_end    window_number    CpG_OE

CpG O/E calculation
--------------------
For each window:

    CpG O/E = observed_CpG / expected_CpG

where:

    observed_CpG = number of "CG" dinucleotides
    expected_CpG = (#C * #G) / window_length

Special values
--------------
"n"
    Returned when the number of N bases is greater than 10% of the
    configured window size (500 bp).

"z"
    Returned when the expected CpG count is zero.

Notes
-----
- Sequences are converted to uppercase before counting.
- Windows are non-overlapping.
- The final window of a chromosome/contig may be shorter than 500 bp.
"""

from Bio import SeqIO
import pandas as pd
import sys


# Command-line arguments
genome_file = sys.argv[1]
output_bed_file = sys.argv[2]
output_cpg_oe_file = sys.argv[3]

# Window parameters
window_size = 500
overlap_size = 0


def count_cpg(seq):
    """Count the number of CpG ('CG') dinucleotides in a sequence."""
    return sum(seq[i:i + 2] == "CG" for i in range(len(seq) - 1))


def calculate_cpg_oe(seq):
    """
    Calculate the CpG observed/expected ratio for a sequence.

    Returns
    -------
    float
        CpG observed/expected ratio.
    str
        "n" if N content exceeds 10% of the configured window size.
        "z" if the expected CpG count is zero.
    """
    c_count = seq.count("C")
    g_count = seq.count("G")
    n_count = seq.count("N")

    observed_cpg = count_cpg(seq)
    expected_cpg = (c_count * g_count) / len(seq)

    if n_count > window_size * 0.1:
        return "n"

    return observed_cpg / expected_cpg if expected_cpg > 0 else "z"


# Store CpG O/E results before writing the final TSV file
cpg_oe_data = []

# Process each chromosome/contig independently
for record in SeqIO.parse(genome_file, "fasta"):
    chromosome = record.id

    # Normalize sequence case before nucleotide counting
    seq = str(record.seq).upper()

    window_start = 0

    while window_start < len(seq):
        # Python uses 0-based, end-exclusive coordinates internally
        window_end = min(window_start + window_size, len(seq))
        window_seq = seq[window_start:window_end]

        cpg_oe_ratio = calculate_cpg_oe(window_seq)
        window_number = window_start // window_size + 1

        # Output coordinates are converted to 1-based, inclusive coordinates
        with open(output_bed_file, "a") as bed:
            bed.write(
                f"{chromosome}\t"
                f"{window_start + 1}\t"
                f"{window_end}\t"
                f"{window_number}\n"
            )

        cpg_oe_data.append(
            [
                chromosome,
                window_start + 1,
                window_end,
                window_number,
                cpg_oe_ratio,
            ]
        )

        # Advance to the next window
        window_start += window_size - overlap_size


# Write CpG O/E results
pd.DataFrame(
    cpg_oe_data,
    columns=["chr", "window_stt", "window_end", "window_number", "CpG_OE"],
).to_csv(
    output_cpg_oe_file,
    index=False,
    sep="\t",
)