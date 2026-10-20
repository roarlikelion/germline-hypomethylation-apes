# Germline hypomethylation shapes dynamic CpG reservoirs in ape genomes

Analysis scripts associated with the bioRxiv preprint:

**Son DR, Loh EY-H, Jeong H, Ma J, Eichler EE, Yi SV. _Germline hypomethylation shapes dynamic CpG reservoirs in ape genomes._ bioRxiv. DOI: 10.64898/2026.06.15.732472.**

This repository contains scripts used for CpG, DNA methylation, Satellite-nonSatellite boundaries, CHEDs, and gene-expression analyses presented in the study.

## Scripts

### CpG and methylation preprocessing

**`Son_et_al_calculate_cpg_oeratio_fromfna.py`**  
Generates fixed genomic windows from a genome FASTA and calculates CpG observed/expected (CpG O/E) ratios.

**`Son_et_al_calculate_methylation_over_windows.py`**  
Calculates mean DNA methylation levels and CpG counts within genomic windows.

### CpG O/E and methylation

**`Son_et_al_methylation_vs_cpgoe_correlationtests.py`**  
Tests relationships between CpG O/E and DNA methylation using Pearson, Spearman, Kendall, and partial correlation analyses.

### Centromere and CT-transition analyses

**`Son_et_al_centromere_methylation_plotting.py`**  
Plots raw and smoothed DNA methylation profiles across centromeric and satellite regions.

**`Son_et_al_CTtransition_methylation_analysis.py`**  
Analyzes DNA methylation patterns relative to centromere-transition (Satellite-to-non-Satellite) boundaries.

**`Son_et_al_CTtransition_methylation_threshold_plot.py`**  
Visualizes chromosome-arm-specific methylation profiles and estimates the distance over which methylation differences remain significant.

**`Son_et_al_CHEDboundaries_pairwise_significance_distance.py`**  
Calculates chromosome-specific distances from CT boundaries over which DNA methylation remains significantly different between pairs of samples or tissues.

### CHED region definition

**`Son_et_al_CHED_Overlapping_region_definition.py`**  
Defines CHED regions from overlapping methylation significance boundaries and merges adjacent CHED-positive windows into continuous genomic intervals.

### Gene-expression analysis

**`Son_et_al_CHED_gene_tissue_expression_rank_analysis.py`**  
Compares tissue-expression rankings between CHED-associated and non-CHED genes using Kolmogorov-Smirnov tests, permutation tests, and Benjamini-Hochberg FDR correction.


## Usage

Input formats, required columns, analysis parameters, statistical procedures, and output files are documented in the header of each script.

## Citation

If you use these scripts, please cite:

Son DR, Loh EY-H, Jeong H, Ma J, Eichler EE, Yi SV. **Germline hypomethylation shapes dynamic CpG reservoirs in ape genomes.** bioRxiv. DOI: **10.64898/2026.06.15.732472**.
