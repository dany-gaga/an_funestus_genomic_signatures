# Genomic Signatures of Selection in Anopheles funestus Reveal Shared 
# and Population-Specific Adaptive Variation Across African Populations

This repository contains the analysis code associated with the manuscript:

> **Genomic signatures of selection in Anopheles funestus reveal shared 
> and population-specific adaptive variation across African populations**


## Overview

This study analyses whole-genome sequence data from 635 *An. funestus* 
mosquitoes across West (Senegal, Guinea, Nigeria), Central (Cameroon), and 
East Africa (Kenya) to characterise population structure and identify targets 
of recent positive selection. Genome-wide selection scans implicate 
detoxification enzymes and neuromodulatory loci — including the GABA receptor 
and octopamine-related genes — as targets of both shared and population-specific 
adaptive processes, highlighting pathways beyond classical metabolic resistance 
as potential targets for next-generation vector control strategies.

## Repository Structure

| File | Description |
|------|-------------|
| `collection_map.ipynb` | Sampling site visualisation |
| `diversity_stats.ipynb` | Nucleotide diversity and Tajima's D statistics |
| `ROH.ipynb` | Runs of homozygosity analysis |
| `plot_roh.ipynb` | ROH visualisation |
| `testing_gwss_improve_plots.ipynb` | Genome-wide selection scans (H12) |
| `prep_for_isafe.py` | Data preparation for iSAFE analysis |
| `new_isafe_results.ipynb` | iSAFE variant prioritisation results |
| `nice_looking_table_for_snps_isafe_results.ipynb` | Summary tables for iSAFE SNPs |
| `nice_looking_table_for_snps_isafe_results.py` | Script version of above |
| `plot_fst_sliding_fsts_all_regions.py` | Pairwise FST sliding window plots |
| `snp_freq_for_regions_picked.py` | Allele frequency analysis in sweep regions |
| `diplotype_clustering_for_pic_regions.ipynb` | Diplotype clustering in candidate regions |

## Data

Whole-genome sequence data were obtained from the MalariaGEN *Anopheles 
funestus* genomic surveillance project, accessible via the 
[malariagen_data API](https://malariagen.github.io/malariagen-data-python/latest/Af1.html).

## Dependencies

- Python 3.11
- malariagen_data
- numpy, pandas, scikit-allel
- matplotlib, seaborn
- iSAFE

## Citation

*To be updated upon publication.*
