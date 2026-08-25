# Genomic Signatures of Selection in *Anopheles funestus* Reveal Shared and Population-Specific Adaptive Variation Across African Populations

This repository contains the analysis code associated with the manuscript:

> **Genomic signatures of selection in *Anopheles funestus* reveal shared
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
| `FST_admin1_name.ipynb` | Pairwise FST analysis at admin1 level |
| `testing_gwss_improve_plots.ipynb` | Genome-wide selection scans (H12) |
| `calibrate_h1x.py` | H1X calibration script |
| `new_plot_fst_sliding_h1x_all_regions.py` | Updated FST and H1X sliding window plots |
| `plot_fst_sliding_fsts_all_regions.py` | Pairwise FST sliding window plots |
| `prep_for_isafe.py` | Data preparation for iSAFE analysis |
| `new_isafe_results.ipynb` | iSAFE variant prioritisation results |
| `build_snp_freq_isafe_table.py` | SNP frequency table builder for iSAFE results |
| `table_allcountries_isafe.csv` | iSAFE results — shared variants across all populations |
| `table_popspecific_isafe.csv` | iSAFE results — population-specific variants |
| `diplotype_clustering_for_pic_regions.ipynb` | Diplotype clustering in candidate sweep regions |

## Data

Whole-genome sequence data were obtained from the MalariaGEN *Anopheles
funestus* genomic surveillance project, accessible via the
[malariagen_data API](https://malariagen.github.io/malariagen-data-python/latest/Af1.html).

## Dependencies

See `environment.yml` to reproduce the analysis environment:

```bash
conda env create -f environment.yml
conda activate malariagen-py311
```

## Citation

*To be updated upon publication.*
