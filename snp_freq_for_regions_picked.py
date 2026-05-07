# iporting libraries
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import allel
import pandas as pd
import numpy as np
import math
import re
#pio.renderers.default = "notebook+colab"
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.path import Path
import matplotlib.patches as patches
import bokeh.io as bkio
bkio.output_notebook()
from collections import defaultdict
import malariagen_data

#load malariagen data
af1 = malariagen_data.Af1()
af1

#set a list of sets to be used and load the datasets based on them
list_countries = ['1188-VO-NIANG-NIEL-SN-2304-VMF00259', '1330-VO-GN-LAMA-VMF00250',
                  '1315-VO-NG-OMITOLA-OMOL-NG-2008',
                  '1281-VO-CM-CHRISTOPHE-VMF00227', 'AG1000G-KE'
                  ]

df_samples = af1.sample_metadata(sample_sets=list_countries, sample_query='taxon == "funestus"')

# set the region of interest
regions_list = ['2RL:8000000-9852553', '2RL:40000000-44000000', '2RL:57629228-58000000', '2RL:76000000-76838987', '2RL:80479740-80828789',
                '3RL:12761557-14664404', 'X:8410801-8500000', 'X:14432245-15024921']
# create a dictionary to store the results
regions_contigs_list = []

# iterate over the regions and get the contigs
for region in regions_list:
    region_mRNAs_df = (af1.genome_features(region=region).query('type == "mRNA"')
    [["contig", "start", "end", "ID", "Parent", "strand", "description"]]
    )

    # Calculate length for each mRNA
    region_mRNAs_df['length'] = region_mRNAs_df['end'] - region_mRNAs_df['start']

    # Custom function to apply to each grouped gene
    def aggregate_gene_info(group):
        min_start = group['start'].min()
        max_end = group['end'].max()

        # Find the row with the maximum length within this group
        # Use idxmax() on the 'length' Series within the group to get the index of the max value
        idx_max_length = group['length'].idxmax()
        selected_id = group.loc[idx_max_length, 'ID']

        # Get the description from the row with the selected ID, or just the first description
        description = group['description'].iloc[0] if not group['description'].empty else None

        return pd.Series({
            'start': min_start,
            'end': max_end,
            'ID': selected_id,  # This is the mRNA ID with max length
            'description': description
        })

    # Group by Parent (gene ID), contig, and strand, then apply the custom aggregation function
    consolidated_genes_df = region_mRNAs_df.groupby(['Parent', 'contig', 'strand'], as_index=False).apply(aggregate_gene_info, include_groups=False).reset_index()

    # The 'Parent' from groupby becomes a column after reset_index()
    # Now, rename 'ID' to 'Parent_ID' if you want to explicitly keep the original gene ID,
    # and the 'ID' selected from the apply function as the representative mRNA ID.
    # However, the user asked to match the original structure, where 'ID' is mRNA and 'Parent' is gene.

    # Reorder columns to match the original structure: ID will be mRNA ID, Parent will be gene ID
    region_contigs_df_final = consolidated_genes_df[[
        "contig", "start", "end", "ID", "Parent", "strand", "description"
    ]].copy() # Added .copy() here

    # Add the region column
    region_contigs_df_final['regions'] = region
    # Set the index
    region_contigs_df_final = region_contigs_df_final.set_index("regions")
    regions_contigs_list.append(region_contigs_df_final)

regions_contigs = pd.concat(regions_contigs_list)
regions_contigs.to_csv('pics_regions_contigs_new.csv')

# Now, let's create a new DataFrame with only unique contigs from regions, as they may have overlapping regions
contigs = regions_contigs.reset_index().drop_duplicates(subset='ID')

# gets for each region the snps and their frequencies in each countries
id_list = [(row['regions'], row['ID']) for index, row in contigs.iterrows()]
allele_freq_list = []

for region,id in id_list:
    region_id_snp_freqs = af1.snp_allele_frequencies(
        transcript=id, cohorts='country', sample_query="taxon=='funestus'",
        sample_sets=list_countries, drop_invariant=False
    )
    # Add the ID column
    region_id_snp_freqs['ID'] = id
    # Add the regions column
    region_id_snp_freqs['regions'] = region
    # Set the index
    region_id_snp_freqs = region_id_snp_freqs.set_index(["regions","ID"])
    allele_freq_list.append(region_id_snp_freqs)

allele_frequencies = pd.concat(allele_freq_list)
ns_allele_frequencies = allele_frequencies[allele_frequencies['effect'].isin(['NON_SYNONYMOUS_CODING'])].query("max_af > 0.05")
allele_frequencies.to_csv('new_all_regions_allele_frequencies.csv')
ns_allele_frequencies.to_csv('new_all_regions_ns_allele_frequencies.csv')

#merge to have description between loci
allele_per_contig = allele_frequencies.merge(contigs, on='ID')
allele_per_contig.to_csv('new_all_regions_allele_per_contig_description.csv')

ns_allele_per_contig = ns_allele_frequencies.merge(contigs, on='ID')
ns_allele_per_contig.to_csv('new_all_regions_ns_allele_per_contig_description.csv')