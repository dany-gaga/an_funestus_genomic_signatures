
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


snps_in_region = pd.read_csv('new_all_regions_allele_per_contig_description.csv')


def extract_pos(label):
    # extract number after ':' and remove commas
    match = re.search(r":([\d,]+)", label)
    return int(match.group(1).replace(",", "")) if match else None

snps_in_region["pos"] = snps_in_region["label"].apply(extract_pos)

snps_in_region = snps_in_region.drop_duplicates(
    subset=["pos", "contig", "transcript"]
)

core = pd.read_csv("all_combined_region_isafe_results_af1_new_threshold_0.25.csv")

split_region = core["region"].str.split(":", expand=True)
core["contig"] = split_region[0]

core = core.loc[:, ~core.columns.str.contains("^Unnamed")]

core_wide = core.pivot_table(
    index=["POS", "contig"],
    columns="population",
    values="iSAFE",
    aggfunc="max"   # important: avoids duplicates
).reset_index()

core_wide.columns.name = None
core_wide.columns = ["POS", "contig"] + [f"iSAFE_{c}" for c in core_wide.columns[2:]]

print("snps columns:", snps_in_region.columns.tolist())
print("core_wide columns:", core_wide.columns.tolist())

merged = snps_in_region.merge(
    core_wide,
    left_on=["pos", "contig"],
    right_on=["POS", "contig"],
    how="inner"
)

merged.to_csv("all_snps_for_all_region_with_isafe_results.csv", index=False)