import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
import allel
import malariagen_data
import pathlib
import logging

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    filename="isafe_prep.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

# -----------------------------
# 1. Setup Af1 and sample sets
# -----------------------------

af1 = malariagen_data.Af1()

# sample sets / cohorts you want
list_countries = [
    "1188-VO-NIANG-NIEL-SN-2304-VMF00259",  # Senegal
    "1330-VO-GN-LAMA-VMF00250",             # Guinea
    "1315-VO-NG-OMITOLA-OMOL-NG-2008",      # Nigeria
    "1281-VO-CM-CHRISTOPHE-VMF00227",       # Cameroon
    "AG1000G-KE",                           # Kenya
]
#New region list based on multiple overlay gwss
regions_list = ['2RL:8000000-9852553',
                '2RL:40000000-44000000', 
                '2RL:57629228-58000000', 
                '2RL:76000000-76838987', 
                '2RL:80479740-80828789',
                '3RL:12761557-14664404', 
                'X:8410801-8500000', 
                'X:14432245-15024921'
]

# site-level + country-level + global (for these sets)
cohort_queries = {
    # per site
    "Kolda":      "country == 'Senegal' and admin1_name == 'Kolda' and taxon == 'funestus'",
    "Louga":      "country == 'Senegal' and admin1_name == 'Louga' and taxon == 'funestus'",
    "SaintLouis": "country == 'Senegal' and admin1_name == 'Saint Louis' and taxon == 'funestus'",
    "South":      "country == 'Cameroon' and admin1_name == 'South' and taxon == 'funestus'",
    "Ogun":       "country == 'Nigeria' and admin1_name == 'Ogun' and taxon == 'funestus'",
    "Oyo":        "country == 'Nigeria' and admin1_name == 'Oyo' and taxon == 'funestus'",
    "Kankan":     "country == 'Guinea' and admin1_name == 'Kankan' and taxon == 'funestus'",
    "Nzerekore":  "country == 'Guinea' and admin1_name == 'Nzerekore' and taxon == 'funestus'",
    "Kilifi":     "country == 'Kenya' and admin1_name == 'Kilifi' and taxon == 'funestus'",

    # country-level
    "Senegal":   "country == 'Senegal'  and taxon == 'funestus'",
    "Guinea":    "country == 'Guinea'   and taxon == 'funestus'",
    "Nigeria":   "country == 'Nigeria'  and taxon == 'funestus'",
    "Cameroon":  "country == 'Cameroon' and taxon == 'funestus'",
    "Kenya":     "country == 'Kenya'    and taxon == 'funestus'",

    # one big cohort using just these sample sets
    "All_countries": "taxon == 'funestus'",
}

# sample metadata for the chosen sample sets only
df_samples = af1.sample_metadata(sample_sets=list_countries)


def make_isafe_inputs_for_region(
    region: str,
    cohort_name: str,
    cohort_query: str,
    out_dir: str = "prep_isafe_inputs",
    analysis: str = "funestus",
):
    """
    For a given genomic region and cohort (defined by cohort_query on df_samples),
    create a hap file for iSAFE:

      <out_dir>/<cohort_name>__<region>.hap

    Columns:
      col0 = genomic position (bp)
      col1..2N = haplotypes (0/1)
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[*] {cohort_name} | {region} : loading haplotypes...")
    ds_haps = af1.haplotypes(
        region=region,
        analysis=analysis,
        sample_sets=list_countries,
    )

    samples_phased = ds_haps["sample_id"].values

    # join metadata to haplotype samples (tolerant: missing metadata → NaN)
    df_samples_phased = (
        pd.DataFrame({"sample_id": samples_phased})
        .join(df_samples.set_index("sample_id"), on="sample_id")
    )

    n_missing = df_samples_phased["country"].isna().sum()
    if n_missing > 0:
        logging.info(
            f"    {cohort_name} | {region}: "
            f"{n_missing} samples in haplotypes have no metadata; ignored for cohort selection."
        )

    # apply cohort query
    df_cohort = df_samples_phased.query(cohort_query)
    loc_cohort_samples = df_cohort.index.to_numpy()

    if len(loc_cohort_samples) == 0:
        logging.info(f"    {cohort_name} | {region}: no samples match cohort_query → skipping.")
        return

    # subset haplotypes
    ds_haps_cohort = ds_haps.isel(samples=loc_cohort_samples)
    cohort_sample_ids = ds_haps_cohort["sample_id"].values

    logging.info(
        f"    {cohort_name} | {region}: {len(cohort_sample_ids)} samples "
        f"({2*len(cohort_sample_ids)} haplotypes)"
    )

    # genotypes → haplotypes
    gt = allel.GenotypeDaskArray(ds_haps_cohort["call_genotype"].data)
    ht = gt.to_haplotypes()
    ht_np = ht.compute().astype("int8")  # (variants, 2*n_samples)

    # positions
    pos = np.asarray(ds_haps_cohort["variant_position"].values, dtype=int)

    if ht_np.shape[0] != pos.shape[0]:
        raise RuntimeError(
            f"Variant count mismatch: haplotypes={ht_np.shape[0]}, positions={pos.shape[0]}"
        )

    # sort by position
    order = np.argsort(pos)
    pos_sorted = pos[order]
    ht_sorted = ht_np[order, :]

    combined = np.column_stack([pos_sorted, ht_sorted])

    # safe region string for filename
    safe_region = region.replace(":", "_").replace("-", "_")
    prefix = out_dir / f"{cohort_name}__{safe_region}"

    # write hap file: TAB-separated, no header
    np.savetxt(prefix.with_suffix(".hap"), combined, fmt="%d", delimiter="\t")

    # optional: sample IDs for bookkeeping
    with open(prefix.with_suffix(".samples"), "w") as f:
        for sid in cohort_sample_ids:
            f.write(f"{sid}\n")

    logging.info(
        f"    {cohort_name} | {region}: "
        f"{ht_np.shape[0]} SNPs, {ht_np.shape[1]} haplotypes "
        f"→ {prefix}.hap"
    )


def main():
    for cohort_name, cq in cohort_queries.items():
        for region in regions_list:
            try:
                make_isafe_inputs_for_region(
                    region=region,
                    cohort_name=cohort_name,
                    cohort_query=cq,
                    out_dir="prep_isafe_inputs",
                )
            except Exception as e:
                logging.error(f"[ERROR] {cohort_name} | {region}: {e}")


if __name__ == "__main__":
    main()