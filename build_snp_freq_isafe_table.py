#!/usr/bin/env python3
# =============================================================================
# build_snp_freq_isafe_table.py
#
# Builds,  the combined SNP-frequency + iSAFE table for the
# An. funestus selection-scan regions, 
#                                                              
# =============================================================================

import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=UserWarning)

# =============================================================================
# CONFIG  ---  edit here, nothing below should need touching
# =============================================================================

OUTDIR = "."

# malariaGEN Af1 sample sets (one per country cohort)
SAMPLE_SETS = [
    "1188-VO-NIANG-NIEL-SN-2304-VMF00259",   # Senegal
    "1330-VO-GN-LAMA-VMF00250",              # Guinea
    "1315-VO-NG-OMITOLA-OMOL-NG-2008",       # Nigeria
    "1281-VO-CM-CHRISTOPHE-VMF00227",        # Cameroon
    "AG1000G-KE",                            # Kenya
]

REGIONS = [
    "2RL:8000000-9852553",
    "2RL:40000000-44000000",
    "2RL:57629228-58000000",
    "2RL:76000000-76838987",
    "2RL:80479740-80828789",
    "3RL:12761557-14664404",
    "X:8410801-8500000",
    "X:14432245-15024921",
]

# Combined iSAFE results (POS, iSAFE, DAF, population, region, filename).

ISAFE_FILE = "all_combined_region_isafe_results_af1_new_threshold_0.25.csv"

COUNTRIES = ["Senegal", "Guinea", "Nigeria", "Cameroon", "Kenya"]
POOLED_LABEL = "Allcountries"

# verify this snps specifically
PIN_SNPS = [
    8702553,   # L261 
    8702578,   # F253 
    8703047,   # N96K
    8703055,   # I94L
    8703059,   # R92R
]

# When AF_alt and pooled DAF disagree by more than this, flag the SNP.
POLARISATION_TOL = 0.15

# =============================================================================
# PART 1 --- per-country SNP allele frequencies from malariaGEN (checkpointed)
# =============================================================================

FREQ_CONTIG_CSV = os.path.join(OUTDIR, "new_all_regions_allele_per_contig_description.csv")
CONTIGS_CSV = os.path.join(OUTDIR, "pics_regions_contigs_new.csv")
FREQ_CACHE = os.path.join(OUTDIR, "freq_cache")   # one CSV per transcript
GENE_ANNOT_CSV = os.path.join(OUTDIR, "gene_annotations_all_regions.csv")


def build_gene_annotations(af1=None):
    """Gene content + annotations for every REGION, ALL mRNA transcripts.

    """
    if os.path.exists(GENE_ANNOT_CSV):
        print(f"[genes] found {GENE_ANNOT_CSV} -> reusing")
        return pd.read_csv(GENE_ANNOT_CSV)
    if af1 is None:
        import malariagen_data
        af1 = malariagen_data.Af1()

    frames = []
    for region in REGIONS:
        g = (
            af1.genome_features(region=region)
            .query('type == "mRNA"')[
                ["contig", "ID", "start", "end", "strand", "description"]
            ]
            .copy()
        )
        g.insert(0, "Region", region)
        frames.append(g)
    genes = pd.concat(frames, ignore_index=True).rename(
        columns={"contig": "Contig", "start": "Start",
                 "end": "End", "strand": "Strand", "description": "Description"}
    )[["Region", "Contig", "ID", "Start", "End", "Strand", "Description"]]
    genes.to_csv(GENE_ANNOT_CSV, index=False)
    print(f"[genes] wrote {GENE_ANNOT_CSV} ({len(genes)} transcripts)")
    return genes


def _consolidate_genes(af1):
    """Consolidate genes per region (longest mRNA per gene). Checkpointed."""
    if os.path.exists(CONTIGS_CSV):
        print(f"[PART 1] found {CONTIGS_CSV} -> reusing gene list")
        return pd.read_csv(CONTIGS_CSV)

    contigs_frames = []
    for region in REGIONS:
        mrna = (
            af1.genome_features(region=region)
            .query('type == "mRNA"')[
                ["contig", "start", "end", "ID", "Parent", "strand", "description"]
            ]
            .copy()
        )
        mrna["length"] = mrna["end"] - mrna["start"]

        def _agg(group):
            idx = group["length"].idxmax()
            return pd.Series(
                {
                    "start": group["start"].min(),
                    "end": group["end"].max(),
                    "ID": group.loc[idx, "ID"],
                    "description": group["description"].iloc[0]
                    if not group["description"].empty
                    else None,
                }
            )

        consolidated = (
            mrna.groupby(["Parent", "contig", "strand"], as_index=False)
            .apply(_agg, include_groups=False)
            .reset_index()
        )
        keep = consolidated[
            ["contig", "start", "end", "ID", "Parent", "strand", "description"]
        ].copy()
        keep["regions"] = region
        contigs_frames.append(keep.set_index("regions"))

    regions_contigs = pd.concat(contigs_frames)
    regions_contigs.to_csv(CONTIGS_CSV)
    return regions_contigs.reset_index()


def compute_allele_frequencies():
    """Pull per-country SNP allele frequencies + gene descriptions.

    """
    if os.path.exists(FREQ_CONTIG_CSV):
        print(f"[PART 1] found {FREQ_CONTIG_CSV} -> skipping malariaGEN pull")
        return pd.read_csv(FREQ_CONTIG_CSV, low_memory=False)

    print("[PART 1] computing allele frequencies from malariaGEN Af1 ...")
    import malariagen_data
    af1 = malariagen_data.Af1()

    contigs = _consolidate_genes(af1).drop_duplicates(subset="ID")
    os.makedirs(FREQ_CACHE, exist_ok=True)

    id_list = [(r["regions"], r["ID"]) for _, r in contigs.iterrows()]
    done, skipped, failed = 0, 0, []
    for region, gid in id_list:
        cache_f = os.path.join(FREQ_CACHE, f"{gid}.csv")
        if os.path.exists(cache_f):
            skipped += 1
            continue
        try:
            freqs = af1.snp_allele_frequencies(
                transcript=gid,
                cohorts="country",
                sample_query="taxon=='funestus'",
                sample_sets=SAMPLE_SETS,
                drop_invariant=False,     # keep everything; we filter later
            )
            freqs["ID"] = gid
            freqs["regions"] = region
            # write to a temp file then rename -> a half-written cache file can
            # never be mistaken for a complete one on the next run.
            tmp = cache_f + ".tmp"
            freqs.reset_index().to_csv(tmp, index=False)
            os.replace(tmp, cache_f)
            done += 1
            print(f"[PART 1]   ok  {gid} ({region})  [{done} done, {skipped} cached]")
        except Exception as e:   # network/timeout/etc -> log and keep going
            failed.append((region, gid, str(e)))
            print(f"[PART 1]   FAIL {gid} ({region}): {e}")

    if failed:
        pd.DataFrame(failed, columns=["regions", "ID", "error"]).to_csv(
            os.path.join(OUTDIR, "freq_failed_transcripts.csv"), index=False
        )
        print(f"[PART 1] {len(failed)} transcript(s) failed -> "
              f"see freq_failed_transcripts.csv. Re-run to retry ONLY those.")

    # assemble from every cached transcript (this run + previous runs)
    cached = [os.path.join(FREQ_CACHE, f) for f in os.listdir(FREQ_CACHE)
              if f.endswith(".csv")]
    if not cached:
        raise RuntimeError("[PART 1] no transcripts cached; nothing to assemble.")
    allele_frequencies = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in cached], ignore_index=True
    )
    allele_frequencies.to_csv(
        os.path.join(OUTDIR, "new_all_regions_allele_frequencies.csv"), index=False
    )

    allele_per_contig = allele_frequencies.merge(
        contigs.drop(columns=[c for c in ["regions"] if c in contigs.columns]),
        on="ID",
    )
    # Only write the "final" checkpoint if nothing failed, so that a partial
    # run doesn't get treated as complete next time.
    if not failed:
        allele_per_contig.to_csv(FREQ_CONTIG_CSV, index=False)
        print(f"[PART 1] wrote {FREQ_CONTIG_CSV}")
    else:
        print("[PART 1] partial result assembled from cache; final CSV NOT "
              "written until all transcripts succeed.")
    return allele_per_contig


# =============================================================================
# PART 2 --- tidy frequencies into long format, keeping POS *and* ALT allele
# =============================================================================

def _pos_from_label(label):
    m = re.search(r":([\d,]+)", str(label))
    return int(m.group(1).replace(",", "")) if m else np.nan


def _alt_from_label(label):
    # label looks like "2RL:8,702,553 A>C (L261W)" -> ref 'A', alt 'C'
    m = re.search(r"\b([ACGT])>([ACGT])\b", str(label))
    return (m.group(1), m.group(2)) if m else (None, None)


def _normalize_regions(df):
    """Collapse regions_x / regions_y / region / index into one 'regions' col.

    """
    df = df.copy()
    if "regions" in df.columns:
        return df
    candidates = [c for c in ["regions_x", "regions_y", "region"] if c in df.columns]
    if candidates:
        reg = df[candidates[0]]
        for c in candidates[1:]:
            reg = reg.fillna(df[c])
        df["regions"] = reg
        df = df.drop(columns=candidates)
    return df


def make_long_frequencies(df_snps):
    """Wide (frq_<Country>) -> long (country, AF_alt), with POS + REF/ALT."""
    df_snps = _normalize_regions(df_snps.copy())
    df_snps["POS"] = df_snps["label"].apply(_pos_from_label).astype("Int64")
    ra = df_snps["label"].apply(_alt_from_label)
    df_snps["REF"] = ra.apply(lambda x: x[0])
    df_snps["ALT"] = ra.apply(lambda x: x[1])

    freq_cols = [c for c in df_snps.columns if c.startswith("frq_")]
    id_vars = [
        c
        for c in [
            "transcript", "effect", "impact", "ref_codon", "alt_codon",
            "aa_pos", "ref_aa", "alt_aa", "label", "POS", "REF", "ALT",
            "max_af", "ID", "regions", "contig", "start", "end", "description",
        ]
        if c in df_snps.columns
    ]
    long = df_snps.melt(
        id_vars=id_vars,
        value_vars=freq_cols,
        var_name="country",
        value_name="AF_alt",
    )
    long["country"] = long["country"].str.replace("frq_", "", regex=False)
    long.to_csv(os.path.join(OUTDIR, "long_format_SNP_freq_for_regions_picked.csv"),
                index=False)
    return long


# =============================================================================
# PART 3 --- load iSAFE, split pooled vs per-country
# =============================================================================

def load_isafe():
    isafe = pd.read_csv(ISAFE_FILE)
    isafe = isafe.rename(columns={"population": "country", "region": "regions"})
    isafe["POS"] = isafe["POS"].astype("Int64")
    # per-country iSAFE (the five cohorts) vs pooled (Allcountries)
    per_country = isafe[isafe["country"].isin(COUNTRIES)].copy()
    pooled = (
        isafe[isafe["country"] == POOLED_LABEL]
        .loc[:, ["POS", "iSAFE", "DAF"]]
        .rename(columns={"iSAFE": "iSAFE_pooled", "DAF": "DAF_pooled"})
        .drop_duplicates("POS")
    )
    return per_country, pooled


# =============================================================================
# PART 4 --- corrected merge
# =============================================================================

def merge_freq_isafe(long, isafe_country, isafe_pooled):
    """LEFT join per-country iSAFE + attach pooled iSAFE on POS.

    """
    # (1) + (2): left join, and bring pooled in on POS regardless of country
    m = long.merge(
        isafe_country[["POS", "country", "iSAFE", "DAF"]].rename(
            columns={"iSAFE": "iSAFE_country", "DAF": "DAF_country"}
        ),
        on=["POS", "country"],
        how="left",
    )
    m = m.merge(isafe_pooled, on="POS", how="left")

    # ---- polarisation / wrong-allele flag --------------------------------
    # If the pooled derived allele is common (DAF_pooled high) but the annotated
    # ALT allele is ~absent (AF_alt ~ 0 across countries), the annotation row is
    # the wrong allele at a multiallelic site -> flag for manual resolution.
    pos_max_afalt = m.groupby("POS")["AF_alt"].transform("max")
    m["polarisation_flag"] = (
        m["DAF_pooled"].notna()
        & (m["DAF_pooled"] - pos_max_afalt.fillna(0)).abs().gt(POLARISATION_TOL)
    )

    # ---- de-duplicate overlapping-gene rows (same variant, >1 transcript) --
    # keep one row per SNP x allele x country, preferring the annotated
    # (amino-acid-bearing) transcript when a bare duplicate exists.
    m["_has_aa"] = m["aa_pos"].notna().astype(int)
    m = (
        m.sort_values(["POS", "ALT", "country", "_has_aa"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["POS", "ALT", "country", "transcript"])
        .drop(columns="_has_aa")
    )
    return m


# =============================================================================
# PART 5 --- pin key SNPs, build wide manuscript table, QC report
# =============================================================================

def _safe_sort(df, by, ascending=True):
    """sort_values but only on columns that actually exist."""
    if isinstance(ascending, bool):
        cols = [c for c in by if c in df.columns]
        asc = ascending
    else:
        pairs = [(c, a) for c, a in zip(by, ascending) if c in df.columns]
        cols = [c for c, _ in pairs]
        asc = [a for _, a in pairs]
    return df.sort_values(cols, ascending=asc) if cols else df


def build_wide_table(matched):
    """One row per SNP (POS x ALT x transcript), one frequency column per
    country, plus pooled iSAFE + DAF and the annotation."""
    ann_cols = [
        "regions", "transcript", "ID", "description", "POS", "REF", "ALT",
        "effect", "impact", "aa_pos", "ref_aa", "alt_aa", "label",
        "iSAFE_pooled", "DAF_pooled", "polarisation_flag",
    ]
    ann_cols = [c for c in ann_cols if c in matched.columns]

    # per-country AF_alt -> wide
    wide_af = matched.pivot_table(
        index=["POS", "ALT", "transcript"],
        columns="country",
        values="AF_alt",
        aggfunc="first",
    )
    wide_af.columns = [f"AF_alt_{c}" for c in wide_af.columns]

    # per-country iSAFE -> wide (population-specific selection signal)
    wide_is = matched.pivot_table(
        index=["POS", "ALT", "transcript"],
        columns="country",
        values="iSAFE_country",
        aggfunc="first",
    )
    wide_is.columns = [f"iSAFE_{c}" for c in wide_is.columns]

    # per-country DAF -> wide (population-specific derived-allele frequency)
    wide_daf = matched.pivot_table(
        index=["POS", "ALT", "transcript"],
        columns="country",
        values="DAF_country",
        aggfunc="first",
    )
    wide_daf.columns = [f"DAF_{c}" for c in wide_daf.columns]

    ann = (
        matched[ann_cols]
        .drop_duplicates(subset=["POS", "ALT", "transcript"])
        .set_index(["POS", "ALT", "transcript"])
    )
    wide = ann.join(wide_af).join(wide_daf).join(wide_is).reset_index()

    # order country columns consistently
    af_order = [f"AF_alt_{c}" for c in COUNTRIES if f"AF_alt_{c}" in wide.columns]
    daf_order = [f"DAF_{c}" for c in COUNTRIES if f"DAF_{c}" in wide.columns]
    is_order = [f"iSAFE_{c}" for c in COUNTRIES if f"iSAFE_{c}" in wide.columns]
    front = [c for c in [
        "regions", "transcript", "ID", "description", "POS", "REF", "ALT",
        "effect", "aa_pos", "ref_aa", "alt_aa", "label",
        "iSAFE_pooled", "DAF_pooled",
    ] if c in wide.columns]
    wide = wide[front + af_order + daf_order + is_order + ["polarisation_flag"]]
    return _safe_sort(wide, ["regions", "POS"]).reset_index(drop=True)


# ---- the two manuscript tables ---------------------------------------------

def build_allcountries_table(wide):
    """Supp-7 style: the POOLED (all-countries) analysis.

    One row per SNP, ranked by the pooled iSAFE, with per-country frequency
    columns. 
    """
    t = wide[wide["iSAFE_pooled"].notna()].copy()
    af_cols = [f"AF_alt_{c}" for c in COUNTRIES if f"AF_alt_{c}" in t.columns]
    keep = [c for c in [
        "regions", "transcript", "POS", "REF", "ALT",
        "iSAFE_pooled", "DAF_pooled",
    ] if c in t.columns] + af_cols + [c for c in [
        "effect", "aa_pos", "ref_aa", "alt_aa", "label",
        "description", "polarisation_flag",
    ] if c in t.columns]
    return _safe_sort(t[keep], ["regions", "iSAFE_pooled"],
                      ascending=[True, False]).reset_index(drop=True)


def build_popspecific_table(wide):
    """Supp-8 style: the POPULATION-SPECIFIC analysis.

    One row per SNP, ranked by the max iSAFE across the country cohorts, with
    the top country and the per-country iSAFE + DAF. Only SNPs that carry at
    least one per-country iSAFE appear
    """
    t = wide.copy()
    is_cols = [f"iSAFE_{c}" for c in COUNTRIES if f"iSAFE_{c}" in t.columns]
    daf_cols = [f"DAF_{c}" for c in COUNTRIES if f"DAF_{c}" in t.columns]
    if not is_cols:
        return t.iloc[0:0]
    t = t[t[is_cols].notna().any(axis=1)].copy()
    t["iSAFE_max"] = t[is_cols].max(axis=1)
    t["top_country"] = (
        t[is_cols].idxmax(axis=1).str.replace("iSAFE_", "", regex=False)
    )
    keep = [c for c in [
        "regions", "transcript", "POS", "REF", "ALT",
        "iSAFE_max", "top_country",
    ] if c in t.columns] + is_cols + daf_cols + [c for c in [
        "effect", "aa_pos", "ref_aa", "alt_aa", "label", "description",
    ] if c in t.columns]
    return _safe_sort(t[keep], ["regions", "iSAFE_max"],
                      ascending=[True, False]).reset_index(drop=True)


def qc_report(long, matched, wide):
    print("\n" + "=" * 70)
    print("QC REPORT")
    print("=" * 70)
    print(f"frequency SNP rows (long)      : {len(long):>8}")
    print(f"merged rows                    : {len(matched):>8}")
    print(f"unique SNPs in wide table      : {wide['POS'].nunique():>8}")
    print(f"SNPs with a pooled iSAFE signal: {wide['iSAFE_pooled'].notna().sum():>8}")
    print(f"SNPs flagged (polarisation)    : {int(wide['polarisation_flag'].sum()):>8}")
    is_cols = [f"iSAFE_{c}" for c in COUNTRIES if f"iSAFE_{c}" in wide.columns]
    n_popspec = int(wide[is_cols].notna().any(axis=1).sum()) if is_cols else 0
    print(f"SNPs in all-countries table    : {wide['iSAFE_pooled'].notna().sum():>8}")
    print(f"SNPs in population-specific tbl : {n_popspec:>8}")

    print("\nPinned SNPs -- did they survive?")
    for pos in PIN_SNPS:
        rows = wide[wide["POS"] == pos]
        if rows.empty:
            print(f"  POS {pos}: *** MISSING *** (no frequency record at this POS)")
            continue
        for _, r in rows.iterrows():
            flag = "  <-- POLARISATION FLAG" if r["polarisation_flag"] else ""
            print(
                f"  POS {pos} {r.get('label','')}: "
                f"iSAFE_pooled={r['iSAFE_pooled']}, DAF_pooled={r['DAF_pooled']}"
                f"{flag}"
            )
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_snps = compute_allele_frequencies()          # PART 1
    long = make_long_frequencies(df_snps)           # PART 2
    isafe_country, isafe_pooled = load_isafe()      # PART 3
    matched = merge_freq_isafe(long, isafe_country, isafe_pooled)  # PART 4

    matched.to_csv(os.path.join(OUTDIR, "snps_isafe_matching_FIXED_long.csv"), index=False)
    wide = build_wide_table(matched)                # PART 5
    wide.to_csv(os.path.join(OUTDIR, "snps_isafe_FIXED_wide_per_region.csv"), index=False)

    allc = build_allcountries_table(wide)           # Supp-7 style (pooled)
    pops = build_popspecific_table(wide)            # Supp-8 style (per-country)
    allc.to_csv(os.path.join(OUTDIR, "table_allcountries_isafe.csv"), index=False)
    pops.to_csv(os.path.join(OUTDIR, "table_popspecific_isafe.csv"), index=False)

    build_gene_annotations()                        # gene table (same REGIONS/source)

    qc_report(long, matched, wide)
    print("\nwrote:")
    print("  snps_isafe_matching_FIXED_long.csv   (one row per SNP x country)")
    print("  snps_isafe_FIXED_wide_per_region.csv (one row per SNP; all metrics)")
    print("  table_allcountries_isafe.csv         (pooled / all-countries table)")
    print("  table_popspecific_isafe.csv          (population-specific / max table)")
    print("  gene_annotations_all_regions.csv     (all mRNA transcripts per region)")


if __name__ == "__main__":
    main()