import malariagen_data
import numpy as np
import pandas as pd
import os
import pickle

# ===============================
# CONFIG
# ===============================

CACHE_DIR = "cache_h12_calibration"
os.makedirs(CACHE_DIR, exist_ok=True)

CONTIGS = ["2RL", "3RL", "X"]  # adjust to the contigs covering your regions_list

H12_THRESHOLD = 0.1  # 95th percentile of H12 must fall below this
PERCENTILE = 95

WINDOW_SIZES = (100, 200, 500, 1000, 2000, 5000, 10000, 20000)

sample_sets = [
    "1188-VO-NIANG-NIEL-SN-2304-VMF00259",
    "1330-VO-GN-LAMA-VMF00250",
    "1315-VO-NG-OMITOLA-OMOL-NG-2008",
    "1281-VO-CM-CHRISTOPHE-VMF00227",
    "AG1000G-KE"
]

POP_MAP = {
    "Senegal": sample_sets[0],
    "Guinea": sample_sets[1],
    "Nigeria": sample_sets[2],
    "Cameroon": sample_sets[3],
    "Kenya": sample_sets[4]
}

af1 = malariagen_data.Af1()


# ===============================
# FUNCTIONS
# ===============================

def calibrate_population(pop, sample_set, contig, force_recompute=False):
    """Run h12_calibration for a single population on a single contig and
    return the smallest window size at which the 95th percentile of H12
    drops below H12_THRESHOLD. Returns np.nan if no window size satisfies
    this within the tested range.
    """

    cache_file = os.path.join(
        CACHE_DIR, f"{pop}_{contig}.pkl"
    )

    if os.path.exists(cache_file) and not force_recompute:
        print(f"[CACHE] Loading H12 calibration: {pop} / {contig}")
        with open(cache_file, "rb") as f:
            calib = pickle.load(f)
    else:
        print(f"[COMPUTE] H12 calibration: {pop} / {contig}")
        query = f'(sample_set == "{sample_set}") and (taxon == "funestus")'

        try:
            calib = af1.h12_calibration(
                contig=contig,
                sample_query=query,
                sample_sets=sample_sets,
                window_sizes=WINDOW_SIZES,
                min_cohort_size=15,
                max_cohort_size=50,
            )
        except Exception as e:
            print(f"[WARN] H12 calibration failed for {pop} / {contig}: {e}")
            return np.nan, {}

        with open(cache_file, "wb") as f:
            pickle.dump(calib, f)

    # calib is a Mapping[str(window_size), ndarray of H12 values]
    pct_by_window = {}
    for ws in WINDOW_SIZES:
        key = str(ws)
        if key not in calib:
            continue
        vals = np.asarray(calib[key])
        pct_by_window[ws] = np.nanpercentile(vals, PERCENTILE)

    # smallest window size where the percentile drops below threshold
    recommended = np.nan
    for ws in sorted(pct_by_window):
        if pct_by_window[ws] < H12_THRESHOLD:
            recommended = ws
            break

    if np.isnan(recommended) and pct_by_window:
        # nothing satisfied the threshold; fall back to the largest tested window
        recommended = max(pct_by_window)
        print(f"[WARN] {pop}/{contig}: no window size met threshold "
              f"{H12_THRESHOLD}; falling back to largest tested window "
              f"({recommended}).")

    return recommended, pct_by_window


# ===============================
# RUN: per-population recommended window size, per contig
# ===============================

records = []
for contig in CONTIGS:
    for pop, sset in POP_MAP.items():
        rec_ws, pct_by_window = calibrate_population(pop, sset, contig)
        row = {"contig": contig, "population": pop, "recommended_window_snps": rec_ws}
        row.update({f"p{PERCENTILE}_at_{ws}snp": pct_by_window.get(ws, np.nan)
                     for ws in WINDOW_SIZES})
        records.append(row)

df_calib = pd.DataFrame(records)
df_calib.to_csv("h12_calibration_per_population.csv", index=False)
print("\nPer-population H12 calibration summary:")
print(df_calib[["contig", "population", "recommended_window_snps"]])


# ===============================
# BUILD PER-PAIR WINDOW SIZE FOR H1X (per contig)
# take the LARGER recommended window size of the two populations in a pair,
# since the noisier/lower-diversity population needs the larger window.
# ===============================

pop_list = list(POP_MAP.keys())
pair_records = []

for contig in CONTIGS:
    sub = df_calib[df_calib["contig"] == contig].set_index("population")["recommended_window_snps"]
    for i, p1 in enumerate(pop_list):
        for p2 in pop_list[i + 1:]:
            if p1 not in sub.index or p2 not in sub.index:
                continue
            w1, w2 = sub[p1], sub[p2]
            if np.isnan(w1) or np.isnan(w2):
                pair_ws = np.nan
            else:
                pair_ws = max(w1, w2)
            pair_records.append({
                "contig": contig,
                "pop1": p1,
                "pop2": p2,
                "window_pop1": w1,
                "window_pop2": w2,
                "h1x_window_snps": pair_ws
            })

df_pairs = pd.DataFrame(pair_records)
df_pairs.to_csv("h1x_window_size_per_pair.csv", index=False)
print("\nRecommended H1X window size per population pair (SNP units, from H12 proxy):")
print(df_pairs)

print(
    "\nNOTE: these are SNP-window sizes from the H12 calibration procedure "
    "used as a documented proxy, because malariagen_data has no native "
    "h1x_calibration function. h1x_gwss's window_size argument is in bp, "
    "not SNPs, so you will need to translate the recommended SNP window "
    "into an approximate bp window for your candidate regions (e.g. via "
    "local SNP density) before passing it to h1x_gwss, or re-run this "
    "calibration logic directly against bp windows if you prefer."
)