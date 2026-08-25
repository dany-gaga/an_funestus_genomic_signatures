import malariagen_data
import allel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

import os
import pickle

# ===============================
# CACHE — separate dir so old results are untouched
# ===============================
CACHE_DIR = "cache_region_stats_new"
os.makedirs(CACHE_DIR, exist_ok=True)

# ===============================
# CONFIG
# ===============================
regions_list = [
    '2RL:8000000-9852553',
    '2RL:40000000-44000000',
    '2RL:57629228-58000000',
    '2RL:76000000-76838987',
    '2RL:80479740-80828789',
    '3RL:12761557-14664404',
    'X:8410801-8500000',
    'X:14432245-15024921',
]

GENE_COLOR_MAP = {
    "CYTOCHROME": "#d62728",
    "GLUTATHIONE": "#1f77b4",
    "ESTERASE": "#2ca02c",
    "G_PROTEIN": "#9467bd",
    "GAMMA-AMINOBUTYRIC": "#e377c2",
    "SODIUM": "#17becf",
    "CUTICULAR": "#8c564b",
    "ZINC": "#ff7f0e",
    "OTHER": "#bdbdbd"
}

GENE_LABEL_MAP = {
    "CYTOCHROME": "CYP",
    "GLUTATHIONE": "GST",
    "ESTERASE": "Esterase",
    "G_PROTEIN": "Octopamine",
    "GAMMA-AMINOBUTYRIC": "GABA",
    "SODIUM": "Ion transport",
    "CUTICULAR": "Cuticular",
    "ZINC": "Zinc finger",
    "OTHER": "Other genes"
}

POP_ORDER = ["Senegal", "Guinea", "Nigeria", "Cameroon", "Kenya"]

sample_sets = [
    "1188-VO-NIANG-NIEL-SN-2304-VMF00259",
    "1330-VO-GN-LAMA-VMF00250",
    "1315-VO-NG-OMITOLA-OMOL-NG-2008",
    "1281-VO-CM-CHRISTOPHE-VMF00227",
    "AG1000G-KE"
]

POP_MAP = {
    "Senegal": sample_sets[0],
    "Guinea":  sample_sets[1],
    "Nigeria": sample_sets[2],
    "Cameroon": sample_sets[3],
    "Kenya":   sample_sets[4]
}

# ===============================
# H1X WINDOW SIZE — calibrated via H12 proxy (per-population H12 calibration)
# West African pairs (Senegal, Guinea, Nigeria): 1000 SNPs
# Any pair involving Cameroon or Kenya:         10000 SNPs
# ===============================
WEST_AFRICA = {"Senegal", "Guinea", "Nigeria"}
LARGE_WINDOW_POPS = {"Cameroon", "Kenya"}

def get_h1x_window(p1, p2):
    """Return calibrated H1X window size (SNPs) for a population pair."""
    if p1 in LARGE_WINDOW_POPS or p2 in LARGE_WINDOW_POPS:
        return 10000
    return 1000


af1 = malariagen_data.Af1()

# ===============================
# COMPUTATION
# ===============================

def compute_region_data(region, force_recompute=False):

    cache_file = os.path.join(CACHE_DIR, f"{region.replace(':','_')}.pkl")

    if os.path.exists(cache_file) and not force_recompute:
        print(f"[CACHE] Loading {region}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[COMPUTE] Processing {region}")

    ds = af1.snp_calls(region=region, sample_sets=sample_sets)

    gt = allel.GenotypeArray(ds["call_genotype"].values)
    pos = ds["variant_position"].values

    df_samples = af1.sample_metadata(
        sample_sets=sample_sets,
        sample_query='taxon == "funestus"'
    )

    sample_ids = ds["sample_id"].values
    df_meta = df_samples.set_index("sample_id").reindex(sample_ids)

    valid_mask = df_meta["sample_set"].notna().values
    gt = gt[:, valid_mask]
    df_meta = df_meta.iloc[valid_mask]

    pop_indices = {}
    for pop, sset in POP_MAP.items():
        idx = np.where(df_meta["sample_set"] == sset)[0]
        if len(idx) > 0:
            pop_indices[pop] = idx

    subpops = list(pop_indices.values())

    fst, windows, *_ = allel.windowed_weir_cockerham_fst(
        pos, gt, subpops, size=10000
    )
    win_pos = windows.mean(axis=1)

    pops = list(pop_indices.keys())
    matrix = np.zeros((len(pops), len(pops)))

    contig, coords = region.split(":")
    start, end = map(int, coords.split("-"))

    for i, p1 in enumerate(pops):
        for j, p2 in enumerate(pops):
            if i == j:
                matrix[i, j] = 0
                continue
            if i < j:
                q1 = f'(sample_set == "{POP_MAP[p1]}") and (taxon == "funestus")'
                q2 = f'(sample_set == "{POP_MAP[p2]}") and (taxon == "funestus")'

                # --- calibrated window size for this pair ---
                ws = get_h1x_window(p1, p2)
                print(f"  H1X {p1} vs {p2}: window_size={ws} SNPs")

                try:
                    x, h1x, _ = af1.h1x_gwss(
                        contig=contig,
                        window_size=ws,
                        cohort1_query=q1,
                        cohort2_query=q2,
                        sample_sets=sample_sets,
                        min_cohort_size=15
                    )

                    region_mask = (x >= start) & (x <= end)

                    if region_mask.sum() == 0:
                        print(f"[WARN] No H1X windows in region {region} for {p1} vs {p2}")
                        val = np.nan
                    else:
                        val = np.nanmedian(h1x[region_mask])

                except Exception as e:
                    print(f"[WARN] H1X failed for {p1} vs {p2}: {e}")
                    val = np.nan

                matrix[i, j] = matrix[j, i] = val

    matrix = pd.DataFrame(matrix, index=pops, columns=pops)
    matrix = matrix.loc[POP_ORDER, POP_ORDER]

    genes = af1.geneset(region=region)
    genes = genes[genes["type"] == "protein_coding_gene"]

    result = (win_pos, fst, genes, matrix)

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


# ===============================
# PLOTTING
# ===============================

def make_region_figure(region, save=True):

    win_pos, fst, genes, matrix = compute_region_data(region)

    valid = np.isfinite(fst)
    win_pos = win_pos[valid]
    fst = fst[valid]

    fig = plt.figure(figsize=(10, 5))

    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[2, 0.8],
        hspace=0.15,
        wspace=0.25
    )

    ax_fst   = fig.add_subplot(gs[0, 0])
    ax_genes = fig.add_subplot(gs[1, 0], sharex=ax_fst)
    ax_heat  = fig.add_subplot(gs[:, 1])

    # --- FST ---
    ax_fst.plot(win_pos, fst, color="black", lw=1, alpha=0.7)

    vmin = np.nanpercentile(fst, 50)
    vmax = np.nanpercentile(fst, 99)

    ax_fst.scatter(win_pos, fst, c=fst, cmap="Reds", vmin=vmin, vmax=vmax, s=12)

    thr = np.nanpercentile(fst, 95)
    peak_mask = fst > thr

    if peak_mask.sum() > 3:
        ax_fst.axvspan(
            win_pos[peak_mask].min(), win_pos[peak_mask].max(),
            color="red", alpha=0.1
        )

    fst_max = np.nanpercentile(fst, 99.5)
    ax_fst.set_ylim(0, max(fst_max * 1.25, 0.1))
    ax_fst.set_ylabel("FST", fontsize=11, fontweight="bold")
    ax_fst.set_title(region)

    # --- GENES ---
    if peak_mask.sum() > 3:
        ax_genes.axvspan(
            win_pos[peak_mask].min(), win_pos[peak_mask].max(),
            color="red", alpha=0.08
        )

    for _, g in genes.iterrows():
        start, end = g["start"], g["end"]
        strand = g.get("strand", "+")
        y = 0.04 if strand == "+" else -0.04
        name = str(g.get("description", g.get("ID", ""))).upper()
        color = next(
            (GENE_COLOR_MAP[key] for key in GENE_COLOR_MAP if key in name),
            GENE_COLOR_MAP["OTHER"]
        )
        ax_genes.plot([start, end], [y, y], color=color, lw=3)

    ax_genes.set_ylim(-0.1, 0.1)
    ax_genes.set_ylabel("Gene track", fontsize=11, fontweight="bold")
    ax_genes.set_yticks([0.04, -0.04])
    ax_genes.set_yticklabels(["+", "–"])

    # --- HEATMAP ---
    norm = mpl.colors.Normalize(vmin=0, vmax=0.06)

    sns.heatmap(
        matrix,
        cmap="Blues",
        norm=norm,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax_heat
    )

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="Blues"), ax=ax_heat)

    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")
    ax_heat.set_title("Shared haplotype (H1X)", fontsize=11, fontweight="bold", pad=10)

    # --- CLEAN AXES ---
    for ax in [ax_fst, ax_genes, ax_heat]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)
        ax.tick_params(top=False, right=False)

    if ax_heat.collections:
        ax_heat.collections[0].set_edgecolor("face")

    # --- LEGEND ---
    legend_elements = [
        Line2D([0], [0],
               color=GENE_COLOR_MAP[key],
               lw=4 if key != "OTHER" else 2,
               label=GENE_LABEL_MAP[key])
        for key in GENE_COLOR_MAP
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower left",
        ncol=4,
        frameon=False,
        fontsize=8
    )

    plt.subplots_adjust(bottom=0.18)

    # --- SAVE with new_ prefix ---
    if save:
        out_path = f"new_h1x_region_{region.replace(':','_')}.png"
        fig.savefig(out_path, dpi=600)
        print(f"[SAVED] {out_path}")

    return fig


# ===============================
# RUN ALL REGIONS
# ===============================
all_figs = []

for region in regions_list:
    fig = make_region_figure(region)
    all_figs.append(fig)
    plt.close(fig)


# ===============================
# MULTIPANEL A4
# ===============================

fig, axes = plt.subplots(
    len(regions_list),
    1,
    figsize=(8.27, 11.69),
    constrained_layout=True
)

if len(regions_list) == 1:
    axes = [axes]

for i, region in enumerate(regions_list):
    img_path = f"new_h1x_region_{region.replace(':','_')}.png"
    img = mpimg.imread(img_path)
    axes[i].imshow(img, aspect='auto')
    axes[i].axis("off")

# --- LEGEND ---
legend_elements = [
    Line2D([0], [0],
           color=GENE_COLOR_MAP[key],
           lw=4 if key != "OTHER" else 2,
           label=GENE_LABEL_MAP[key])
    for key in GENE_COLOR_MAP
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=8
)

fig.savefig("new_ALL_REGIONS_A4_clean.png", dpi=300)
print("[SAVED] new_ALL_REGIONS_A4_clean.png")
plt.show()