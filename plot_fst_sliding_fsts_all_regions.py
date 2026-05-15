import malariagen_data
import allel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

import os
import pickle

CACHE_DIR = "cache_region_stats"
os.makedirs(CACHE_DIR, exist_ok=True)
# ===============================
# CONFIG
# ===============================
#regions_list = [
#    '2RL:8000000-9852553',
#    '2RL:76000000-76038987',
#    '3RL:12761557-14664404',
#    'X:14432245-15024921'
#]

regions_list = ['2RL:8000000-9852553', '2RL:40000000-44000000', '2RL:57629228-58000000', 
                '2RL:76000000-76838987', '2RL:80479740-80828789', '3RL:12761557-14664404', 
                'X:8410801-8500000', 'X:14432245-15024921' ]

GENE_COLOR_MAP = {
    "CYTOCHROME": "#d62728",
    "GLUTATHIONE": "#1f77b4",
    "ESTERASE": "#2ca02c",
    "G_PROTEIN": "#9467bd",
    #"GATA": "#7f7f7f",
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
    #"GATA": "GATA",
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
    "Guinea": sample_sets[1],
    "Nigeria": sample_sets[2],
    "Cameroon": sample_sets[3],
    "Kenya": sample_sets[4]
}

af1 = malariagen_data.Af1()

# ===============================
# FUNCTIONS
# ===============================

def compute_region_data(region, force_recompute=False):

    cache_file = os.path.join(
        CACHE_DIR,
        f"{region.replace(':','_')}.pkl"
    )

    # LOAD if exists
    if os.path.exists(cache_file) and not force_recompute:
        print(f"[CACHE] Loading {region}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[COMPUTE] Processing {region}")

    # ===============================
    # ORIGINAL COMPUTATION (unchanged)
    # ===============================
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

    def get_ac(idx):
        return gt.take(idx, axis=1).count_alleles()

    for i, p1 in enumerate(pops):
        for j, p2 in enumerate(pops):
            if i < j:
                ac1 = get_ac(pop_indices[p1])
                ac2 = get_ac(pop_indices[p2])
                num, den = allel.hudson_fst(ac1, ac2)
                fst_val = np.nansum(num) / np.nansum(den)
                matrix[i, j] = matrix[j, i] = fst_val

    matrix = pd.DataFrame(matrix, index=pops, columns=pops)
    matrix = matrix.loc[POP_ORDER, POP_ORDER]

    genes = af1.geneset(region=region)
    genes = genes[genes["type"] == "protein_coding_gene"]

    result = (win_pos, fst, genes, matrix)

    # SAVE cache
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


# ===============================
# PLOTTING
# ===============================

def plot_fst(ax, win_pos, fst):

    ax.scatter(win_pos, fst, s=5, color="lightgrey")
    vmin = np.nanpercentile(fst, 50)
    vmax = np.nanpercentile(fst, 99)

    ax.scatter(
        win_pos,
        fst,
        c=fst,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        s=12
    )

    peak_mask = fst > 0.3
    if peak_mask.any():
        ax.axvspan(win_pos[peak_mask].min(), win_pos[peak_mask].max(),
                   color="red", alpha=0.1)

    ax.set_ylabel("FST", fontsize=11)
    ax.set_xlim(win_pos.min(), win_pos.max())
    ax.set_ylim(0, 0.8)
  
    ax.margins(x=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=True, bottom=True, right=False, top=False)
    ax.set_frame_on(False)

    ax.grid(False)


def plot_genes(ax, genes, win_pos, fst):

    peak_mask = fst > 0.3
    peak_start = win_pos[peak_mask].min() if peak_mask.any() else None
    peak_end   = win_pos[peak_mask].max() if peak_mask.any() else None

    if peak_start is not None:
        ax.axvspan(peak_start, peak_end, color="red", alpha=0.08)

    for _, g in genes.iterrows():

        start, end = g["start"], g["end"]
        strand = g.get("strand", "+")
        y = 0.04 if strand == "+" else -0.04

        name = str(g.get("description", g["ID"])).upper()

        color = next(
            (GENE_COLOR_MAP[key] for key in GENE_COLOR_MAP if key in name),
            GENE_COLOR_MAP["OTHER"],
        )
        ax.plot([start, end], [y, y], color=color, lw=3)

    ax.set_ylim(-0.1, 0.1)
    ax.set_xlim(win_pos.min(), win_pos.max())
    ax.set_ylabel("Gene track", fontsize=11)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.set_yticks([0.04, -0.04])
    ax.set_yticklabels(["+", "–"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=True, bottom=True, right=False, top=False)
    ax.set_frame_on(False)


def plot_heatmap(ax, matrix):

    norm = mpl.colors.Normalize(vmin=0, vmax=0.4)

    sns.heatmap(matrix, cmap="Blues", norm=norm,
                annot=True, fmt=".2f",
                cbar=False, ax=ax)
    
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=25,
    ha="right")

    return norm


def make_region_figure(region, save=True):

    win_pos, fst, genes, matrix = compute_region_data(region)

    # 🔥 align arrays (avoid crash)
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

    # ===============================
    # FST
    # ===============================
    ax_fst.plot(win_pos, fst, color="black", lw=1, alpha=0.7)

    vmin = np.nanpercentile(fst, 50)
    vmax = np.nanpercentile(fst, 99)

    ax_fst.scatter(
        win_pos,
        fst,
        c=fst,
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        s=12
    )

    thr = np.nanpercentile(fst, 95)
    peak_mask = fst > thr

    if peak_mask.sum() > 3:
        ax_fst.axvspan(
            win_pos[peak_mask].min(),
            win_pos[peak_mask].max(),
            color="red",
            alpha=0.1
        )

    ax_fst.set_ylim(0, 0.8)
    ax_fst.set_ylabel("FST")
    ax_fst.set_title(region)

    # ===============================
    # GENES
    # ===============================
    if peak_mask.sum() > 3:
        ax_genes.axvspan(
            win_pos[peak_mask].min(),
            win_pos[peak_mask].max(),
            color="red",
            alpha=0.08
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
    ax_genes.set_ylabel("Gene track")
    ax_genes.set_yticks([0.04, -0.04])
    ax_genes.set_yticklabels(["+", "–"])

    # ===============================
    # HEATMAP
    # ===============================
    norm = mpl.colors.Normalize(vmin=0, vmax=0.4)

    sns.heatmap(
        matrix,
        cmap="Blues",
        norm=norm,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax_heat
    )

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="Blues"),
        ax=ax_heat
    )

    ax_heat.set_xticklabels(
        ax_heat.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    # ===============================
    # CLEAN AXES
    # ===============================
    for ax in [ax_fst, ax_genes, ax_heat]:

        # remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # remove frame
        ax.set_frame_on(False)

        # clean ticks (keep bottom/left only if you want)
        ax.tick_params(top=False, right=False)

        # optional: remove tick marks completely
        # ax.tick_params(length=0)

    # 🔥 VERY IMPORTANT (seaborn heatmap adds edges)
    if ax_heat.collections:
        ax_heat.collections[0].set_edgecolor("face") 
        
    
    # ===============================
    # LEGEND
    # ===============================
    legend_elements = [
    Line2D(
        [0], [0],
        color=GENE_COLOR_MAP[key],
        lw=4 if key != "OTHER" else 2,
        label=GENE_LABEL_MAP[key]
    )
    for key in GENE_COLOR_MAP
]

    fig.legend(
        handles=legend_elements,
        loc="lower left",
        #bbox_to_anchor=(0.1, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8
    )

    plt.subplots_adjust(bottom=0.18)


    # ===============================
    # SAVE
    # ===============================
    if save:
        fig.savefig(
            f"region_{region.replace(':','_')}.png",
            dpi=600   # 🔥 high quality
        )

    return fig


# ===============================
# RUN ALL REGIONS
# ===============================
all_figs = []

for region in regions_list:
    fig = make_region_figure(region)
    fig.savefig(f"region_{region.replace(':','_')}.png", dpi=300)
    all_figs.append(fig)
    plt.close(fig)


# ===============================
# MULTIPANEL A4
# ===============================

fig, axes = plt.subplots(
    len(regions_list),
    1,
    figsize=(8.27, 11.69),
    constrained_layout=True   # 🔥 KEY FIX
)

if len(regions_list) == 1:
    axes = [axes]

for i, region in enumerate(regions_list):

    img_path = f"region_{region.replace(':','_')}.png"
    img = mpimg.imread(img_path)

    axes[i].imshow(img, aspect='auto')   # 🔥 allow stretching
    axes[i].axis("off")

# ===============================
# LEGEND
# ===============================
legend_elements = [
    Line2D(
        [0], [0],
        color=GENE_COLOR_MAP[key],
        lw=4 if key != "OTHER" else 2,
        label=GENE_LABEL_MAP[key]
    )
    for key in GENE_COLOR_MAP
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=8
)

fig.savefig("ALL_REGIONS_A4_clean.png", dpi=300)
plt.show()

