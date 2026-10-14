#!/usr/bin/env python3
# ======================================================================================
#  make_4panel_figure.py
#
#  ONE self-contained script that builds the whole composite figure and saves it as a
#  single high-DPI PNG:
#
#      A  collection map (bright)      C  neighbour-joining tree   [aligned grid layout]
#      B  PCA + zoomed inset           D  pairwise Fst heatmap (bright)
#
#  Data is pulled from the MalariaGEN API and cached to disk, so the slow steps (PCA,
#  NJ tree, Fst jackknife) run once; restyling afterwards is instant. Run:
#
#      python make_4panel_figure.py
#
#  Requires (your MRCG env has these): malariagen_data, numpy, pandas, matplotlib,
#  seaborn, cartopy (or a local Natural Earth shapefile), and kaleido (for the tree
#  image export:  pip install -U kaleido).
# ======================================================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image
import seaborn as sns

warnings.filterwarnings("ignore")

# ======================================================================================
#  CONFIG  — the part you normally edit
# ======================================================================================
USE_API   = False                # False -> build entirely from the cache (no API calls,
                                 #          no recompute; only re-renders the panels).
                                 #          Set True only to (re)populate the cache.
CACHE_DIR = "panel_cache"
OUTDIR    = "."
FINAL_PNG = "figure_4panel_aligned.png"
FINAL_DPI = 300

SITE_FILTERS = "sc_20220908"
SAMPLE_SETS = [
    "1188-VO-NIANG-NIEL-SN-2304-VMF00259",   # Senegal
    "1330-VO-GN-LAMA-VMF00250",              # Guinea
    "1315-VO-NG-OMITOLA-OMOL-NG-2008",       # Nigeria
    "1281-VO-CM-CHRISTOPHE-VMF00227",        # Cameroon
    "AG1000G-KE",                            # Kenya
]
SAMPLE_QUERY = "taxon == 'funestus'"

# Region / SNPs used for PCA *and* the NJ tree (from your PCA___NJT notebook)
ANALYSIS_REGION = "2RL:57604655-90000000"
N_SNPS          = 100_000

# Fst (Panel D), PCA (Panel B) and the NJ tree (Panel C) all use the SAME region.
FST_REGION = ANALYSIS_REGION     # 2RL:57,604,655-90,000,000
N_JACK     = 200

# Fallback variance labels if the cached PCA has no evr saved
PC1_VAR_FALLBACK = "3.89%"
PC2_VAR_FALLBACK = "0.68%"

# Basemap for Panel A:  "cartopy"  (Natural Earth features)  or  "shapefile"
BASEMAP = "cartopy"
NE_DIR  = "ne"                   # only used when BASEMAP == "shapefile"

FST_CMAP = "crest"               # teal->blue heatmap for Panel D
                                 # ("crest"/"mako" = seaborn teal-blues; "aqua" = the
                                 #  custom mint->teal->blue ramp defined in _cmaps())

# One shared, vivid categorical palette for A / B / C
COUNTRY_COLORS = {
    "Senegal": "#2E86DE", "Guinea": "#D4AC0D", "Nigeria": "#1DB954",
    "Cameroon": "#8E44AD", "Kenya": "#E74C3C",
}
ORDER = ["Senegal", "Guinea", "Nigeria", "Cameroon", "Kenya"]

# The PCA callout + inset zoom into ONLY these West African sites (Cameroon and Kenya
# stay in the main scatter but are excluded from the zoom).
WEST = ["Senegal", "Guinea", "Nigeria"]

# Panel C (NJ tree): point this at a ready-made tree PNG (e.g. your correct tree image)
# to use it verbatim with no recompute. Leave None to use the cached export, or to
# regenerate via plot_njt when USE_API=True (or COMPUTE_NJT=True).
NJT_OVERRIDE_PNG = "njt_tree1.png"  # None or path to a real tree PNG (e.g. your exported NJ tree)

# Compute ONLY the NJ tree from the API (everything else still loads from cache).
# Needs kaleido for the PNG export:  pip install -U kaleido
COMPUTE_NJT = True

# Auto-trim white margins from an override / cached tree image so it fills Panel C.
TRIM_NJT_WHITESPACE = True
# Drop the plotly header text ("Sample sets / query / region") above the tree, keeping
# just the tree + its legend, and upscale+sharpen it for a crisp Panel C.
NJT_DROP_HEADER = True
NJT_UPSCALE     = 3.0        # 1.0 = no upscale; 3.0 gives a sharp tree in the panel
# Crop the tree's own (baked-in) legend off, and rebuild a shared country legend that
# sits between the map (A) and the tree, like the earlier layout.
CROP_TREE_LEGEND  = True
ADD_SHARED_LEGEND = True
# Draw one connector line from the PCA zoom box ("quadruplet") to the inset.
CONNECT_INSET = True

WHITE, INK = "#FFFFFF", "#1A1A1A"
OCEAN_COL, LAND_COL = "#CFE7F7", "#ECE9E1"
BORDER_COL, COAST_COL = "#BFBAB0", "#9AA6AE"

# hand-tuned per-admin label offsets (dlon, dlat in degrees) for the map
ADMIN_OFFSETS = {
    "Saint Louis": (0.2, 1.4), "Louga": (-2.4, 0.4), "Kaolack": (-2.4, -0.4),
    "Kolda": (-2.4, -1.3), "Kedougou": (-0.2, -2.1), "Kankan": (1.5, 0.5),
    "Nzerekore": (-1.8, -1.2), "Ogun": (0.3, -1.8), "Oyo": (0.3, 1.5),
    "South": (1.4, 0.0), "Kilifi": (1.4, 0.0),
}
COUNTRY_LABEL_POS = {
    "Senegal": (-14.6, 15.0), "Guinea": (-11.2, 11.4), "Nigeria": (7.6, 9.8),
    "Cameroon": (12.0, 5.0), "Kenya": (37.2, 1.0),
}
MAP_EXTENT = [-20, 46, -6.5, 18.5]


# ======================================================================================
#  BRIGHT FST HELPERS
# ======================================================================================
def _cmaps():
    return {
        "flare": sns.color_palette("flare", as_cmap=True),
        "crest": sns.color_palette("crest", as_cmap=True),
        "rocket": sns.color_palette("rocket_r", as_cmap=True),
        "mako": sns.color_palette("mako_r", as_cmap=True),
        "sunrise": LinearSegmentedColormap.from_list(
            "sunrise", ["#FFF7B0", "#FFC24B", "#FB6A4A", "#DE2D8B", "#6A2C91"]),
        "aqua": LinearSegmentedColormap.from_list(
            "aqua", ["#E8FBEF", "#5AD1B0", "#1FA3C9", "#2E5FD0", "#20135F"]),
    }

def _resolve_cmap(cmap):
    if isinstance(cmap, mpl.colors.Colormap):
        return cmap
    p = _cmaps()
    return p[cmap] if cmap in p else plt.get_cmap(cmap)

def _fst_matrix(fst_df, order):
    vcol = "fst" if "fst" in fst_df.columns else [c for c in fst_df.columns if "fst" in c.lower()][0]
    scol = "se" if "se" in fst_df.columns else None
    idx = {l: i for i, l in enumerate(order)}
    n = len(order)
    F = np.full((n, n), np.nan); SE = np.full((n, n), np.nan)
    for r in fst_df.itertuples(index=False):
        a, b = r.cohort1, r.cohort2
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            F[i, j] = F[j, i] = getattr(r, vcol)
            if scol:
                SE[i, j] = SE[j, i] = getattr(r, scol)
    return F, SE

def _text_color(rgba):
    return "white" if (0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]) < 0.55 else INK


# ======================================================================================
#  ALIGNED LAYOUT  —  every panel is drawn straight into ONE figure on a shared grid
#
#      A (map)   | C (tree + legend)      row 1: A and C have the SAME top and bottom edge
#      B (PCA)   | D (Fst heatmap)        row 2: B and D have the SAME top and bottom edge
#
#  Column 1: A and B share the same left and right edge.  Column 2: C and D share the
#  same left and right edge.  (Sizes below are in inches.)
# ======================================================================================
ROW_H      = 5.0     # height of the frames in row 1 (A, C)
LEFT_W     = 9.0     # width of column 1 (A and B)
LEGEND_W   = 1.45    # country legend block, left of the tree (set 0 to drop the legend)
HEAT_LBL_W = 0.95    # space for the row labels, left of the heatmap
CB_GAP, CB_W, CB_LBL_W = 0.15, 0.18, 0.70    # colour bar gap / width / room for its labels
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 0.95, 0.25, 0.55, 0.95
GAP_X, GAP_Y = 0.45, 0.65
LETTER_FS = 26


def draw_A(fig, rect, sites):
    sites = sites.dropna(subset=["latitude", "longitude", "country"])
    lon0, lon1, lat0, lat1 = MAP_EXTENT

    if BASEMAP == "cartopy":
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        ax = fig.add_axes(rect, projection=proj)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=proj); ax.set_aspect("auto")
        ax.set_facecolor(OCEAN_COL)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN_COL, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COL, zorder=1)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=OCEAN_COL,
                       edgecolor=BORDER_COL, linewidth=0.3, zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5,
                       edgecolor=BORDER_COL, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6,
                       edgecolor=COAST_COL, zorder=3)
        tf = {"transform": proj}
    else:
        import glob, geopandas as gpd
        ax = fig.add_axes(rect)
        ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_aspect("auto")
        ax.set_facecolor(OCEAN_COL)
        pick = lambda pat: (sorted(glob.glob(os.path.join(NE_DIR, pat))) or [None])[0]
        land = gpd.read_file(pick("ne_*m_admin_0_countries.shp"))
        if pick("ne_*m_ocean.shp"):
            gpd.read_file(pick("ne_*m_ocean.shp")).plot(ax=ax, color=OCEAN_COL, zorder=0)
        land.plot(ax=ax, color=LAND_COL, edgecolor=BORDER_COL, linewidth=0.4, zorder=1)
        if pick("ne_*m_lakes.shp"):
            gpd.read_file(pick("ne_*m_lakes.shp")).plot(ax=ax, color=OCEAN_COL,
                          edgecolor=BORDER_COL, linewidth=0.3, zorder=2)
        namecol = "NAME" if "NAME" in land.columns else "ADMIN"
        for c in ORDER:
            poly = land[land[namecol] == c]
            if len(poly):
                poly.plot(ax=ax, color=COUNTRY_COLORS[c], alpha=0.10,
                          edgecolor=COUNTRY_COLORS[c], linewidth=0.6, zorder=1.5)
        tf = {}

    for country, grp in sites.groupby("country"):
        ax.scatter(grp["longitude"], grp["latitude"], s=150,
                   c=COUNTRY_COLORS.get(country, "#888"), edgecolors="white",
                   linewidths=1.6, zorder=6, **tf)

    cent = sites.groupby(["country", "admin1_name"])[["latitude", "longitude"]].mean().reset_index()
    for _, r in cent.iterrows():
        la0, lo0 = float(r["latitude"]), float(r["longitude"])
        dlon, dlat = ADMIN_OFFSETS.get(r["admin1_name"], (0.6, 0.6))
        ax.text(lo0 + dlon, la0 + dlat, r["admin1_name"], fontsize=7.5, fontweight="bold",
                color=INK, ha="left" if dlon >= 0 else "right",
                va="bottom" if dlat >= 0 else "top", zorder=8,
                path_effects=[pe.withStroke(linewidth=3.0, foreground=WHITE)], **tf)
        ax.plot([lo0, lo0 + dlon*0.72], [la0, la0 + dlat*0.72], color=INK, lw=0.5,
                alpha=0.45, zorder=5, **tf)

    for c, (lon, lat) in COUNTRY_LABEL_POS.items():
        ax.text(lon, lat, c.upper(), fontsize=9, fontweight="bold", fontstyle="italic",
                fontfamily="serif", color=COUNTRY_COLORS[c], ha="center", va="center",
                zorder=9, path_effects=[pe.withStroke(linewidth=2.4, foreground=WHITE)], **tf)

    nx, ny = lon1 - 3.0, lat1 - 3.5
    ax.annotate("", xy=(nx, ny + 2.4), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=13),
                zorder=20, **({"xycoords": tf["transform"], "textcoords": tf["transform"]} if tf else {}))
    ax.text(nx, ny - 0.4, "N", fontsize=8, fontweight="bold", fontfamily="serif",
            color=INK, ha="center", va="top", zorder=20,
            path_effects=[pe.withStroke(linewidth=2, foreground=WHITE)], **tf)

    xb0, xb1, yb, h = lon0 + 1.5, lon0 + 10.5, lat0 + 1.0, 0.28
    xm = (xb0 + xb1) / 2
    for px, pc in [(xb0, WHITE), (xm, INK)]:
        ax.add_patch(mpatches.Rectangle((px, yb), (xb1 - xb0)/2, h, facecolor=pc,
                     edgecolor=INK, linewidth=0.7, zorder=12,
                     **({"transform": tf["transform"]} if tf else {})))
    for xp, lbl in [(xb0, "0"), (xb1, "~1000 km")]:
        ax.text(xp, yb + h + 0.15, lbl, fontsize=6.5, fontfamily="serif", color=INK,
                ha="center", va="bottom", zorder=13,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=WHITE)], **tf)

    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.0); sp.set_edgecolor("#444444")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def draw_B(fig, rect, pca, pc1_var, pc2_var):
    ax = fig.add_axes(rect)
    for c in ORDER:
        d = pca[pca["country"] == c]
        if len(d):
            ax.scatter(d["PC1"], d["PC2"], s=34, c=COUNTRY_COLORS[c],
                       edgecolors="white", linewidths=0.4, alpha=0.9, zorder=3)
    ax.set_xlabel(f"PC1 ({pc1_var})", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pc2_var})", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # tight zoom window around the West-African cluster (SN/GN/NG)
    sub = pca[pca["country"].isin(WEST)]
    xpad = (sub["PC1"].max() - sub["PC1"].min())*0.18 + 1.0
    ypad = (sub["PC2"].max() - sub["PC2"].min())*0.18 + 1.0
    zx0, zx1 = sub["PC1"].min() - xpad, sub["PC1"].max() + xpad
    zy0, zy1 = sub["PC2"].min() - ypad, sub["PC2"].max() + ypad

    axin = ax.inset_axes([0.46, 0.46, 0.52, 0.50])
    for c in WEST:
        d = sub[sub["country"] == c]
        if len(d):
            axin.scatter(d["PC1"], d["PC2"], s=26, c=COUNTRY_COLORS[c],
                         edgecolors="white", linewidths=0.3, alpha=0.9, zorder=3)
    axin.set_xlim(zx0, zx1); axin.set_ylim(zy0, zy1)
    axin.set_xlabel(f"PC1 ({pc1_var})", fontsize=8); axin.set_ylabel(f"PC2 ({pc2_var})", fontsize=8)
    axin.tick_params(labelsize=7)
    for sp in ["top", "right"]:
        axin.spines[sp].set_visible(False)
    axin.set_facecolor("#FBFBFD")

    ax.add_patch(mpatches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                 edgecolor="#555", linewidth=1.1, zorder=4))
    if CONNECT_INSET:
        from matplotlib.patches import ConnectionPatch
        con = ConnectionPatch(xyA=(zx1, zy1), coordsA=ax.transData,
                              xyB=(0.0, 0.0), coordsB=axin.transAxes,
                              color="#555", linewidth=1.0, zorder=2)
        con.set_annotation_clip(False)
        ax.add_artist(con)
    return ax


def trim_white(img, thr=250):
    """Crop away white margins so the visible content touches the image edge."""
    a = np.asarray(img.convert("RGB")); nz = a.min(axis=2) < thr
    ys = np.where(nz.any(axis=1))[0]; xs = np.where(nz.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return img
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def draw_C(fig, rect_tree, rect_leg, tree_img):
    ax = fig.add_axes(rect_tree); ax.axis("off")
    if tree_img is not None:
        ax.imshow(tree_img, aspect="equal")
    else:
        ax.text(0.5, 0.5, "NJ tree not available.\n\nSet NJT_OVERRIDE_PNG to your tree image.",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color=INK,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#F4F4F6", edgecolor="#CCCCCC"))
    if rect_leg is not None:
        axl = fig.add_axes(rect_leg); axl.axis("off")
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=12,
                          markerfacecolor=COUNTRY_COLORS[c], markeredgecolor="white", label=c)
                   for c in ORDER]
        axl.legend(handles=handles, title="Country", loc="center left", frameon=False,
                   fontsize=13, title_fontsize=14, labelspacing=1.05, handletextpad=0.7,
                   borderaxespad=0.0)
    return ax


def draw_D(fig, rect_heat, rect_cb, fst_df):
    F, SE = _fst_matrix(fst_df, ORDER)
    n = len(ORDER)
    cmap = _resolve_cmap(FST_CMAP).copy(); cmap.set_bad("white")
    vmax = np.nanmax(F)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    ax = fig.add_axes(rect_heat)
    im = ax.imshow(np.ma.masked_invalid(F), cmap=cmap, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(F[i, j]):
                continue
            ax.text(j, i, f"{F[i, j]:0.3f}", ha="center", va="center", fontsize=11,
                    fontweight="bold", color=_text_color(cmap(norm(F[i, j]))))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ORDER, fontsize=11, fontstyle="italic")
    ax.set_yticklabels(ORDER, fontsize=11, fontstyle="italic")
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes(rect_cb)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Pairwise $F_{ST}$", fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=9); cb.outline.set_visible(False)
    return ax


def compose_aligned(sites, pca_df, pc1_var, pc2_var, tree_img, fst_df, out=FINAL_PNG):
    """All four panels on one grid: A|C share their top/bottom edges, B|D share theirs;
    A and B share their left/right edges, C and D share theirs."""
    leg_w = LEGEND_W if ADD_SHARED_LEGEND else 0.0
    if tree_img is not None:
        tree_img = trim_white(tree_img)          # circle now touches the cell edges -> exact alignment with A
    tree_asp = (tree_img.width / tree_img.height) if tree_img is not None else 1.0
    H1 = ROW_H
    tree_w = H1 * tree_asp                       # tree circle fills the full row-1 height
    RW = leg_w + tree_w                          # width of column 2
    heat_side = RW - HEAT_LBL_W - CB_GAP - CB_W - CB_LBL_W
    H2 = heat_side                               # row-2 height = heatmap side (square cells)
    LW = LEFT_W

    figW = MARGIN_L + LW + GAP_X + RW + MARGIN_R
    figH = MARGIN_T + H1 + GAP_Y + H2 + MARGIN_B
    xL, xR = MARGIN_L, MARGIN_L + LW + GAP_X
    top1, top2 = MARGIN_T, MARGIN_T + H1 + GAP_Y

    def R(x, top, w, h):                          # inches -> figure fractions
        return [x/figW, 1 - (top + h)/figH, w/figW, h/figH]

    fig = plt.figure(figsize=(figW, figH), facecolor=WHITE)
    draw_A(fig, R(xL, top1, LW, H1), sites)
    draw_C(fig, R(xR + leg_w, top1, tree_w, H1),
           R(xR, top1, leg_w, H1) if leg_w > 0 else None, tree_img)
    draw_B(fig, R(xL, top2, LW, H2), pca_df, pc1_var, pc2_var)
    draw_D(fig, R(xR + HEAT_LBL_W, top2, heat_side, H2),
           R(xR + HEAT_LBL_W + heat_side + CB_GAP, top2, CB_W, H2), fst_df)

    # panel letters: left-aligned with their column, sitting just above each frame
    for letter, x, top in [("A", xL, top1), ("C", xR, top1), ("B", xL, top2), ("D", xR, top2)]:
        fig.text(x/figW, 1 - (top - 0.06)/figH, letter, fontsize=LETTER_FS, fontweight="bold",
                 va="bottom", ha="left", color=INK)

    fig.savefig(os.path.join(OUTDIR, out), dpi=FINAL_DPI, facecolor=WHITE)
    plt.close(fig)
    return out


# ======================================================================================
#  DATA LAYER  —  pull from API with caching
# ======================================================================================
def _cache(path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, path)

def get_sites(af1):
    p = _cache("sites.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    if af1 is None:
        raise FileNotFoundError(p)
    md = af1.sample_metadata(sample_sets=SAMPLE_SETS, sample_query=SAMPLE_QUERY)
    sites = md[["country", "admin1_name", "latitude", "longitude"]].dropna()
    sites.to_csv(p, index=False)
    return sites

def get_pca(af1):
    pc, ev = _cache("pca.csv"), _cache("pca_evr.npy")
    if os.path.exists(pc):
        evr = np.load(ev) if os.path.exists(ev) else None
        return pd.read_csv(pc), evr
    if af1 is None:
        raise FileNotFoundError(pc)
    pca_df, evr = af1.pca(region=ANALYSIS_REGION, n_snps=N_SNPS,
                          sample_sets=SAMPLE_SETS, sample_query=SAMPLE_QUERY)
    pca_df.to_csv(pc, index=False); np.save(ev, np.asarray(evr))
    return pca_df, evr

def make_country_legend_png(out="panelC_legend.png"):
    """Render a standalone country legend (dots + names, 'Country' title), tightly
    trimmed, to sit between the map and the tree."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(1.7, 2.3), facecolor=WHITE)
    ax.axis("off")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=12,
                      markerfacecolor=COUNTRY_COLORS[c], markeredgecolor="white", label=c)
               for c in ORDER]
    ax.legend(handles=handles, title="Country", loc="center", frameon=False,
              fontsize=13, title_fontsize=14, labelspacing=1.05, handletextpad=0.7)
    p = os.path.join(OUTDIR, out)
    fig.savefig(p, dpi=FINAL_DPI, bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    plt.close(fig)
    im = Image.open(p).convert("RGB")               # tight-trim white
    a = np.asarray(im); nz = a.min(axis=2) < 250
    ys = np.where(nz.any(axis=1))[0]; xs = np.where(nz.any(axis=0))[0]
    im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).save(p, dpi=(FINAL_DPI, FINAL_DPI))
    return p


def prepare_njt_image(path):
    """Crop a plotly NJ-tree export down to just the tree + legend (dropping the
    'Sample sets / query / region' header), tight-trim margins, and upscale+sharpen
    so it renders crisply in Panel C. Returns a PIL image."""
    from PIL import ImageFilter
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    nonwhite = a.min(axis=2) < 245
    content = nonwhite.mean(axis=1) > 0.04            # per-row density (ignore faint AA)

    # split rows into content bands separated by near-white gaps
    GAP = 8; segs = []; start = None; gap = 0
    for y, c in enumerate(content):
        if c:
            if start is None: start = y
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= GAP:
                    segs.append((start, y - gap + 1)); start = None; gap = 0
    if start is not None:
        segs.append((start, len(content) - gap))
    if not segs:
        return im

    # drop the header: keep the TALLEST band (the tree) and anything below it
    if NJT_DROP_HEADER and len(segs) >= 2:
        main = max(segs, key=lambda s: s[1] - s[0])
        segs = [s for s in segs if s[0] >= main[0]]

    y0, y1 = segs[0][0], segs[-1][1]

    # columns within the kept rows -> optionally drop the tree's own legend strip
    colden = nonwhite[y0:y1].mean(axis=0)
    colcon = colden > 0.02
    CGAP = 15; cs = []; cst = None; cg = 0
    for x, c in enumerate(colcon):
        if c:
            if cst is None: cst = x
            cg = 0
        else:
            if cst is not None:
                cg += 1
                if cg >= CGAP:
                    cs.append((cst, x - cg + 1)); cst = None; cg = 0
    if cst is not None:
        cs.append((cst, len(colcon) - cg))
    if CROP_TREE_LEGEND and len(cs) >= 2:
        main_c = max(cs, key=lambda s: s[1] - s[0])     # widest block = the tree circle
        x0, x1 = main_c
    else:
        xs = np.where(colcon)[0]; x0, x1 = xs.min(), xs.max() + 1

    pad = 10
    out = im.crop((max(0, x0 - pad), max(0, y0 - pad),
                   min(a.shape[1], x1 + pad), min(a.shape[0], y1 + pad)))
    print(f"  [njt] cropped tree {im.size} -> {out.size}")

    if NJT_UPSCALE and NJT_UPSCALE != 1.0:
        out = out.resize((round(out.width * NJT_UPSCALE), round(out.height * NJT_UPSCALE)),
                         Image.LANCZOS)
        out = out.filter(ImageFilter.UnsharpMask(radius=2.0, percent=120, threshold=2))
        print(f"  [njt] upscaled x{NJT_UPSCALE} -> {out.size}")
    return out


def get_njt_png(af1):
    if NJT_OVERRIDE_PNG and os.path.exists(NJT_OVERRIDE_PNG):
        print(f"  [njt] using override image: {NJT_OVERRIDE_PNG}")
        return NJT_OVERRIDE_PNG
    p = _cache("panelC_njt.png")
    if os.path.exists(p):
        print(f"  [njt] using cached tree: {p}")
        return p
    if af1 is None:
        print("  [njt] NO API CLIENT — tree cannot be computed. "
              "Set COMPUTE_NJT=True (or USE_API=True) AND run on a node with internet.")
        return None
    print(f"  [njt] computing NJ tree from API (region={ANALYSIS_REGION}, n_snps={N_SNPS}) ...")
    try:
        import kaleido  # surface a missing-kaleido problem clearly, before the long compute
        print(f"  [njt] kaleido {getattr(kaleido, '__version__', '?')} available")
    except Exception as e:
        print(f"  [njt] kaleido NOT importable ({e}). Install it: pip install -U kaleido")
    try:
        # NOTE: plot_njt returns the plotly Figure ONLY when show=False (per the API
        # docs). Without show=False it displays and returns None, so write_image(None)
        # would fail -> placeholder. This is the fix for "Panel C not computing".
        fig = af1.plot_njt(
            region=ANALYSIS_REGION,
            n_snps=N_SNPS,
            sample_query=SAMPLE_QUERY,
            sample_sets=SAMPLE_SETS,
            category_orders={"country": ORDER},
            color="country",
            color_discrete_map=COUNTRY_COLORS,
            show=False,            # <-- required so the figure is returned
            title=False,           # clean panel (no plotly auto-title)
            leaf_legend=True,
        )
        if fig is None:
            raise RuntimeError("plot_njt returned None (show=False did not yield a figure)")
        fig.update_layout(legend_title_text="Country", margin=dict(l=10, r=10, t=10, b=10))
        fig.write_image(p, width=1100, height=1000, scale=3)   # needs kaleido
        print(f"  [ok] NJ tree computed and cached -> {p}")
        return p
    except Exception as e:
        print(f"  [warn] NJ tree export failed ({type(e).__name__}: {e}); will draw placeholder.")
        return None

def get_fst(af1):
    p = _cache("fst_paneld.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    if af1 is None:
        raise FileNotFoundError(p)
    df = af1.pairwise_average_fst(region=FST_REGION, sample_sets=SAMPLE_SETS,
                                  cohorts="country", sample_query=SAMPLE_QUERY, n_jack=N_JACK)
    df.to_csv(p, index=False)
    return df


# ======================================================================================
#  MAIN
# ======================================================================================
def main():
    af1 = None
    need_tree_api = (COMPUTE_NJT and NJT_OVERRIDE_PNG is None
                     and not os.path.exists(_cache("panelC_njt.png")))
    if USE_API or need_tree_api:
        import malariagen_data
        af1 = malariagen_data.Af1(site_filters_analysis=SITE_FILTERS)

    print("Gathering data ...")
    sites = get_sites(af1)
    pca_df, evr = get_pca(af1)
    njt_png = get_njt_png(af1)
    fst_df = get_fst(af1)

    pc1_var = f"{evr[0]*100:.2f}%" if evr is not None else PC1_VAR_FALLBACK
    pc2_var = f"{evr[1]*100:.2f}%" if evr is not None else PC2_VAR_FALLBACK

    print("Rendering aligned figure ...")
    tree_img = prepare_njt_image(njt_png) if njt_png else None
    if tree_img is None:
        print("  [warn] no real NJ tree found; panel C will show a note (see NJT_OVERRIDE_PNG).")
    final = compose_aligned(sites, pca_df, pc1_var, pc2_var, tree_img, fst_df)
    print("Saved ->", os.path.join(OUTDIR, final))

if __name__ == "__main__":
    main()