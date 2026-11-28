"""Separate 4x4 daily panels, preserving the supplied linked-figure arrow style.

MLS Jan3--18, 2025; no new temporal/spatial interpolation. This script does
not rerun or overwrite the original figure. PNG only, 20 cm wide, 600 dpi.
"""
from pathlib import Path
import sys
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
import plot_linked_reflection_mechanism_20261019 as reference
import plot_daily_W1_EP_directions_20261010 as original_arrows
from build_acp2018_event_figures_20261010 import extract, flux, background
from ep_flux_display_20261011 import (
    reduced_components, reference_arrow_cmap, draw_arrows, validate_text_frame,
    VERTICAL_FACTOR, ARROW_COLOURS,
)

DAYS = pd.date_range("2025-01-03", "2025-01-18")
SOURCE = ROOT / "W1_three_bands_Dec27_Jan20_20261016/MLS_W123_fullheight_Dec27_Jan20.npz"
OLD = ROOT / "acp2018_event_figures_20261010"
HASHES = {}


def register(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def compute():
    d = dict(np.load(register(SOURCE)))
    old = dict(np.load(register(OLD / "MLS_W123_fullheight_native.npz")))
    dates, old_dates = pd.DatetimeIndex(d["dates"]), pd.DatetimeIndex(old["dates"])
    assert (dates.get_indexer(DAYS) >= 0).all()
    assert np.array_equal(d["lat"], old["lat"])
    assert np.array_equal(d["pressure"], old["pressure"])
    # Frozen, identical original coefficients on Jan3--10, all LT/sector cases.
    overlap = DAYS[DAYS <= "2025-01-10"]
    overlap_checks = {}
    for key in ["cz", "ct", "z0", "t0", "counts"]:
        a = np.take(d[key], dates.get_indexer(overlap), axis=2)
        b = np.take(old[key], old_dates.get_indexer(overlap), axis=2)
        overlap_checks[key] = bool(np.array_equal(a, b, equal_nan=True))
        assert overlap_checks[key]
    frozen = dict(np.load(register(OLD / "daily_W1_EP_12days_fields.npz")))
    settings = json.loads(register(OLD / "daily_W1_EP_12days_settings.json").read_text(encoding="utf8"))
    lat, p = d["lat"], d["pressure"]
    levels = np.zeros(len(p), bool)
    levels[frozen["displayed_pressure_indices"]] = True
    levels[int(np.argmin(abs(p-.1)))] = False
    assert levels.sum() == 14
    assert np.allclose(np.diff(lat), 5)
    z, t, z0, t0 = extract(d, branch=-1, sector=0)
    rows = []
    for day in DAYS:
        i = dates.get_loc(day)
        q = flux(z[i,...,0], t[i,...,0], z0[i], t0[i], lat, p, wave=1, points=3)
        q["height"] = z0[i]/1000
        q["mer_reduced"], q["vert_reduced"], q["magnitude"] = reduced_components(q["P"], q["W"], lat)
        bg = background(z0[i], t0[i], lat, p, points=5)
        q["n2"], q["background_qy"], q["background_U"] = bg["n2"], bg["qy"], bg["U"]
        rows.append(q)
    return lat, p, levels, rows, settings, overlap_checks


def restyle_by_native_cell(ax, artist, boosted, lat, q, keep):
    """Equivalent reference styling without pairing two different edge masks.

    A boosted temporary arrow can be longer and fail the temporary edge test,
    even when the original-length, rotated arrow is valid. Address each original
    patch by its native cell instead; retain the original position and length.
    """
    trans, inv = ax.transData, ax.transData.inverted()
    factor = VERTICAL_FACTOR*reference.DIRECTION_VERTICAL_BOOST
    assert len(artist.fixed_head_patches) == int(keep.sum())
    for patch, (i,j) in zip(artist.fixed_head_patches,np.argwhere(keep)):
        start,end = [np.asarray(trans.transform(p),float) for p in patch._posA_posB]
        centre, length = (start+end)/2, np.hypot(*(end-start))
        u,v = q["mer_reduced"][i,j], q["vert_reduced"][i,j]
        direction = np.array([u,factor*v],float)
        direction /= np.hypot(*direction)
        down = v < 0 and q["height"][i,j] >= 48
        length *= 1.18 if down else 1
        patch.set_positions(tuple(inv.transform(centre-.5*length*direction)),
                            tuple(inv.transform(centre+.5*length*direction)))
        if down:
            patch.set_facecolor("#00e5ff")
            patch.set_edgecolor("#00e5ff")
            patch.set_linewidth(1.25)
            patch.set_zorder(7)
            patch.set_path_effects([pe.withStroke(linewidth=2.40,foreground="#002b49"),pe.Normal()])
        else:
            patch.set_linewidth(1.05)
            patch.set_zorder(5)
            patch.set_path_effects([pe.withStroke(linewidth=1.27,foreground="#1f272d"),pe.Normal()])
    for patch in boosted.fixed_head_patches:
        patch.remove()


def direction_error(ax, artist, q, keep):
    errors = []
    for patch, (i,j) in zip(artist.fixed_head_patches,np.argwhere(keep)):
        a,b = [np.asarray(ax.transData.transform(p),float) for p in patch._posA_posB]
        shown = (b-a)/np.hypot(*(b-a))
        expected = np.array([q["mer_reduced"][i,j],
            VERTICAL_FACTOR*reference.DIRECTION_VERTICAL_BOOST*q["vert_reduced"][i,j]])
        expected /= np.hypot(*expected)
        errors.append(float(np.max(abs(shown-expected))))
    assert max(errors) < 1e-10
    return max(errors)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for name in ["plot_linked_reflection_mechanism_20261019.py", "ep_flux_display_20261011.py",
                 "plot_daily_W1_EP_directions_20261010.py", "build_acp2018_event_figures_20261010.py"]:
        register(ROOT/name)
    image = OLD / "05_Linked_Downward_Wave_and_Reflection_Mechanism.png"
    if image.exists():
        register(image)
    lat, p, levels, daily, prior, overlap_checks = compute()
    assert prior["magnitude_limits"] == [0, 550.0]
    assert original_arrows.DENSE_LENGTH_BOUNDS == (prior["arrow_minimum_inches"], prior["arrow_maximum_inches"])
    assert ARROW_COLOURS == prior["arrow_colour_anchors"]
    norm = PowerNorm(gamma=.5, vmin=0, vmax=550)
    cmap = reference_arrow_cmap()
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12,
        "axes.titlesize":12, "axes.labelsize":12, "xtick.labelsize":12,
        "ytick.labelsize":12, "axes.linewidth":.85, "xtick.direction":"in",
        "ytick.direction":"in", "xtick.top":True, "ytick.right":True})
    figure_height_cm = 22.0
    fig = plt.figure(figsize=(20/2.54, figure_height_cm/2.54))
    # Retain panel widths; reduce panel heights and gaps for the compact revision.
    left, right = .057, .928
    panel_height_cm, row_gap_cm = 4.35, .65
    panel_height = panel_height_cm/figure_height_cm
    row_gap = row_gap_cm/figure_height_cm
    bottom = 1.25/figure_height_cm
    top = bottom + 4*panel_height + 3*row_gap
    fig.legend(handles=[
        Line2D([],[],color="#00e5ff", marker=">", markersize=7, lw=1.8,
               markeredgecolor="#002b49", label="Downward vector ($F_z<0$)"),
        Rectangle((0,0),1,1,fc="#7a7a7a",alpha=.45,ec="#404040",ls="--",label="$n^2<0$"),
        Line2D([],[],color="#8c1d40",lw=1.4,label="$\overline{q}_y=0$")],
        loc="upper center",bbox_to_anchor=(.50,.998),ncol=3,frameon=False,
        handlelength=1.5,handletextpad=.3,columnspacing=1.2,borderaxespad=.1)
    gs = fig.add_gridspec(4,4,left=left,right=right,bottom=bottom,top=top,
                          wspace=.16,hspace=row_gap/panel_height)
    keep_masks, counts, native_maxima, blue_counts = [], [], [], []
    boundary_fallbacks, direction_errors = [], []
    plotted = []
    for i, (day, q) in enumerate(zip(DAYS, daily)):
        row, col = divmod(i,4)
        ax = fig.add_subplot(gs[row,col])
        ax.set(xlim=(20,80),ylim=(20,90),xticks=[20,40,60,80],yticks=[20,40,50,60,80,90])
        ax.tick_params(labelleft=col == 0,labelbottom=row == 3,pad=2,length=3)
        h = np.nanmedian(q["height"],axis=0)
        ax.pcolormesh(lat,h,np.where(np.isfinite(q["n2"]) & (q["n2"]<0),1.,np.nan).T,
            cmap=ListedColormap(["#6e6e6e"]),vmin=0,vmax=1,shading="auto",alpha=.42,zorder=.5)
        if np.any(q["n2"]<0) and np.any(q["n2"]>0):
            ax.contour(lat,h,q["n2"].T,levels=[0],colors="#333333",linewidths=.7,
                       linestyles="--",alpha=.8,zorder=.7)
        if np.any(q["background_qy"]<0) and np.any(q["background_qy"]>0):
            ax.contour(lat,h,q["background_qy"].T,levels=[0],colors="#8c1d40",
                       linewidths=1.05,linestyles="-",alpha=.9,zorder=.8)
        for latitude in [40,60]:
            ax.axvline(latitude,color="#444444",ls="--",lw=.75,alpha=.65,zorder=1.2)
        ax.axhline(50,color="#222222",ls=":",lw=.85,zorder=1.2)
        artist, keep = draw_arrows(ax,lat,q["height"],q["P"],q["W"],levels,cmap,norm,
            readable=True,uniform_heads=True,length_bounds_inches=original_arrows.DENSE_LENGTH_BOUNDS,
            head_points=original_arrows.DENSE_HEAD_POINTS,shaft_points=original_arrows.DENSE_SHAFT_POINTS)
        boosted, keep_boosted = draw_arrows(ax,lat,q["height"],q["P"]*keep,
            q["W"]*reference.DIRECTION_VERTICAL_BOOST*keep,levels,cmap,norm,
            readable=True,uniform_heads=True,length_bounds_inches=original_arrows.DENSE_LENGTH_BOUNDS,
            head_points=original_arrows.DENSE_HEAD_POINTS,shaft_points=original_arrows.DENSE_SHAFT_POINTS)
        if np.array_equal(keep,keep_boosted):
            reference.rotate_and_style_arrows(ax,artist,boosted,lat,q["height"],q["P"],q["W"],keep,
                                              highlight_downward=True,shaft_pts=1.05)
        else:
            boundary_fallbacks.append(str(day.date()))
            restyle_by_native_cell(ax,artist,boosted,lat,q,keep)
        direction_errors.append(direction_error(ax,artist,q,keep))
        for region in reference.FOCUS_REGIONS:
            if pd.Timestamp(region["start"]) <= day <= pd.Timestamp(region["end"]):
                x0,x1 = region["latitude"]
                z0,z1 = region["height"]
                ax.add_patch(Rectangle((x0,z0),x1-x0,z1-z0,fill=False,
                    edgecolor=region["color"],linewidth=2,linestyle="-",zorder=8))
        if col == 0:
            ax.set_ylabel("approx. alt. (km)",labelpad=2)
        if row == 3:
            ax.set_xlabel("Latitude (°N)",labelpad=2)
        # Anchor the frame-end labels away from neighbouring panels.
        if row == 3:
            ax.get_xticklabels()[0].set_ha("left")
            ax.get_xticklabels()[-1].set_ha("right")
        ax.set_title(f"({chr(97+i)}) {day.day} Jan",loc="left",pad=5,weight="bold")
        keep_masks.append(keep)
        counts.append(int(keep.sum()))
        blue_counts.append(int((keep & (q["vert_reduced"]<0) & (q["height"]>=48)).sum()))
        native_maxima.append(float(q["magnitude"][keep].max()))
        for a,b in np.argwhere(keep):
            plotted.append(dict(date=str(day.date()),latitude=lat[a],pressure_hPa=p[b],
                height_km=q["height"][a,b],Fy_reduced_m2_s2=q["mer_reduced"][a,b],
                Fz_reduced_m2_s2=q["vert_reduced"][a,b],magnitude_m2_s2=q["magnitude"][a,b],
                n2_m_minus2=q["n2"][a,b],qy_m_minus1_s_minus1=q["background_qy"][a,b]))
    assert max(native_maxima) <= 550
    # Exactly the combined height of two panels and their intervening gap,
    # aligned with the middle two rows rather than the full four-row stack.
    colorbar_bottom = bottom+panel_height+row_gap
    colorbar_height = 2*panel_height+row_gap
    cax = fig.add_axes([.941,colorbar_bottom,.011,colorbar_height])
    cb = fig.colorbar(artist,cax=cax,ticks=[0,50,150,300,550])
    cb.formatter = FuncFormatter(lambda v,_:f"{v:g}")
    cb.update_ticks()
    cb.ax.tick_params(labelsize=10,direction="in",pad=2)
    cb.ax.set_title("$|\mathbf{F}|$\n(m² s⁻²)",fontsize=10,pad=5)
    validate_text_frame(fig)
    name = "02_daily_W1_EP_Jan03_Jan18_4x4_compact.png"
    fig.savefig(OUT/name,dpi=600)
    fig.savefig(OUT/"02_daily_W1_EP_Jan03_Jan18_4x4_compact_preview.png",dpi=150)
    plt.close(fig)
    with Image.open(OUT/name) as image:
        size = list(image.size)
        assert abs(image.width-4724) <= 1
    pd.DataFrame(plotted).to_csv(OUT/"02_plotted_native_vectors_compact.csv",index=False)
    np.savez_compressed(OUT/"02_daily_native_fields_compact.npz",dates=DAYS.values,latitude=lat,pressure_hPa=p,
        displayed_pressure_indices=np.flatnonzero(levels),keep=np.array(keep_masks),
        **{key:np.stack([q[key] for q in daily]) for key in ["P","W","Q","height","mer_reduced","vert_reduced","magnitude","n2","background_qy","background_U"]})
    checks = dict(figure=name,dates=DAYS.strftime("%Y-%m-%d").tolist(),layout=[4,4],
        pixels=size,width_cm=20,height_cm=figure_height_cm,dpi=600,PDF_generated=False,
        panel_height_cm=panel_height_cm,row_gap_cm=row_gap_cm,
        colourbar_height_cm=colorbar_height*figure_height_cm,
        colourbar_aligned_rows=[2,3],x_tick_labels_visible_rows=[4],
        original_Jan03_Jan10_coefficients_exact=overlap_checks,
        displayed_pressure_hPa=p[levels].tolist(),extra_01_hPa_arrow_row=False,
        latitude_spacing_degrees=5,arrow_counts=counts,highlighted_downward_counts=blue_counts,
        actual_visible_magnitude_maxima=native_maxima,magnitude_limits=[0,550],magnitude_PowerNorm_gamma=.5,
        native_cell_styling_fallback_on_temporary_edge_mask_mismatch=boundary_fallbacks,
        maximum_screen_direction_errors=direction_errors,
        colour_mapping="same original palette; highlighted downward arrows are cyan, not magnitude-coloured",
        direction_display_vertical_factor=VERTICAL_FACTOR*reference.DIRECTION_VERTICAL_BOOST,
        readable_length_bounds_inches=original_arrows.DENSE_LENGTH_BOUNDS,
        highlighted_downward_length_factor_above_48_km=1.18,
        native_layer_sampling="same 14 original pressure indices, excluding the extra 0.1 hPa arrow row",
        focus_rectangles="same original dates only: yellow Jan3--10, green Jan5--9; linewidth2",
        no_smoothing=True,no_missing_filling=True,no_interpolation_of_vectors=True,
        n2="daily total stationary QG refractive index, c=0; not measured vertical m2 or reflection surface",
        source_files_unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in HASHES.items()),
        input_sha256=HASHES)
    assert checks["source_files_unchanged"]
    previous_fields = OUT/"daily_native_fields.npz"
    if previous_fields.exists():
        previous = dict(np.load(previous_fields))
        checks["all_preexisting_diagnostic_fields_exact"] = all(
            np.array_equal(previous[key],np.stack([q[key] for q in daily]),equal_nan=True)
            for key in ["P","W","Q","height","mer_reduced","vert_reduced","magnitude","n2","background_qy","background_U"])
        checks["changed_visible_edge_cells"] = int(np.count_nonzero(previous["keep"] != np.asarray(keep_masks)))
        assert checks["all_preexisting_diagnostic_fields_exact"]
    (OUT/"02_drawing_and_source_checks_compact.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in checks.items() if k!="input_sha256"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
