"""Separate regional Z-envelope extension from whole-envelope displacement.

Read-only source diagnostics; write new PNG/CSV/JSON outputs, never PDFs.
Boundaries inherit the explicitly case-adapted Z thresholds of the daily maps.
"""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import plot_era5_daily_vortex_boundaries as base
import numpy as np
from scipy.ndimage import label
import contourpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

OUT = ROOT / "results" / "ERA5_vortex_regional_extension"
SECTOR = (-130, -60)
PERIODS = [("Dec 28-30", 0, 3, "#444444", "--"),
           ("Jan 4-6", 7, 10, "#0072B2", "-"),
           ("Jan 9-11", 12, 15, "#D55E00", "-")]
PRESSURES = [500, 300, 100, 50, 10, 3, 1]
LINE_COLORS = {300: "#7251A1", 500: "#1B8D61"}


def main_component(mask, weights):
    """Largest area-connected component, with periodic longitude seams."""
    labs, n = label(mask, np.ones((3, 3), dtype=int))
    if n == 0:
        raise ValueError("No enclosed low-Z component")
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in range(mask.shape[0]):
        for adjacent in range(max(0, row - 1), min(mask.shape[0], row + 2)):
            a, b = labs[row, 0], labs[adjacent, -1]
            if a and b:
                parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(n + 1)])
    joined = roots[labs]
    areas = np.bincount(joined.ravel(), weights=weights.ravel(), minlength=n + 1)
    areas[0] = 0
    result = joined == np.argmax(areas)
    fraction = float(np.sum(weights[result]) / np.sum(weights[mask]))
    return result, fraction


def boundary_latitude(field, core, threshold, latitude):
    """Equatorward boundary in each meridian, with linear cell interpolation.

    Fractional values do not increase the native 2.5-degree data resolution.
    A longitude without a component intersection is retained as NaN.
    """
    answer = np.full(field.shape[1], np.nan)
    for j in range(field.shape[1]):
        inside = np.flatnonzero(core[:, j])
        if not len(inside):
            continue
        k = inside[0]
        if k == 0:
            raise ValueError("Boundary touches the southern data limit")
        low, high = field[k - 1, j], field[k, j]
        assert low >= threshold and high <= threshold
        fraction = (threshold - low) / (high - low)
        answer[j] = latitude[k - 1] + fraction * (latitude[k] - latitude[k - 1])
    return answer


def diagnostics(field, threshold, lat, lon, weights):
    smooth = base.smooth_map(field, .4)
    core, fraction = main_component(smooth <= threshold, weights)
    edge = boundary_latitude(smooth, core, threshold, lat)
    stats = base.region_stats(core, lat, lon, weights)
    sector = (lon >= SECTOR[0]) & (lon <= SECTOR[1])
    idx = np.flatnonzero(sector)[np.nanargmin(edge[sector])]
    return {"field": smooth, "mask": core, "edge": edge,
            "centroid_latitude": stats[0], "centroid_longitude": stats[1],
            "area_million_km2": stats[2], "main_component_fraction": fraction,
            "sector_mean_edge_latitude": float(np.nanmean(edge[sector])),
            "sector_min_edge_latitude": float(edge[idx]),
            "sector_min_longitude": float(lon[idx]),
            "edge_at_80W": float(edge[np.argmin(abs(lon + 80))])}


def analyze(data):
    lat, lon = data["lat"], data["lon"]
    weights = base.cell_weights(lat)
    result, rows, sensitivity = {}, [], []
    for pp in PRESSURES:
        j = int(np.flatnonzero(data["pressure"] == pp)[0])
        field = data["z"][:, j]
        before = field[:3].mean(0)
        ref = 50 if pp >= 300 else 60
        threshold = float(before[np.argmin(abs(lat - ref))].mean())
        baseline = diagnostics(before, threshold, lat, lon, weights)
        daily = [diagnostics(f, threshold, lat, lon, weights) for f in field]
        composites = [diagnostics(field[start:end].mean(0), threshold, lat, lon, weights)
                      for _, start, end, _, _ in PERIODS]
        result[pp] = {"threshold_gpm": threshold, "baseline": baseline,
                      "daily": daily, "composites": composites,
                      "height_km": float(np.median(field[:, lat >= 50]) / 1000)}
        for day, dd in zip(data["time"], daily):
            row = {"date_UTC": str(day), "pressure_hPa": pp, "threshold_gpm": threshold}
            for key in ["centroid_latitude", "centroid_longitude", "area_million_km2",
                        "main_component_fraction", "sector_mean_edge_latitude",
                        "sector_min_edge_latitude", "sector_min_longitude", "edge_at_80W"]:
                row[key] = dd[key]
            row["regional_mean_edge_change_deg"] = dd["sector_mean_edge_latitude"] - baseline["sector_mean_edge_latitude"]
            row["centroid_latitude_change_deg"] = dd["centroid_latitude"] - baseline["centroid_latitude"]
            rows.append(row)
        if pp in [300, 500]:
            for rr in [45, 50, 55]:
                thr = float(before[np.argmin(abs(lat - rr))].mean())
                pre = diagnostics(before, thr, lat, lon, weights)
                for name, start, end, _, _ in PERIODS[1:]:
                    post = diagnostics(field[start:end].mean(0), thr, lat, lon, weights)
                    sensitivity.append({"pressure_hPa": pp, "reference_latitude_N": rr,
                                        "period": name,
                                        "regional_mean_edge_change_deg": post["sector_mean_edge_latitude"] - pre["sector_mean_edge_latitude"],
                                        "regional_min_edge_change_deg": post["sector_min_edge_latitude"] - pre["sector_min_edge_latitude"],
                                        "edge_at_80W_change_deg": post["edge_at_80W"] - pre["edge_at_80W"]})
    return result, rows, sensitivity


def export(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=600, facecolor="white")
    with Image.open(path) as im:
        assert abs(im.width - 4724) <= 1
        assert abs(im.info["dpi"][0] - 600) < .1
        im.thumbnail((1700, 1700))
        im.convert("RGB").save(OUT / f"{name}_preview.jpg", quality=94)
    plt.close(fig)
    print(f"Saved {path.name}", flush=True)
    return str(path)


def plot_outer_boundary(ax, field, threshold, lat, lon, color, style):
    """Draw only the outer polar envelope, omitting internal high-Z holes."""
    cyc, ll = base.add_cyclic_point(field, coord=lon)
    segments = contourpy.contour_generator(x=ll, y=lat, z=cyc).lines(threshold)
    def projected_area(segment):
        if len(segment) < 3:
            return 0
        xy = base.PROJ.transform_points(base.PC, segment[:, 0], segment[:, 1])[:, :2]
        return abs(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1) - xy[:, 1]*np.roll(xy[:, 0], -1))) / 2
    outer = max(segments, key=projected_area)
    ax.plot(outer[:, 0], outer[:, 1], color=color, ls=style, lw=1.65,
            transform=base.PC, zorder=8)


def dates_on_axis(ax, dates):
    ticks = np.array(["2024-12-29", "2025-01-01", "2025-01-04", "2025-01-07", "2025-01-10"], dtype="datetime64[D]")
    ax.set_xticks(mdates.date2num(ticks))
    ax.set_xticklabels(["Dec\n29", "Jan\n01", "Jan\n04", "Jan\n07", "Jan\n10"], fontsize=12)
    ax.set_xlim(dates[0] - .5, dates[-1] + .5)
    ax.tick_params(direction="in", top=True, right=True, labelsize=12)
    ax.set_xlabel("UTC date | 2024-2025", fontsize=12)


def plot_main(data, result):
    lat, lon = data["lat"], data["lon"]
    dates = mdates.date2num(data["time"].astype("datetime64[D]"))
    fig = plt.figure(figsize=(20 / 2.54, 18 / 2.54))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.22, 1], left=.105, right=.97,
                           bottom=.11, top=.83, hspace=.45, wspace=.35)
    maps = [fig.add_subplot(grid[0, i], projection=base.PROJ) for i in range(2)]
    series = [fig.add_subplot(grid[1, i]) for i in range(2)]
    fig.suptitle("Regional extension and centroid evolution", fontsize=16, fontweight="normal", y=.989)
    legend = [Line2D([0], [0], color=color, ls=style, lw=1.7, label=name)
              for name, _, _, color, style in PERIODS]
    legend.append(Patch(facecolor="#D55E00", alpha=.18, label="Late gain"))
    fig.legend(handles=legend, ncol=4, loc="upper center", bbox_to_anchor=(.51, .949),
               frameon=False, fontsize=12, handlelength=1.6, columnspacing=1.0, handletextpad=.4)
    for i, (ax, pp) in enumerate(zip(maps, [300, 500])):
        base.map_frame(ax, i == 0)
        rr = result[pp]
        pre, late = rr["composites"][0], rr["composites"][2]
        north_america = np.broadcast_to((lon >= SECTOR[0]) & (lon <= SECTOR[1]), late["mask"].shape)
        gain = late["mask"] & ~pre["mask"] & north_america
        cyc, ll = base.add_cyclic_point(gain.astype(float), coord=lon)
        ax.contourf(ll, lat, cyc, levels=[.5, 1.5], colors=["#D55E00"], alpha=.18,
                    transform=base.PC, zorder=2)
        for dd, (_, _, _, color, style) in zip(rr["composites"], PERIODS):
            plot_outer_boundary(ax, dd["field"], rr["threshold_gpm"], lat, lon, color, style)
        ax.set_title(f"({chr(97+i)}) {pp} hPa (~{rr['height_km']:.0f} km)",
                     loc="left", y=1.16, pad=0, fontsize=13, fontweight="normal")
    for ax in series:
        ax.axvspan(dates[0] - .5, dates[2] + .5, color=".92", zorder=0)
        ax.axhline(0, color=".35", lw=.85, zorder=1)
        ax.grid(True, color=".75", lw=.6, alpha=.7, ls="--")
        dates_on_axis(ax, dates)
    for pp in [300, 500]:
        rr = result[pp]
        edge = np.array([d["sector_mean_edge_latitude"] for d in rr["daily"]]) - rr["baseline"]["sector_mean_edge_latitude"]
        center = np.array([d["centroid_latitude"] for d in rr["daily"]]) - rr["baseline"]["centroid_latitude"]
        series[0].plot(dates, edge, color=LINE_COLORS[pp], lw=2.1, marker="o", ms=3.6, label=f"{pp} hPa")
        series[1].plot(dates, center, color=LINE_COLORS[pp], lw=2.1, marker="o", ms=3.6)
    series[0].set_title("(c) Regional edge: 130-60°W", loc="left", fontsize=13, pad=10)
    series[1].set_title("(d) Envelope centroid", loc="left", fontsize=13, pad=10)
    series[0].set_ylabel(r"$\Delta\phi_{\rm edge}$ (°)", fontsize=13)
    series[1].set_ylabel(r"$\Delta\phi_{\rm centroid}$ (°)", fontsize=13)
    series[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=.95, fontsize=12, handlelength=1.4)
    series[0].set_ylim(-13, 4)
    series[1].set_ylim(-2, 9)
    return export(fig, "01_regional_extension_vs_centroid")


def plot_vertical(data, result):
    dates = mdates.date2num(data["time"].astype("datetime64[D]"))
    pp = np.array(sorted(PRESSURES))
    logp = np.log(pp)
    log_edges = np.r_[logp[0] - (logp[1]-logp[0])/2, (logp[:-1]+logp[1:])/2, logp[-1]+(logp[-1]-logp[-2])/2]
    pe = np.exp(log_edges)
    te = np.r_[dates - .5, dates[-1] + .5]
    edge, center = [], []
    for p in pp:
        rr = result[int(p)]
        edge.append([dd["sector_mean_edge_latitude"] - rr["baseline"]["sector_mean_edge_latitude"] for dd in rr["daily"]])
        center.append([dd["centroid_latitude"] - rr["baseline"]["centroid_latitude"] for dd in rr["daily"]])
    fig, axes = plt.subplots(1, 2, figsize=(20 / 2.54, 11.8 / 2.54), sharey=True)
    fig.subplots_adjust(left=.105, right=.94, top=.83, bottom=.22, wspace=.43)
    fig.suptitle("Pressure-time structure of Z-envelope changes", fontsize=16, fontweight="normal", y=.97)
    for ax, field, maximum, title in zip(axes, [np.array(edge), np.array(center)], [20, 10],
                                       ["(a) Regional edge change", "(b) Centroid latitude change"]):
        mesh = ax.pcolormesh(te, pe, np.clip(field, -maximum, maximum), cmap="RdBu_r",
                             norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum), shading="flat", rasterized=True)
        ax.set_yscale("log")
        ax.set_ylim(pe[-1], pe[0])
        ax.set_yticks(pp)
        ax.set_yticklabels([str(int(p)) for p in pp], fontsize=12)
        dates_on_axis(ax, dates)
        ax.set_title(title, loc="left", fontsize=13, pad=15)
        cb = fig.colorbar(mesh, ax=ax, fraction=.045, pad=.04, ticks=np.linspace(-maximum, maximum, 5))
        cb.ax.tick_params(direction="in", labelsize=10.5, length=3, pad=2)
        cb.ax.set_title("°", fontsize=12, pad=3)
    axes[0].set_ylabel("Pressure (hPa)", fontsize=13)
    axes[1].tick_params(labelleft=False)
    np.savez_compressed(OUT / "pressure_time_diagnostics.npz", time=data["time"], pressure=pp,
                        regional_mean_edge_change=np.array(edge), centroid_latitude_change=np.array(center))
    return export(fig, "02_pressure_time_edge_and_centroid")


def write_results(data, result, rows, sensitivity, pngs):
    for filename, records in [("daily_connected_envelope_diagnostics.csv", rows),
                              ("threshold_sensitivity.csv", sensitivity)]:
        with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {"source_cache": str(base.CACHE), "baseline": ["2024-12-28", "2024-12-30"],
               "domain": "20-90 N, global longitude", "regional_sector_degrees_E": list(SECTOR),
               "boundary_method": "Existing case-fixed Z contours; largest 8-neighbour connected component with periodic longitude. Smooth sigma 0.4 grid cells. Outer edge at each longitude interpolated linearly in latitude.",
               "native_grid_degrees": 2.5, "regional_metric": "Longitude mean of southernmost connected boundary, 130-60 W. Negative change means equatorward.",
               "centroid_metric": "Spherical area-weighted centroid, same connected component. Positive latitude change means closer to North Pole; not a rigid displacement or a formal Seviour moment diagnostic.",
               "reference_caution": "A short case baseline, not climatology; no significance or causality test. Z contour is an operational envelope, not a uniquely defined material PV edge.",
               "plot_windows": [p[0] for p in PERIODS], "source_maps_unchanged": True,
               "PDF_export": False, "PNG_files": pngs, "levels": {}}
    for p, rr in result.items():
        summary["levels"][str(p)] = {"height_km": rr["height_km"], "threshold_gpm": rr["threshold_gpm"],
            "baseline": {k: v for k, v in rr["baseline"].items() if k not in ["field", "mask", "edge"]},
            "periods": {name: {k: v for k, v in dd.items() if k not in ["field", "mask", "edge"]}
                        for dd, (name, _, _, _, _) in zip(rr["composites"], PERIODS)}}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for pp in [300, 500]:
        rr=result[pp]
        print(pp, "hPa baseline 80W", round(rr["baseline"]["edge_at_80W"],2),
              "Jan4-6 80W", round(rr["composites"][1]["edge_at_80W"],2),
              "Jan9-11 80W", round(rr["composites"][2]["edge_at_80W"],2), flush=True)
    assert all(r["main_component_fraction"] > .95 for r in rows if r["pressure_hPa"] in [300, 500])
    assert all(r["regional_mean_edge_change_deg"] < 0 for r in sensitivity)
    print("Connected-component and threshold-direction checks passed", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titleweight": "normal", "xtick.direction": "in", "ytick.direction": "in"})
    with np.load(base.CACHE, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    result, rows, sensitivity = analyze(data)
    paths = [plot_main(data, result), plot_vertical(data, result)]
    write_results(data, result, rows, sensitivity, paths)


if __name__ == "__main__":
    main()
