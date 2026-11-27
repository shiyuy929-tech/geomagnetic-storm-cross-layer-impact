"""Case test of latitude-band GPH W1 as a vortex-geometry proxy.

PNG-only outputs. All source fields are read-only. Fourier reconstruction is a
kinematic diagnostic, not a dynamical intervention or causal attribution test.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_vortex_regional_extension as geometry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

OUT = ROOT / "results" / "ERA5_W1_vortex_hypothesis_test"
BANDS = [(60, 80), (40, 60), (50, 70), (30, 60), (55, 75), (45, 65)]
LEVELS = [1, 3, 10, 50, 100, 300, 500]
WINDOWS = {"Dec28-30": slice(0, 3), "Jan4-6": slice(7, 10),
           "Jan6-9": slice(9, 13), "Jan9-11": slice(12, 15)}


def wrap(degrees):
    return (np.asarray(degrees) + 180) % 360 - 180


def coefficient(field, lon, k):
    return np.mean(field * np.exp(-1j * k * np.deg2rad(lon)), axis=-1)


def band_weights(lat, bounds):
    edges = np.r_[lat[0] - (lat[1]-lat[0])/2,
                  (lat[:-1]+lat[1:])/2, lat[-1] + (lat[-1]-lat[-2])/2]
    low = np.maximum(edges[:-1], bounds[0])
    high = np.minimum(edges[1:], bounds[1])
    weights = np.maximum(0, np.sin(np.deg2rad(high))-np.sin(np.deg2rad(low)))
    return weights / weights.sum()


def correlation(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4 or np.std(np.asarray(a)[mask]) < 1e-10 or np.std(np.asarray(b)[mask]) < 1e-10:
        return None
    return float(np.corrcoef(np.asarray(a)[mask], np.asarray(b)[mask])[0, 1])


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_harmonic_convention(lon):
    # Exact synthetic harmonics check amplitude, trough direction and variance.
    sample = 10000 + 250*np.cos(np.deg2rad(lon-35)) + 70*np.cos(2*np.deg2rad(lon+20))
    c1 = coefficient(sample, lon, 1)
    assert np.isclose(2*abs(c1), 250, atol=1e-9)
    assert abs(float(wrap(180-np.rad2deg(np.angle(c1)))) + 145) < 1e-9
    rebuilt = sample.mean() + sum(2*np.real(coefficient(sample, lon, k)*np.exp(1j*k*np.deg2rad(lon))) for k in (1, 2))
    assert np.max(abs(rebuilt-sample)) < 1e-8


def analyze(data):
    lat, lon, dates = data["lat"], data["lon"], data["time"]
    check_harmonic_convention(lon)
    weights = geometry.base.cell_weights(lat)
    waves, geoms, wave_rows, geom_rows, sensitivity = {}, {}, [], [], []
    sector = (lon >= -130) & (lon <= -60)
    all_coeff = {k: coefficient(data["z"], lon, k) for k in (1, 2, 3)}
    for j, pp in enumerate(data["pressure"].astype(int)):
        z = data["z"][:, j]
        for band in BANDS:
            ww = band_weights(lat, band)
            profile = np.sum(z*ww[None, :, None], axis=1)
            cs = np.stack([np.sum(all_coeff[k][:, j]*ww, axis=-1) for k in (1, 2, 3)], axis=1)
            amps = 2*abs(cs)
            variance = np.var(profile, axis=-1)
            fractions = 2*abs(cs)**2 / variance[:, None]
            trough = wrap(180-np.rad2deg(np.angle(cs[:, 0])))
            mean_amp = np.sum(2*abs(all_coeff[1][:, j])*ww, axis=-1)
            depression = -np.mean(2*np.real(cs[:, 0, None]*np.exp(1j*np.deg2rad(lon[sector]))), axis=-1)
            waves[pp, band] = {"amplitude": amps, "fractions": fractions, "trough": trough,
                               "mean_lat_amplitude": mean_amp, "depression": depression}
            assert np.all(fractions.sum(1) <= 1+1e-9)
            assert np.all(amps[:, 0] <= mean_amp+1e-7)
            for t, date in enumerate(dates):
                row = {"date_UTC": str(date), "pressure_hPa": pp,
                       "band_lower_N": band[0], "band_upper_N": band[1]}
                for k in range(3):
                    row[f"W{k+1}_amplitude_gpm"] = amps[t, k]
                    row[f"W{k+1}_spatial_variance_fraction"] = fractions[t, k]
                row.update(W1_trough_longitude_degE=trough[t],
                           W1_mean_of_latitude_amplitudes_gpm=mean_amp[t],
                           W1_latitude_phase_coherence=amps[t, 0]/mean_amp[t],
                           W1_NAmerica_height_depression_gpm=depression[t],
                           W1_amplitude_change_gpm=amps[t, 0]-amps[:3, 0].mean(),
                           W1_amplitude_change_percent=100*(amps[t, 0]/amps[:3, 0].mean()-1))
                wave_rows.append(row)
        if pp not in LEVELS:
            continue
        ref = 50 if pp >= 300 else 60
        threshold = float(z[:3, np.argmin(abs(lat-ref)), :].mean())
        full = [geometry.diagnostics(f, threshold, lat, lon, weights) for f in z]
        edge = np.array([r["edge"] for r in full])
        assert np.all(np.isfinite(edge))
        edge_c1 = coefficient(edge, lon, 1)
        edge_a1 = 2*abs(edge_c1)
        edge_trough = wrap(180-np.rad2deg(np.angle(edge_c1)))
        center_lat = np.array([r["centroid_latitude"] for r in full])
        center_lon = np.array([r["centroid_longitude"] for r in full])
        sector_edge = edge[:, sector].mean(1)
        south = sector_edge[:3].mean()-sector_edge
        center_shift = center_lat[:3].mean()-center_lat
        geoms[pp] = {"full": full, "edge_W1_amplitude": edge_a1,
                     "edge_W1_trough": edge_trough, "centroid_lat": center_lat,
                     "centroid_lon": center_lon, "regional_southward_change": south,
                     "centroid_equatorward_change": center_shift,
                     "threshold": threshold,
                     "height_km": float(np.median(z[:, lat >= 50])/1000),
                     "reconstructed": {}}
        for kmax in (0, 1, 3, 8):
            rebuilt = np.broadcast_to(z.mean(-1, keepdims=True), z.shape).copy()
            for k in range(1, kmax+1):
                cc = coefficient(z, lon, k)
                rebuilt += 2*np.real(cc[..., None]*np.exp(1j*k*np.deg2rad(lon)))
            dd = [geometry.diagnostics(f, threshold, lat, lon, weights) for f in rebuilt]
            ee = np.array([r["sector_mean_edge_latitude"] for r in dd])
            geoms[pp]["reconstructed"][kmax] = ee[:3].mean()-ee
        for t, date in enumerate(dates):
            row = {"date_UTC": str(date), "pressure_hPa": pp,
                   "boundary_W1_amplitude_deg": edge_a1[t],
                   "boundary_W1_southern_lobe_longitude_degE": edge_trough[t],
                   "centroid_latitude_N": center_lat[t], "centroid_longitude_degE": center_lon[t],
                   "centroid_distance_from_pole_deg": 90-center_lat[t],
                   "centroid_equatorward_change_deg": center_shift[t],
                   "NAmerica_mean_edge_latitude_N": sector_edge[t],
                   "NAmerica_equatorward_change_deg": south[t],
                   "threshold_gpm": threshold}
            for k, arr in geoms[pp]["reconstructed"].items():
                row[f"Zbar_plus_W1_to_W{k}_NAmerica_equatorward_change_deg"] = arr[t]
            geom_rows.append(row)
        for rr in ([45, 50, 55] if pp >= 300 else [55, 60, 65]):
            thr = float(z[:3, np.argmin(abs(lat-rr)), :].mean())
            try:
                ds = [geometry.diagnostics(f, thr, lat, lon, weights) for f in z]
                ee = np.array([r["sector_mean_edge_latitude"] for r in ds])
                off = 90-np.array([r["centroid_latitude"] for r in ds])
                for band in ([(40, 60), (30, 60), (45, 65)] if pp >= 300 else [(60, 80), (50, 70), (55, 75)]):
                    wav = waves[pp, band]
                    aa = wav["amplitude"][:, 0]
                    sensitivity.append({"pressure_hPa": pp, "threshold_reference_latitude_N": rr,
                                        "band_lower_N": band[0], "band_upper_N": band[1],
                                        "r_W1_vs_centroid_distance": correlation(aa, off),
                                        "r_W1_vs_NAmerica_southward_extent": correlation(aa, -ee),
                                        "W1_Jan6_9_change_percent": 100*(aa[9:13].mean()/aa[:3].mean()-1),
                                        "NAmerica_Jan6_9_southward_change_deg": ee[:3].mean()-ee[9:13].mean(),
                                        "status": "valid"})
            except ValueError as exc:
                sensitivity.append({"pressure_hPa": pp, "threshold_reference_latitude_N": rr,
                                    "band_lower_N": None, "band_upper_N": None,
                                    "r_W1_vs_centroid_distance": None,
                                    "r_W1_vs_NAmerica_southward_extent": None,
                                    "W1_Jan6_9_change_percent": None,
                                    "NAmerica_Jan6_9_southward_change_deg": None,
                                    "status": str(exc)})
    return waves, geoms, wave_rows, geom_rows, sensitivity


def export(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=600, facecolor="white")
    with Image.open(path) as im:
        assert abs(im.width-4724) <= 1
        assert abs(im.info["dpi"][0]-600) < .1
        size = list(im.size)
        im.thumbnail((1700, 1900))
        im.convert("RGB").save(OUT/f"{name}_preview.jpg", quality=94)
    plt.close(fig)
    return {"file": str(path), "pixels": size, "dpi": 600}


def date_axis(ax, dates, labels=True):
    ticks = mdates.date2num(np.array(["2024-12-29", "2025-01-01", "2025-01-04", "2025-01-07", "2025-01-10"], dtype="datetime64[D]"))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Dec\n29", "Jan\n01", "Jan\n04", "Jan\n07", "Jan\n10"] if labels else [])
    ax.set_xlim(dates[0]-.5, dates[-1]+.5)
    ax.tick_params(direction="in", top=True, right=True, labelsize=12)
    if labels:
        ax.set_xlabel("UTC date | 2024-2025", fontsize=12)


def phase_line(ax, dates, phase, color, style="-", label=None):
    # Put the circular cut outside the phase paths in these two panels.
    phase = (phase+150) % 360-150
    arr = phase.copy()
    arr[np.r_[False, abs(np.diff(arr)) > 180]] = np.nan
    ax.plot(dates, arr, color=color, ls=style, lw=1.7, label=label)
    ax.scatter(dates, phase, color=color, s=13, zorder=3)


def plot_geometry(data, waves, geoms):
    dates = mdates.date2num(data["time"].astype("datetime64[D]"))
    fig, axes = plt.subplots(3, 2, figsize=(20/2.54, 22/2.54), sharey="row")
    fig.subplots_adjust(left=.115, right=.98, top=.875, bottom=.095, wspace=.17, hspace=.30)
    fig.suptitle("Wave-1 and vortex geometry", fontsize=16, fontweight="normal", y=.99)
    for col, pp in enumerate([1, 300]):
        band = (60, 80) if pp < 300 else (40, 60)
        ww, gg = waves[pp, band], geoms[pp]
        xpos = sum([axes[0, col].get_position().x0, axes[0, col].get_position().x1])/2
        fig.text(xpos, .936, f"{pp} hPa (~{gg['height_km']:.0f} km)", ha="center", fontsize=14)
        fig.text(xpos, .906, f"GPH band: {band[0]}-{band[1]}°N", ha="center", fontsize=12)
        a, b, c = axes[:, col]
        aa, ee = ww["amplitude"][:, 0], gg["edge_W1_amplitude"]
        a.plot(dates, aa/aa[:3].mean(), color="#0072B2", lw=2, marker="o", ms=3, label="GPH W1")
        a.plot(dates, ee/ee[:3].mean(), color="#D55E00", lw=1.8, ls="--", marker="s", ms=3, label="Boundary W1")
        a.axhline(1, color=".5", lw=.8, ls=":")
        phase_line(b, dates, ww["trough"], "#0072B2")
        phase_line(b, dates, gg["edge_W1_trough"], "#D55E00", "--")
        b.axhspan(-130, -60, color="#009E73", alpha=.10)
        b.text(.03, .11, "North America", transform=b.transAxes, fontsize=12, color="#227257")
        b.set_ylim(-150, 210)
        b.set_yticks([-90, 0, 90, 180], ["90°W", "0°", "90°E", "180°"])
        c.plot(dates, gg["regional_southward_change"], color="#6F3C97", lw=2, marker="o", ms=3, label="N. America edge")
        c.plot(dates, gg["centroid_equatorward_change"], color=".25", lw=1.8, ls="--", marker="s", ms=3, label="Centroid")
        c.axhline(0, color=".4", lw=.8)
        for row, ax in enumerate(axes[:, col]):
            ax.axvspan(dates[0]-.5, dates[2]+.5, color=".93", zorder=-2)
            ax.grid(True, color=".80", ls=":", lw=.65)
            date_axis(ax, dates, row == 2)
            if col:
                ax.tick_params(labelleft=False)
            ax.set_title(f"({chr(97+row*2+col)})", loc="left", fontsize=13, pad=4)
    axes[0, 0].set_ylabel("W1 amplitude / baseline", fontsize=12)
    axes[1, 0].set_ylabel("W1 southern-lobe longitude", fontsize=12)
    axes[2, 0].set_ylabel("Equatorward change (°)", fontsize=12)
    axes[0, 0].set_ylim(0, 2.4)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=12, handlelength=1.5)
    axes[2, 0].legend(loc="upper left", frameon=False, fontsize=12, handlelength=1.5)
    return export(fig, "01_W1_amplitude_phase_vs_vortex_geometry")


def plot_vertical(data, waves):
    dates = mdates.date2num(data["time"].astype("datetime64[D]"))
    p = data["pressure"]
    logp = np.log(p)
    edges = np.exp(np.r_[logp[0]-(logp[1]-logp[0])/2,
                         (logp[:-1]+logp[1:])/2, logp[-1]+(logp[-1]-logp[-2])/2])
    te = np.r_[dates-.5, dates[-1]+.5]
    fig, axes = plt.subplots(1, 2, figsize=(20/2.54, 11.5/2.54), sharey=True)
    fig.subplots_adjust(left=.105, right=.875, top=.81, bottom=.22, wspace=.16)
    fig.suptitle("Height-dependent W1 amplitude changes", fontsize=16, fontweight="normal", y=.97)
    for col, band in enumerate([(60, 80), (40, 60)]):
        arr = np.stack([waves[int(pp), band]["amplitude"][:, 0] for pp in p])
        change = 100*(arr/arr[:, :3].mean(1, keepdims=True)-1)
        mesh = axes[col].pcolormesh(te, edges, change, cmap="RdBu_r",
                                   norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=150), shading="flat")
        axes[col].set_yscale("log")
        axes[col].set_ylim(edges[-1], edges[0])
        axes[col].set_yticks(LEVELS, [str(x) for x in LEVELS])
        axes[col].set_title(f"({chr(97+col)}) {band[0]}-{band[1]}°N", loc="left", fontsize=13, pad=10)
        date_axis(axes[col], dates)
    axes[0].set_ylabel("Pressure (hPa)", fontsize=13)
    axes[1].tick_params(labelleft=False)
    cax = fig.add_axes([.90, .22, .020, .59])
    cb = fig.colorbar(mesh, cax=cax, extend="max", ticks=[-100, -50, 0, 50, 100, 150])
    cb.ax.set_title("%", fontsize=13, pad=5)
    cb.ax.tick_params(labelsize=11, direction="in")
    return export(fig, "02_W1_relative_change_pressure_time")


def make_summary(data, waves, geoms, sensitivity):
    output = {"baseline": ["2024-12-28", "2024-12-30"],
              "time_range": [str(data["time"][0]), str(data["time"][-1])],
              "n_days": len(data["time"]), "levels": {}}
    for pp in LEVELS:
        band = (40, 60) if pp >= 300 else (60, 80)
        ww, gg = waves[pp, band], geoms[pp]
        aa = ww["amplitude"][:, 0]
        phase_error = abs(wrap(ww["trough"]-gg["edge_W1_trough"]))
        item = {"latitude_band_N": list(band), "height_km": gg["height_km"],
                "W1_max_date_UTC": str(data["time"][np.argmax(aa)]),
                "NAmerica_extension_max_date_UTC": str(data["time"][np.argmax(gg["regional_southward_change"])]),
                "r_W1_vs_boundary_W1": correlation(aa, gg["edge_W1_amplitude"]),
                "r_W1_vs_centroid_distance": correlation(aa, 90-gg["centroid_lat"]),
                "r_W1_vs_NAmerica_southward_extent": correlation(aa, gg["regional_southward_change"]),
                "r_phase_aware_W1_vs_NAmerica_southward_extent": correlation(ww["depression"], gg["regional_southward_change"]),
                "median_GPH_vs_boundary_W1_longitude_error_deg": float(np.median(phase_error)),
                "windows": {}}
        for name, ss in WINDOWS.items():
            item["windows"][name] = {
                "W1_amplitude_gpm": float(aa[ss].mean()),
                "W1_change_percent": float(100*(aa[ss].mean()/aa[:3].mean()-1)),
                "mean_latitude_W1_amplitude_gpm": float(ww["mean_lat_amplitude"][ss].mean()),
                "W1_W2_W3_amplitude_gpm": ww["amplitude"][ss].mean(0).tolist(),
                "W1_W2_W3_spatial_variance_fraction": ww["fractions"][ss].mean(0).tolist(),
                "NAmerica_southward_change_deg": float(gg["regional_southward_change"][ss].mean()),
                "centroid_equatorward_change_deg": float(gg["centroid_equatorward_change"][ss].mean()),
                "mean_centroid_latitude_N": float(gg["centroid_lat"][ss].mean()),
                "W1_trough_circular_mean_longitude_degE": float(np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(ww["trough"][ss])))))),
                "boundary_W1_amplitude_deg": float(gg["edge_W1_amplitude"][ss].mean()),
                "reconstruction_NAmerica_southward_change_deg": {str(k): float(v[ss].mean()) for k, v in gg["reconstructed"].items()}}
        output["levels"][str(pp)] = item
    output["sensitivity_cases"] = len(sensitivity)
    output["invalid_sensitivity_cases"] = [r for r in sensitivity if r["status"] != "valid"]
    return output


def report(summary):
    lines = ["# W1与极涡南伸猜想的个例验证", "",
             "研究时段：2024-12-28至2025-01-11；基准：12月28–30日。反射期比较窗口为1月6–9日。",
             "", "## 主要数值", "",
             "|气压(hPa)|GPH纬度带|W1基准→1月6–9日(m)|变化|北美边界南伸(°)|质心向低纬变化(°)|r(W1,边界W1)|r(W1,北美南伸)|",
             "|---|---|---|---|---|---|---|---|"]
    for p in LEVELS:
        s=summary["levels"][str(p)]; pre=s["windows"]["Dec28-30"]; post=s["windows"]["Jan6-9"]
        lines.append(f"|{p}|{s['latitude_band_N'][0]}–{s['latitude_band_N'][1]}°N|{pre['W1_amplitude_gpm']:.1f}→{post['W1_amplitude_gpm']:.1f}|{post['W1_change_percent']:+.1f}%|{post['NAmerica_southward_change_deg']:+.2f}|{post['centroid_equatorward_change_deg']:+.2f}|{s['r_W1_vs_boundary_W1']:.2f}|{s['r_W1_vs_NAmerica_southward_extent']:.2f}|")
    lines += ["", "南伸和质心指标均以正值表示向低纬移动。北美指130°–60°W。相关系数只描述这15日的共同变化；没有把逐日值当独立样本做显著性推断。",
              "", "## 判断", "",
              "- 支持：指定纬度带的GPH W1可作为极涡边界一波不对称性的代理，本例幅值和相位均有较好对应。这是同一场的几何一致性检验。",
              "- 部分支持：1 hPa的W1增强并转向北美，伴随整体偏心和北美边界南伸；但该关系不能推广成所有层、所有经度都满足‘W1越大，局部南伸越强’。",
              "- 反例：300/500 hPa的40–60°N W1在1月6–9日明显减弱，全球一波偏心也减弱，但北美区域边界南伸。改变纬度带和边界阈值，仍保持W1减弱而区域南伸的方向。",
              "- 对流层谱结构改变：300 hPa W1空间方差份额由约85%降至9%，W2由约5%增至43%；500 hPa W1约84%降至7%，W2约6%增至48%。W2、W3及更高波数和纬向平均场应共同检查。空间方差份额不是时间方差，也不是能量沉积或因果贡献份额。",
              "- 40–60°N是纬度带，不是高度层。在50 km处计算的40–60°N W1仍是平流层附近的信号，不能由此替代300/500 hPa的对流层诊断。",
              "- 尚未检验：上下行波的相干叠加是否造成峰值，以及上层波动是否驱动下层形变。此处结果不能作为磁暴因果证明。",
              "", "## 方法与解释", "",
              "1. 对每日4个时次平均后的绝对位势高度Z作经向带平均，再进行纬向Fourier分解。纬度权重为与指定纬度带相交的网格面积，边界网格按覆盖比例计权。c_k=mean[Z exp(-ikλ)]，A_k=2|c_k|；W1槽经度为180°-arg(c_1)。同样计算先求各纬度振幅再平均的结果，用于检查纬度间相位抵消。",
              "2. 主要分析的是绝对环流的波结构（去掉纬向平均后k≥1不变），再计算振幅相对基准期的变化。没有将‘基准场相减后得到的扰动W1振幅’与‘实际场W1振幅的增减’混淆；二者不同。这里也没有多年气候态异常。",
              "3. 极涡边界沿用既有个例Z包络：基准期纬向平均Z在平流层60°N、对流层50°N的值，逐日使用同一阈值。最大面积连通区域采用周期经度；每条经线取该区域最南侧边界，线性插值得到φ_e(λ)。φ_e的一波振幅直接量化边界的一波不对称性。此Z包络不是唯一的物质PV边界。",
              "4. 分别比较GPH W1与边界W1振幅、质心离极距离，以及北美局部边界纬度。几何变化以逐日12月28–30日指标的均值为基准（与先平均场再提边界可有小差异）。全球偏心与特定经度南伸不是同一指标。",
              "5. Zbar+W1、Zbar+W1–3、Zbar+W1–8重建保留每一天随纬度变化的背景场和各波系数，再用相同阈值提取边界。每种重建以自己的基准日指标为参照。这是几何重建，不能解释为开关波强迫的因果试验；非线性边界位置也不能简单按波数相加。",
              "6. 1 hPa在此资料中的极区代表高度约44 km；ERA5此数据上限不能直接复核MLS约50 km的幅值峰顶。跨高度的GPH振幅还受背景高度梯度和波结构影响，不能用幅值最大高度直接判定极涡位移最大高度。",
              "7. 15日短样本、同一个GPH场的两种几何表达，不能独立证明‘上下行波共同作用造成极涡南伸’，也不能识别磁暴因果。应继续用独立的PV边界、波相位/三维波作用通量、EP通量散度及收支检查跨层连接，并通过集合对照区分上层强迫与已有对流层驱动。",
              "8. 敏感性文件改变平流层阈值参考纬度55/60/65°N、对流层45/50/55°N，以及相邻纬度带。网格为2.5°，插值产生小数不等于实际分辨率提高。",
              "", "## 图的读法", "",
              "图1上排：蓝线为GPH W1，橙线为直接提取的边界W1，均除以各自基准均值；中排：两种一波结构的南侧瓣经度，淡绿范围为北美；下排：紫线为北美边界向低纬变化，灰线为整个连通包络质心向低纬变化。灰色时间带表示基准期。",
              "图2：两个纬度带的W1振幅相对基准期变化，红色增强、蓝色减弱；每个气压层单独归一化，因此不能由颜色比较绝对振幅。稀疏气压层间未补造数据。",
              "", "## 参考文献", "",
              "- Chandra et al. (2026): https://doi.org/10.1038/s41612-026-01418-4 — 100 hPa、60–80°N的W1/W2振幅、相位和空间方差贡献率联合诊断。",
              "- Seviour et al. (2013): https://doi.org/10.1002/grl.50927 — 极涡几何与质心；本项目使用短期个例阈值，非原文DJFM气候态阈值。",
              "- Waugh et al. (2017): https://doi.org/10.1175/BAMS-D-15-00212.1 — 平流层极涡与对流层区域槽脊形变的区分。",
              "- Ning et al. (2026): https://doi.org/10.5194/wcd-7-277-2026 — 非对称强迫相位影响极涡和对流层响应的对照模拟。"]
    (OUT/"README_猜想验证与方法.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = geometry.base.CACHE
    source_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    with np.load(cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titleweight": "normal", "xtick.direction": "in", "ytick.direction": "in"})
    waves, geoms, wave_rows, geom_rows, sensitivity = analyze(data)
    write_csv("daily_GPH_wave_diagnostics.csv", wave_rows)
    write_csv("daily_boundary_and_centroid_validation.csv", geom_rows)
    write_csv("latitude_band_and_boundary_sensitivity.csv", sensitivity)
    summary = make_summary(data, waves, geoms, sensitivity)
    summary["figures"] = [plot_geometry(data, waves, geoms), plot_vertical(data, waves)]
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == source_hash
    summary["source"] = str(cache)
    summary["source_sha256"] = source_hash
    summary["checks"] = {"source_unchanged": True, "synthetic_phase_amplitude": True,
                          "variance_fractions_valid": True, "all_boundary_longitudes_present": True,
                          "output_width_cm": 20, "output_dpi": 600, "PDF_export": False}
    (OUT/"validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report(summary)
    for pp in LEVELS:
        s=summary["levels"][str(pp)]; post=s["windows"]["Jan6-9"]
        print(pp, "r_edgeW1=", round(s["r_W1_vs_boundary_W1"], 3),
              "r_centroid=", round(s["r_W1_vs_centroid_distance"], 3),
              "r_NA=", round(s["r_W1_vs_NAmerica_southward_extent"], 3),
              "amp_change%=", round(post["W1_change_percent"], 1),
              "NA_south_deg=", round(post["NAmerica_southward_change_deg"], 2), flush=True)
    print("Invalid sensitivity cases:", summary["invalid_sensitivity_cases"], flush=True)
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
