"""Figure 3: thermal structure -> wind -> PV terms -> propagation conditions.

28 Dec 2024--11 Jan 2025; PRE=28--30 Dec; POST=6--9 Jan. Use the same
native MLS operators and daily EP fluxes as the existing figures. Retain actual
coverage, including unknown n2 near weak wind and missing backgrounds >70N.
Neither the overlay nor thermal-wind consistency uniquely attributes a storm.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch
from matplotlib.text import Text
import matplotlib.patheffects as pe
from PIL import Image

import plot_two_region_mechanism_profiles as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'mechanism_chain_barriers_Dec28_Jan11_20261022'
DAYS = pd.date_range('2024-12-28', '2025-01-11')
MID = DAYS + pd.Timedelta(hours=12)
PRE = DAYS <= '2024-12-30'
POST = (DAYS >= '2025-01-06') & (DAYS <= '2025-01-09')
WIDTH, HEIGHT = 20., 14.4
XS, WS = [1.10, 6.05, 11.15, 15.10], [3.45, 3.45, 3.05, 4.55]
YS, PANEL_H = [7.85, 1.65], 3.95
FONT = ['Arial', 'DejaVu Sans']
GOLD, GREEN, DARK = '#e2b000', '#00a65a', '#36464f'
BLUE, RED, PURPLE, MAGENTA, CYAN = '#0072b2', '#c53b2b', '#64359b', '#a23b54', '#00e5ff'
REGIONS = {
    'mid': dict(band=(40, 60), latlim=(35, 70), box=(40.3, 59.7, 52., 64.),
                start='2025-01-03', end='2025-01-10', color=GOLD, attribution_layer=(52, 64)),
    'high': dict(band=(60, 70), latlim=(50, 80), box=(60.3, 79.5, 52., 88.),
                 start='2025-01-05', end='2025-01-10', color=GREEN, attribution_layer=(65, 85)),
}
DAILY_ROWS, PROFILE_ROWS, COVERAGE, NEW_ROWS, SPATIAL_ROWS, ARROW_ROWS = [], [], [], [], [], []
METRICS, GEOMETRY, MANIFEST = [], {}, []
N2_MIN_POST_DAYS = 2
EVENTS_PATH = HERE / 'figure2_baseline_Dec28_Dec30_20261022/data_and_layout_checks.json'


def axes_at(fig, x, y, width, height):
    return fig.add_axes([x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT])


def load():
    base.DAYS, base.MID, base.PRE, base.POST = DAYS, MID, PRE, POST
    data = base.load()
    path = ROOT / 'W1_three_bands_Dec27_Jan20_20261016/MLS_W123_fullheight_Dec27_Jan20.npz'
    raw = dict(np.load(path))
    indices = pd.DatetimeIndex(raw['dates']).get_indexer(DAYS)
    _, _, height, _ = [a[indices] for a in base.extract(raw)]
    data['reference_height_km'] = height[PRE].mean(0) / 1000.
    np.testing.assert_allclose(data['qy'], data['beta'] + data['qmer'] + data['qvert'],
                               atol=1e-25, rtol=1e-12, equal_nan=True)
    return data


def band_mean(data, key, band):
    latitude = data['latitude']
    select = (latitude >= band[0]) & (latitude < band[1])
    a = data[key][:, select]
    complete = np.isfinite(a).all(axis=(0, 1))
    result = np.full((len(DAYS), a.shape[-1]), np.nan)
    weights = np.cos(np.deg2rad(latitude[select]))
    result[:, complete] = np.average(a[:, :, complete], axis=1, weights=weights)
    height = np.average(data['reference_height_km'][select], axis=0, weights=weights)
    for j in np.flatnonzero((height >= 40) & (height <= 100)):
        COVERAGE.append(dict(variable=key, band=f'{band[0]}-{band[1]}',
            pressure_hPa=float(data['pressure_hPa'][j]), reference_height_km=float(height[j]),
            requested_centres=';'.join(map(str, latitude[select])), complete_requested_band=bool(complete[j]),
            complete_centres=';'.join(map(str, latitude[select][np.isfinite(a[:, :, j]).all(0)]))))
    return result, height


def complete_mean(a, mask):
    return np.where(np.isfinite(a[mask]).all(0), a[mask].mean(0), np.nan)


def record_daily(region, panel, variable, values, height, pressure):
    for it, day in enumerate(DAYS):
        for j, p in enumerate(pressure):
            DAILY_ROWS.append(dict(region=region, panel=panel, variable=variable,
                date=str(day.date()), pressure_hPa=float(p), reference_height_km=float(height[j]),
                value=float(values[it, j])))


def axis_style(ax, top, labels=False, ylabel=False):
    ax.set_ylim(45, top)
    ax.set_yticks([z for z in [50, 60, 70, 80, 90] if z <= top])
    ax.tick_params(which='major', direction='in', top=True, right=True,
                   labelsize=12, labelleft=labels, pad=2, length=3)
    ax.tick_params(which='minor', direction='in', top=True, right=True, length=1.6)
    ax.set_ylabel('approx. alt. (km)' if ylabel else '', fontsize=12, labelpad=2)
    ax.set_facecolor('white')


def title(ax, letter, heading, subtitle):
    ax.set_title(f'({letter}) {heading}\n{subtitle}', loc='left',
                 fontsize=12, fontweight='normal', pad=5)


def calendar(ax, row, events):
    ax.set_xlim(DAYS[0], DAYS[-1] + pd.Timedelta(days=1))
    labels = pd.date_range('2024-12-29', '2025-01-10', freq='3D') + pd.Timedelta(hours=12)
    ax.set_xticks(labels)
    ax.set_xticks(MID, minor=True)
    ax.xaxis.set_major_formatter(md.DateFormatter('%b\n%d'))
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel('UTC date' if row == 1 else '', fontsize=12, labelpad=3)
    for key, color, style in [('storm_start', '#d62728', '--'), ('storm_end', '#666666', '-.')]:
        line = ax.axvline(pd.Timestamp(events[key]['time_UTC']), color=color, ls=style, lw=.75, zorder=9)
        line.set_gid(key)


def response_time_box(ax, conf):
    left = md.date2num(pd.Timestamp(conf['start']))
    right = md.date2num(pd.Timestamp(conf['end']) + pd.Timedelta(days=1))
    bottom, top = conf['box'][2:]
    # One plain outline: no white halo and no doubled line implying two bounds.
    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
        fill=False, ec=conf['color'], lw=1.35, ls=(0, (4, 2)), zorder=8))


def time_mesh(ax, values, height, norm, cmap):
    chosen = np.isfinite(height) & (height >= 40) & (height <= 100)
    colour = plt.get_cmap(cmap).copy()
    colour.set_bad('white')
    xedges = md.date2num(DAYS.append(pd.DatetimeIndex([DAYS[-1] + pd.Timedelta(days=1)])))
    return ax.pcolormesh(xedges, base.edges(height[chosen]),
        np.ma.masked_invalid(values[:, chosen].T), cmap=colour, norm=norm, shading='flat', zorder=0)


def zero_contour(ax, x, z, values, color, linestyle='-', linewidth=1.):
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() < 0 < finite.max():
        return ax.contour(x, z, np.ma.masked_invalid(values), levels=[0], colors=[color],
            linestyles=linestyle, linewidths=linewidth, corner_mask=False, zorder=6)
    return None


def gradient_panel(ax, data, conf, region, letter, row, events):
    gradient, height = band_mean(data, 'T_gradient', conf['band'])
    shear, _ = band_mean(data, 'Uz', conf['band'])
    delta_g = gradient - gradient[PRE].mean(0)
    delta_s = shear - shear[PRE].mean(0)
    top = float(height[np.isfinite(gradient).all(0)].max())
    image = time_mesh(ax, delta_g, height, TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12), 'RdBu_r')
    contour = ax.contour(md.date2num(MID), height, np.ma.masked_invalid(delta_s.T),
        levels=[-2, -1, 1, 2], colors='#303030', linewidths=.8,
        linestyles=['--', '--', '-', '-'], corner_mask=False, zorder=4)
    for item in ax.clabel(contour, fmt='%+g', fontsize=9.5, inline_spacing=2):
        item.set_gid('shear-label')
    axis_style(ax, top, labels=True, ylabel=True)
    calendar(ax, row, events)
    response_time_box(ax, conf)
    title(ax, letter, 'ΔTᵧ / Δshear', f'{conf["band"][0]}–{conf["band"][1]}°N')
    for key, value in [('delta_T_gradient', delta_g), ('delta_shear', delta_s)]:
        record_daily(region, letter, key, value, height, data['pressure_hPa'])
    predicted, _ = band_mean(data, 'thermal_shear', conf['band'])
    for name, select in [('Dec28-30', PRE), ('Jan6-9', POST)]:
        good = (height >= 45) & np.isfinite(shear[select]).all(0) & np.isfinite(predicted[select]).all(0)
        METRICS.append(dict(region=region, diagnostic='thermal_wind_consistency', period=name,
            RMSE_m_s_km=float(np.sqrt(np.mean((shear[select].mean(0)[good]
                                             - predicted[select].mean(0)[good]) ** 2))),
            interpretation='shared-input balance consistency, not independent storm causation'))
    return image, top


def n2_masks(data):
    n = data['n2']
    pre_known = np.isfinite(n[PRE]).all(0)
    pre_nonnegative = pre_known & (n[PRE] >= 0).all(0)
    daily_new = pre_nonnegative[None] & np.isfinite(n) & (n < 0)
    post_known = np.isfinite(n[POST]).all(0)
    post_negative = post_known & ((n[POST] < 0).sum(0) >= N2_MIN_POST_DAYS)
    new_post = pre_nonnegative & post_negative
    pre_negative = pre_known & ((n[PRE] < 0).sum(0) >= 2)
    return pre_nonnegative, daily_new, post_negative, new_post, pre_negative


def wind_panel(ax, data, conf, region, letter, row, top, events, masks):
    u, height = band_mean(data, 'U', conf['band'])
    qy, _ = band_mean(data, 'qy', conf['band'])
    image = time_mesh(ax, u, height, TwoSlopeNorm(vmin=-40, vcenter=0, vmax=80), 'PuOr_r')
    zero_contour(ax, md.date2num(MID), height, u.T, PURPLE, linewidth=1.1)
    zero_contour(ax, md.date2num(MID), height, qy.T, MAGENTA, '--', 1.)
    if region == 'mid':
        latitude = (data['latitude'] >= conf['band'][0]) & (data['latitude'] < conf['band'][1])
        daily_new = masks[1][:, latitude]
        counts = daily_new.sum(1)
        edges = base.edges(height)
        for it, j in zip(*np.where(counts > 0)):
            if conf['box'][2] <= height[j] <= conf['box'][3]:
                ax.add_patch(Rectangle((md.date2num(DAYS[it]), edges[j]), 1., edges[j + 1] - edges[j],
                    fill=False, ec='#252525', lw=.65, hatch='//', zorder=5))
        for it, day in enumerate(DAYS):
            for j in range(len(height)):
                NEW_ROWS.append(dict(date=str(day.date()), band='40-60', pressure_hPa=float(data['pressure_hPa'][j]),
                    reference_height_km=float(height[j]), new_negative_cells=int(counts[it, j]),
                    latitudes=';'.join(map(str, data['latitude'][latitude][daily_new[it, :, j]]))))
    axis_style(ax, top)
    calendar(ax, row, events)
    response_time_box(ax, conf)
    title(ax, letter, 'Zonal $U_g$', f'{conf["band"][0]}–{conf["band"][1]}°N')
    record_daily(region, letter, 'U_geostrophic', u, height, data['pressure_hPa'])
    record_daily(region, letter, 'qy_absolute', qy, height, data['pressure_hPa'])
    return image


def pv_panel(ax, data, conf, region, letter):
    arrays = {}
    for key in ['qy', 'qmer', 'qvert']:
        arrays[key], height = band_mean(data, key, conf['band'])
    common = np.logical_and.reduce([np.isfinite(a).all(0) for a in arrays.values()])
    delta = {key: (a[POST].mean(0) - a[PRE].mean(0)) * 1e11 for key, a in arrays.items()}
    np.testing.assert_allclose(delta['qy'], delta['qmer'] + delta['qvert'], atol=1e-12, equal_nan=True)
    top = float(height[common].max())
    selected = common & (height >= 39) & (height <= top)
    for key, color, linestyle in [('qy', DARK, '-'), ('qmer', BLUE, '--'), ('qvert', RED, '-.')]:
        ax.plot(delta[key][selected], height[selected], color=color, ls=linestyle, lw=1.5,
                marker='o' if key == 'qy' else None, ms=2.5, zorder=3)
        for j in np.flatnonzero(common & (height >= 40)):
            PROFILE_ROWS.append(dict(region=region, panel=letter, band=f'{conf["band"][0]}-{conf["band"][1]}',
                variable=key, pressure_hPa=float(data['pressure_hPa'][j]), reference_height_km=float(height[j]),
                pre_mean=float(arrays[key][PRE, j].mean()), post_mean=float(arrays[key][POST, j].mean()),
                delta_1e11=float(delta[key][j])))
    local_time = []
    for setting in data['LT']:
        temporary = dict(data, qy=setting['qy'])
        q, _ = band_mean(temporary, 'qy', conf['band'])
        local_time.append((q[POST].mean(0) - q[PRE].mean(0)) * 1e11)
    local_time = np.stack(local_time)
    valid = selected & np.isfinite(local_time).all(0)
    ax.fill_betweenx(height[valid], local_time[:, valid].min(0), local_time[:, valid].max(0),
                     color='.5', alpha=.18, lw=0, zorder=0)
    lower, upper = conf['attribution_layer']
    ax.axhspan(lower, min(upper, top), color=conf['color'], alpha=.12, zorder=-2)
    ax.axvline(0, color='.6', lw=.65, zorder=1)
    ax.axhline(50, color='.5', ls=':', lw=.6, zorder=1)
    ax.grid(True, axis='x', color='#bbbbbb', ls=':', lw=.5, alpha=.6)
    values = np.concatenate([delta[key][selected] for key in delta]
                            + [local_time[:, valid].ravel()])
    lo, hi = np.nanmin(values), np.nanmax(values)
    pad = .06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel('Δqᵧ (10⁻¹¹ m⁻¹ s⁻¹)', fontsize=11.5, labelpad=3)
    axis_style(ax, top, labels=True)
    title(ax, letter, 'Δqᵧ terms', f'{conf["band"][0]}–{conf["band"][1]}°N')
    region_levels = common & (height >= lower) & (height <= upper)
    for j in np.flatnonzero(region_levels):
        METRICS.append(dict(region=region, diagnostic='qy_contributions', height_km=float(height[j]),
            total_1e11=float(delta['qy'][j]), meridional_1e11=float(delta['qmer'][j]),
            vertical_1e11=float(delta['qvert'][j]),
            pre_qy_1e11=float(arrays['qy'][PRE, j].mean() * 1e11),
            post_qy_1e11=float(arrays['qy'][POST, j].mean() * 1e11)))


def native_polygons(data):
    height, lat = data['reference_height_km'], data['latitude']
    vertices, indices = [], []
    for il, la in enumerate(lat):
        for j in range(1, height.shape[1] - 1):
            h = height[il, j - 1:j + 2]
            if np.isfinite(h).all():
                vertices.append([(la - 2.5, (h[0] + h[1]) / 2), (la + 2.5, (h[0] + h[1]) / 2),
                                 (la + 2.5, (h[1] + h[2]) / 2), (la - 2.5, (h[1] + h[2]) / 2)])
                indices.append((il, j))
    return vertices, indices


def arrows(ax, data, p, w, region, panel):
    latitude, height = data['latitude'], data['reference_height_km']
    levels = np.zeros(len(data['pressure_hPa']), bool)
    levels[data['arrow_levels']] = True
    levels[np.argmin(abs(data['pressure_hPa'] - .1))] = False
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    height_in = ax.get_position().height * ax.figure.get_figheight()
    # Same effective direction stretch as final Figure 1: vertical x100 x4.
    # Fixed display length makes these direction-only arrows, not a flux scale.
    for il, la in enumerate(latitude):
        for j in np.flatnonzero(levels):
            z, v, s = height[il, j], p[il, j], w[il, j]
            if not (xlim[0] <= la <= xlim[1] and ylim[0] <= z <= ylim[1]
                    and np.isfinite(v) and np.isfinite(s) and np.hypot(v, s) > 0):
                continue
            norm = np.hypot(v, 400. * s)
            ux, uy = v / norm, 400. * s / norm
            length = .105 * (1.18 if s < 0 else 1.)
            dx, dz = length * ux * np.diff(xlim)[0] / width_in, length * uy * np.diff(ylim)[0] / height_in
            ends = [(la - dx / 2, z - dz / 2), (la + dx / 2, z + dz / 2)]
            if not all(xlim[0] + .1 < x < xlim[1] - .1 and ylim[0] + .1 < y < ylim[1] - .1 for x, y in ends):
                continue
            color = CYAN if s < 0 else DARK
            arrow = FancyArrowPatch(*ends, arrowstyle='-|>,head_length=.4,head_width=.2',
                mutation_scale=5.5 if s < 0 else 3.2,
                linewidth=1. if s < 0 else .7, facecolor=color,
                edgecolor=color, shrinkA=0, shrinkB=0,
                capstyle='round', joinstyle='miter', zorder=7, clip_on=True)
            if s < 0:
                arrow.set_path_effects([pe.withStroke(linewidth=1.6, foreground='#00354f'),
                                        pe.Normal()])
            ax.add_patch(arrow)
            vector = ax.transData.transform(ends[1]) - ax.transData.transform(ends[0])
            expected = np.array([ux, uy]) * length * ax.figure.dpi
            np.testing.assert_allclose(vector, expected, rtol=1e-10, atol=1e-10)
            ARROW_ROWS.append(dict(region=region, panel=panel, latitude=float(la),
                pressure_hPa=float(data['pressure_hPa'][j]), reference_height_km=float(z),
                P_period_mean=float(v), W_period_mean=float(s), downward=bool(s < 0),
                effective_vertical_display_factor=400, display_length_inches=length))


def map_panel(ax, data, conf, region, letter, masks):
    lat, height = data['latitude'], data['reference_height_km']
    vertices, indices = native_polygons(data)
    common = PRE | POST
    u = complete_mean(data['U'], POST)
    u[~np.isfinite(data['U'][common]).all(0)] = np.nan
    qy = complete_mean(data['qy'], POST)
    qy[~np.isfinite(data['qy'][common]).all(0)] = np.nan
    p, w = [complete_mean(data[key], POST) for key in ['P', 'W']]
    good_flux = np.isfinite(data['P'][common]).all(0) & np.isfinite(data['W'][common]).all(0)
    p[~good_flux] = np.nan
    w[~good_flux] = np.nan
    valid = [np.isfinite(u[index]) for index in indices]
    image = PolyCollection([v for v, good in zip(vertices, valid) if good],
        array=np.array([u[index] for index, good in zip(indices, valid) if good]),
        cmap='PuOr_r', norm=TwoSlopeNorm(vmin=-40, vcenter=0, vmax=80),
        edgecolors='none', zorder=0)
    ax.add_collection(image)
    post_negative, newly_negative, pre_negative = masks[2:]
    shaded = [v for v, index in zip(vertices, indices) if post_negative[index] and np.isfinite(u[index])]
    ax.add_collection(PolyCollection(shaded, facecolors='none', edgecolors='#656565',
                                     linewidths=0, hatch='///', zorder=2))
    x0, x1, z0, z1 = conf['box']
    framed = [v for v, index in zip(vertices, indices) if newly_negative[index] and np.isfinite(u[index])
              and x0 <= lat[index[0]] <= x1 and z0 <= height[index] <= z1]
    ax.add_collection(PolyCollection(framed, facecolors='none', edgecolors='#151515',
                                     linewidths=.95, zorder=4))
    xx = np.broadcast_to(lat[:, None], height.shape)
    # PRE edge is a reference outline, not an added PRE EP panel.
    pre_known = np.isfinite(data['n2'][PRE]).all(0)
    pre_field = np.where(pre_known, pre_negative.astype(float), np.nan)
    vals = pre_field[np.isfinite(pre_field)]
    if vals.size and vals.min() < .5 < vals.max():
        ax.contour(xx, height, np.ma.masked_invalid(pre_field), levels=[.5],
            colors=['#6f6f6f'], linestyles=':', linewidths=.85, corner_mask=False, zorder=3)
    zero_contour(ax, xx, height, u, PURPLE, linewidth=1.1)
    zero_contour(ax, xx, height, qy, MAGENTA, '--', 1.)
    ax.set_xlim(conf['latlim'])
    axis_style(ax, 90, labels=True)
    ax.set_xticks([40, 50, 60, 70] if region == 'mid' else [50, 60, 70, 80])
    ax.set_xlabel('Latitude (°N)', fontsize=12, labelpad=3)
    if region == 'high':
        ax.axvline(70, color='#858585', ls=':', lw=.75, zorder=3)
    arrows(ax, data, p, w, region, letter)
    x0, x1, z0, z1 = conf['box']
    box = Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False,
                    edgecolor=conf['color'], linewidth=2, zorder=9)
    ax.add_patch(box)
    assert box.get_path_effects() == []
    title(ax, letter, '$U_g$ & EP | Jan 6–9', f'{conf["latlim"][0]}–{conf["latlim"][1]}°N')
    for il, la in enumerate(lat):
        for j, pressure in enumerate(data['pressure_hPa']):
            if conf['latlim'][0] <= la <= conf['latlim'][1] and 40 <= height[il, j] <= 95:
                SPATIAL_ROWS.append(dict(region=region, panel=letter, latitude=float(la),
                    pressure_hPa=float(pressure), reference_height_km=float(height[il, j]),
                    U_pre=float(complete_mean(data['U'], PRE)[il, j]), U_post=float(u[il, j]),
                    qy_pre=float(complete_mean(data['qy'], PRE)[il, j]), qy_post=float(qy[il, j]),
                    P_post=float(p[il, j]), W_post=float(w[il, j]),
                    pre_all_n2_known=bool(np.isfinite(data['n2'][PRE, il, j]).all()),
                    pre_negative_days=int((data['n2'][PRE, il, j] < 0).sum()),
                    post_all_n2_known=bool(np.isfinite(data['n2'][POST, il, j]).all()),
                    post_negative_days=int((data['n2'][POST, il, j] < 0).sum()),
                    post_negative_at_least_half=bool(post_negative[il, j]),
                    newly_negative=bool(newly_negative[il, j])))
    return image


def colorbar(fig, image, x, y, heading, unit, ticks):
    cax = axes_at(fig, x, y, .15, PANEL_H)
    bar = fig.colorbar(image, cax=cax, ticks=ticks, extend='neither')
    bar.ax.set_title(heading, fontsize=10, fontweight='normal', pad=8)
    bar.set_label(unit, fontsize=9.5, labelpad=-8)
    bar.ax.tick_params(direction='in', labelsize=10, pad=2, length=2.5)
    return bar


def legend(fig):
    handles = [Patch(fc='none', ec='#656565', hatch='///', label='n² < 0 (≥2/4 d)'),
        Patch(fc='none', ec='#151515', lw=.95, label='New n² < 0 (focus)'),
        Line2D([], [], color=PURPLE, lw=1.1, label='$U_g = 0$'),
        Line2D([], [], color=MAGENTA, ls='--', label='$\overline{q}_y = 0$'),
        Line2D([], [], color=DARK, marker='o', ms=3, label='Total Δqᵧ'),
        Line2D([], [], color=BLUE, ls='--', label='Meridional Δqᵧ'),
        Line2D([], [], color=RED, ls='-.', label='Vertical Δqᵧ'),
        Patch(fc='.5', ec='none', alpha=.18, label='LT-branch range'),
        Line2D([], [], color='#6f6f6f', ls=':', lw=.85, label='Pre n² < 0 edge'),
        Line2D([], [], color=CYAN, marker='>', ms=6, markeredgecolor='#00354f', label='Downward EP')]
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.515, (HEIGHT - .16) / HEIGHT),
        ncol=5, frameon=False, fontsize=10, handlelength=1.35, handletextpad=.40,
        columnspacing=.85, labelspacing=.35, borderpad=0, borderaxespad=0)


def verify(fig, axes, bars, key):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Omit only automatic contour text outside its data rectangle.
    for ax in axes:
        for text in list(ax.texts):
            if text.get_gid() == 'shear-label':
                b = text.get_window_extent(renderer)
                if not ax.bbox.contains(b.x0, b.y0) or not ax.bbox.contains(b.x1, b.y1):
                    text.remove()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inactive = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    inactive.update([tick.label1, tick.label2])
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or text in inactive or not text.get_text().strip():
            continue
        assert not re.search(r'[\u3400-\u9fff]', text.get_text())
        b = text.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5:
            outside.append(text.get_text())
    assert not outside, ('Clipped text', outside)
    collisions = []
    for ax in axes:
        lo, hi = ax.get_xlim()
        ticks = [t.label1 for t in ax.xaxis.get_major_ticks() if lo <= t.get_loc() <= hi and t.label1.get_visible()]
        for a, b in zip(ticks[:-1], ticks[1:]):
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                collisions.append((ax.get_title(loc='left'), a.get_text(), b.get_text()))
    assert not collisions, ('Tick overlap', collisions)
    decorations = [key]
    for ax in axes:
        decorations.extend([ax._left_title, ax.xaxis.label, ax.yaxis.label])
    for bar in bars:
        decorations.extend([bar.ax.title, bar.ax.yaxis.label])
        unit = bar.ax.yaxis.label.get_window_extent(renderer)
        for tick in bar.ax.get_yticklabels():
            assert not unit.overlaps(tick.get_window_extent(renderer)), ('Colorbar unit overlap', tick.get_text())
    for i, a in enumerate(decorations):
        if isinstance(a, Text) and not a.get_text():
            continue
        for b in decorations[i + 1:]:
            if isinstance(b, Text) and not b.get_text():
                continue
            assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), ('Decoration overlap', str(a), str(b))
    for col in range(4):
        below = axes[4 + col]._left_title.get_window_extent(renderer)
        above = axes[col].xaxis.label.get_window_extent(renderer)
        if axes[col].get_xlabel():
            assert below.y1 < above.y0
        for tick in axes[col].get_xticklabels():
            if tick.get_visible():
                assert not below.overlaps(tick.get_window_extent(renderer)), 'Row heading/date overlap'
    GEOMETRY.update(clipped_text=outside, date_tick_overlap=collisions,
        legend_below_canvas_top=True, eight_panels=True, physical_width_cm=WIDTH,
        height_cm=HEIGHT, regular_titles=True)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': FONT, 'font.size': 12, 'axes.labelsize': 12,
        'axes.titlesize': 12, 'axes.titleweight': 'normal', 'axes.linewidth': .65,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'hatch.linewidth': .40,
        'savefig.facecolor': 'white'})
    events = json.loads(EVENTS_PATH.read_text(encoding='utf-8'))
    data = load()
    masks = n2_masks(data)
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54))
    axes = [axes_at(fig, x, y, w, PANEL_H) for y in YS for x, w in zip(XS, WS)]
    bars = []
    for row, (region, conf) in enumerate(REGIONS.items()):
        a, b, c, d = axes[4 * row:4 * row + 4]
        letters = [chr(97 + row * 4 + j) for j in range(4)]
        gi, top = gradient_panel(a, data, conf, region, letters[0], row, events)
        ui = wind_panel(b, data, conf, region, letters[1], row, top, events, masks)
        pv_panel(c, data, conf, region, letters[2])
        map_panel(d, data, conf, region, letters[3], masks)
        bars.extend([colorbar(fig, gi, 4.68, YS[row], 'ΔTᵧ', 'K / 1000 km', [-12, 0, 12]),
                     colorbar(fig, ui, 9.63, YS[row], '$U_g$', 'm s⁻¹', [-40, 0, 40, 80])])
        for letter, meaning in zip(letters, ['temperature_gradient_and_shear', 'background_wind_time_height',
                                            'qy_contributions', 'propagation_conditions_and_period_mean_EP']):
            MANIFEST.append(dict(panel=letter, region=region, diagnostic=meaning,
                background_band=f'{conf["band"][0]}-{conf["band"][1]}N',
                response_display_box=list(conf['box'])))
    key = legend(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    print('Layout bounds (cm):', json.dumps(dict(
        legend=(np.array(key.get_window_extent(renderer).bounds) * 2.54 / fig.dpi).tolist(),
        upper_titles=[(np.array(ax._left_title.get_window_extent(renderer).bounds) * 2.54 / fig.dpi).tolist()
                      for ax in axes[:4]])), flush=True)
    fig.savefig(OUT / 'layout_draft.png', dpi=140)
    verify(fig, axes, bars, key)
    png = OUT / 'Figure3_mechanism_chain_barriers_Dec28_Jan11.png'
    fig.savefig(png, dpi=600)
    fig.savefig(png.with_name(png.stem + '_preview.png'), dpi=160)
    plt.close(fig)
    with Image.open(png) as image:
        pixels = list(image.size)
    assert abs(pixels[0] - WIDTH / 2.54 * 600) < 2
    for name, rows in [('plotted_daily_diagnostics.csv', DAILY_ROWS), ('qy_contributions.csv', PROFILE_ROWS),
        ('native_coverage.csv', COVERAGE), ('new_n2_time_height_audit.csv', NEW_ROWS),
        ('spatial_pre_post_values.csv', SPATIAL_ROWS), ('displayed_EP_period_means.csv', ARROW_ROWS),
        ('diagnostic_metrics.csv', METRICS), ('panel_manifest.csv', MANIFEST)]:
        table = pd.DataFrame(rows)
        if name == 'native_coverage.csv':
            table = table.drop_duplicates()
        table.to_csv(OUT / name, index=False)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in base.HASHES.items())
    checks = dict(figure=str(png), pixels=pixels, width_cm=WIDTH, height_cm=HEIGHT, dpi=600,
        period_display=['2024-12-28', '2025-01-11'], baseline=['2024-12-28', '2024-12-30'],
        comparison=['2025-01-06', '2025-01-09'], baseline_days=3, comparison_days=4,
        period_comparison='arithmetic daily means, not period sums; unequal lengths explicit',
        daily_ticks='cell centres at12:00 UTC; labels every3 days beginningDec29',
        new_n2_definition='baseline all3 finite and all>=0; current daily finite<0; spatial POST all4 finite and at least2 negative',
        new_n2_display='black frames/hatching highlighted only in the prespecified response-box focus; full counts exported',
        pre_n2_edge='all3 days known, at least2 negative',
        n2_validity='N2>0 and abs(Ug)>=5m/s; easterly Ug<-5 is not automatically masked',
        latitude_coverage='full40-60N; complete high-latitude background60-70N only; EP to80N',
        map_high_latitude_guide_degN=70, unknown_background_is_not_propagating=True,
        Ug_operator='native5-point GPH geostrophic diagnostic; same denominator as n2',
        EP_operator='mean of daily W1 fluxes using3-point derivatives; not flux of mean wave coefficients',
        EP_display='direction-only fixed length; effective vertical x400 matches Figure1; cyan iff W<0',
        no_reflection_rays_or_arbitrary_turning_arrows=True, thermal_wind_is_shared_input_consistency=True,
        qy0_not_exact_vertical_turning_surface=True, zero_wind_may_absorb_waves=True,
        stationary_phase_speed_assumption_c_ms=0, unique_storm_causal_attribution=False,
        focus_boxes_are_descriptive_not_diagnostic_masks=True,
        no_duplicated_forcing_ozone_temperature_amplitude_or_net_flux_panels=True,
        geometry_checks=GEOMETRY, panel_manifest=MANIFEST, metrics=METRICS,
        storm_start=events['storm_start'], storm_end=events['storm_end'],
        input_sha256=base.HASHES, PDF_export=False)
    (OUT / 'data_and_layout_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figure=str(png), pixels=pixels, geometry=GEOMETRY,
                         metrics=METRICS, arrow_count=len(ARROW_ROWS)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
