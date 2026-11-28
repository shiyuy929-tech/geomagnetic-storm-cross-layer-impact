"""Extend Figure 2 to 28 Dec 2024--11 Jan 2025; retain the Dec 27--30 baseline.

Existing dates retain their original values and masks. The extra MLS day uses
the original joint-profile QC, night selection and weighted W1--W3 fit. The
extra MEPED day uses the ORIGINAL fixed latitude/MLT support, not a reselected
support that would change the existing series. No smoothing or gap filling.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import re
import sys

import h5py
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from matplotlib.text import Text
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = HERE / 'five_row_ozone_BrBG_20261022'
OUT = HERE / 'figure2_Dec28_Jan11_20261022'
START, END = pd.Timestamp('2024-12-28'), pd.Timestamp('2025-01-11')
STOP = END + pd.Timedelta(days=1)
DAYS = pd.date_range(START, END)
MID = DAYS + pd.Timedelta(hours=12)
WIDTH_CM, HEIGHT_CM = 20., 19.75
COLORS = dict(Kp='#8b75ae', Dst='#c74d3d', AE='#ae7832', PCN='#287bb5',
              **{'40-60': '#2166ac', '60-80': '#00965e'})
HASHES, DOWNLOADS, VALIDATION, COVERAGE = {}, [], {}, []


def source(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mls_day(day, helper):
    suffix = day.strftime('%Yd%j')
    paths = []
    for product in ['O3', 'Temperature', 'GPH']:
        if product == 'O3':
            path = ROOT / f'event_evidence_20261002/data/O3/MLS-Aura_L2GP-O3_v06-01-c01_{suffix}.he5'
        elif day <= pd.Timestamp('2025-01-10'):
            path = helper.base.mls_files(product)[day]
        else:
            path = ROOT / ('W1_three_bands_Dec27_Jan20_20261016/raw_MLS_20250111_20250120/'
                           f'MLS-Aura_L2GP-{product}_v06-01-c01_{suffix}.he5')
        paths.append(source(path))
    o, t, z = [helper.read_mls(path, name)
               for path, name in zip(paths, ['O3', 'Temperature', 'GPH'])]
    for key in ['Pressure', 'Latitude', 'Longitude', 'Time', 'LocalSolarTime']:
        np.testing.assert_array_equal(o[key], t[key])
        np.testing.assert_array_equal(o[key], z[key])
    qc = ((o['Status'] % 2 == 0) & (o['Quality'] > 1) & (o['Convergence'] < 1.03)
          & (t['Status'] % 2 == 0) & (t['Quality'] > .2) & (t['Convergence'] < 1.03)
          & (z['Status'] % 2 == 0) & (z['Quality'] > .2) & (z['Convergence'] < 1.03))
    rows = []
    for j, pressure in enumerate(o['Pressure']):
        if not .001 <= pressure <= 1.:
            continue
        oo, tt, zz = o['L2gpValue'][:, j] * 1e6, t['L2gpValue'][:, j], z['L2gpValue'][:, j] / 1000.
        good = (qc & (o['L2gpPrecision'][:, j] > 0) & (t['L2gpPrecision'][:, j] > 0)
                & (z['L2gpPrecision'][:, j] > 0) & np.isfinite(oo) & (abs(oo) < 100)
                & (tt > 100) & (tt < 350) & (zz > 0) & (zz < 120)
                & (o['LocalSolarTime'] < 8) & (o['SolarZenithAngle'] > 95))
        frame = pd.DataFrame(dict(lat=o['Latitude'][good], lon=o['Longitude'][good],
            lst=o['LocalSolarTime'][good], ozone_ppmv=oo[good], temperature_K=tt[good],
            height_km=zz[good], normalized_ozone_ppmv=oo[good] / helper.factor(tt[good])))
        for lo, hi in [(40, 60), (60, 80)]:
            result = helper.fit_frame(frame, ['ozone_ppmv', 'temperature_K', 'height_km',
                'normalized_ozone_ppmv'], lo, hi, 'reference')
            rows.append(dict(date=str(day.date()), band=f'{lo}-{hi}',
                             pressure_hPa=float(pressure), **result))
    return pd.DataFrame(rows)


def chemistry():
    sys.path.insert(0, str(ROOT))
    helper = load_module('figure2_original_chemistry',
                         source(ROOT / 'check_2025_ozone_thermal_circulation_20261012.py'))
    raw = pd.read_csv(source(ROOT / 'storm_two_path_test_20261020/MLS_night_O3_T_reference.csv'))
    raw = raw[(raw['mode'] == 'night') & (raw['case'] == 'reference')
              & raw.band.isin(['40-60', '60-80'])]
    # Recompute a known day first: the extension must reproduce the old method.
    known = mls_day(pd.Timestamp('2025-01-10'), helper)
    old = raw[raw.date == '2025-01-10'].copy()
    known['pressure_key'] = known.pressure_hPa.round(9)
    old['pressure_key'] = old.pressure_hPa.round(9)
    merged = known.merge(old, on=['date', 'band', 'pressure_key'], suffixes=('_new', '_old'),
                         validate='one_to_one')
    assert len(merged) == len(old) == 26
    for field in ['ozone_ppmv', 'temperature_K', 'height_km']:
        np.testing.assert_allclose(merged[field + '_new'], merged[field + '_old'],
                                   rtol=1e-11, atol=1e-11, equal_nan=True)
    VALIDATION['Jan10_MLS_recomputation_matches_original'] = True
    extra = mls_day(END, helper)
    extra.to_csv(OUT / 'MLS_Jan11_matched_fits.csv', index=False)
    before = pd.read_csv(source(BASE / 'night_ozone_temperature_native_daily.csv'))
    rows = before[before.date.ge(str(START.date()))].copy().to_dict('records')
    for band, group in raw.groupby('band'):
        base = group[group.date.between('2024-12-27', '2024-12-30')]
        for pressure, original in base.groupby('pressure_hPa'):
            assert len(original) == 4
            fit = extra[(extra.band == band) & np.isclose(extra.pressure_hPa, pressure)].iloc[0]
            ozone_base = original.ozone_ppmv.mean()
            ozone_good = (original.ozone_ppmv > .05).all()
            height = before[(before.band == band) & np.isclose(before.pressure_hPa, pressure)].reference_height_km.iloc[0]
            rows.append(dict(date=str(END.date()), band=band, pressure_hPa=pressure,
                reference_height_km=height,
                dO3=100 * (fit.ozone_ppmv / ozone_base - 1) if ozone_good else np.nan,
                dT=fit.temperature_K - original.temperature_K.mean()))
    result = pd.DataFrame(rows).sort_values(['date', 'band', 'pressure_hPa'], ascending=[True, True, False])
    result.to_csv(OUT / 'night_ozone_temperature_native_daily.csv', index=False)
    common = result.merge(before, on=['date', 'band', 'pressure_hPa'], suffixes=('_new', '_old'))
    assert len(common) == 14 * 2 * 13
    for column in ['dO3', 'dT', 'reference_height_km']:
        np.testing.assert_array_equal(common[column + '_new'], common[column + '_old'])
    VALIDATION['overlap_MLS_values_heights_and_masks_unchanged'] = True
    return result


def download_meped(satellite):
    name = f'poes_{satellite}_20250111_proc.nc'
    path = OUT / 'raw_MEPED' / name
    url = ('https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/'
           f'l1b/v01r00/2025/metop{satellite[1:]}/{name}')
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        response = requests.get(url, timeout=(15, 45))
        response.raise_for_status()
        assert response.content[:3] == b'\x89HD', 'Expected the same HDF5/NetCDF4 product as the original files'
        # Binary scientific-data download; never replace an original input.
        path.write_bytes(response.content)
    with h5py.File(path) as handle:
        assert 'mep_ele_tel0_flux_e1' in handle and 'mep_ele_tel0_flux_e3' in handle
    DOWNLOADS.append(dict(url=url, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print('MEPED available:', satellite, path.stat().st_size, flush=True)
    return path


def meped_day(day, paths, support):
    rows = []
    for path in paths:
        satellite = path.name.split('_')[1]
        with h5py.File(source(path)) as h:
            time = h['time'][:].astype('int64')
            lat, lon, aacgm, mlt = [h[key][:].astype(float) for key in
                                    ['geod_lat_foot', 'geod_lon_foot', 'aacgm_lat_foot', 'MLT']]
            b_sat, b_foot = h['Btot_sat'][:], h['Btot_foot'][:]
            with np.errstate(invalid='ignore'):
                cone = np.degrees(np.arcsin(np.sqrt(np.where(
                    (b_sat > 0) & (b_foot > b_sat), b_sat / b_foot, np.nan))))
            alpha = h['meped_alpha_0_sat'][:]
            fold = np.minimum(alpha, 180 - alpha)
            lval = h['L_IGRF'][:]
            eligible = (np.isfinite(lat) & np.isfinite(aacgm) & (abs(aacgm) <= 90)
                & np.isfinite(lon) & (lon >= 0) & (lon < 360) & (mlt >= 0) & (mlt < 24)
                & (h['mep_IFC_on'][:] == 0) & (fold >= 0) & (fold + 15 < cone)
                & (lval >= 2) & (lval <= 10))
            for band in ['40-60', '60-80']:
                lo, hi = map(int, band.split('-'))
                for channel in ['e1', 'e3']:
                    value = h[f'mep_ele_tel0_flux_{channel}'][:].astype(float)
                    good = eligible & (lat >= lo) & (lat < hi) & np.isfinite(value) & (value >= 0)
                    x = pd.DataFrame(dict(block=time[good] // 6_000_000,
                        latbin=np.floor(lat[good] / 5).astype(int),
                        mltbin=np.floor(mlt[good] / 2).astype(int), value=value[good]))
                    blocks = x.groupby(['block', 'latbin', 'mltbin']).value.agg(['mean', 'size'])
                    cell = blocks[blocks['size'] >= 2]['mean'].groupby(['latbin', 'mltbin']).mean()
                    kept = support[(support.satellite == satellite) & (support.telescope == 0)
                        & (support.band == band) & (support.channel == channel)]
                    keys = list(zip(kept.latbin, kept.mltbin))
                    values = cell.reindex(pd.MultiIndex.from_tuples(keys, names=['latbin', 'mltbin']))
                    complete = bool(keys) and values.notna().all()
                    weights = np.array([np.sin(np.deg2rad(k[0] * 5 + 5))
                                        - np.sin(np.deg2rad(k[0] * 5)) for k in keys])
                    flux = float(np.average(values, weights=weights)) if complete else np.nan
                    rows.append(dict(date=str(day.date()), satellite=satellite, band=band,
                        channel=channel, telescope=0, flux=flux, original_fixed_cells=len(keys),
                        observed_original_cells=int(values.notna().sum()), complete=bool(complete)))
    table = pd.DataFrame(rows)
    if day == END:
        table.to_csv(OUT / 'MEPED_Jan11_original_support_audit.csv', index=False)
    paired = []
    for (band, channel), group in table.groupby(['band', 'channel']):
        complete = set(group.satellite) == {'m01', 'm03'} and group.complete.all()
        paired.append(dict(region='shared', panel='particle', variable='MEPED_' + channel,
            date=str(day.date()), band=band, value=group.flux.mean() if complete else np.nan))
    return pd.DataFrame(paired)


def particles():
    with ThreadPoolExecutor(max_workers=2) as pool:
        new_paths = list(pool.map(download_meped, ['m01', 'm03']))
    support = pd.read_csv(source(ROOT / 'two_region_diagnostics_20261021/forcing/fixed_support_cells.csv'))
    original = pd.read_csv(source(BASE / 'MEPED_plotted_values.csv'))
    paths = [ROOT / f'event_evidence_20261002/data/MEPED/poes_{sat}_20250110_proc.nc'
             for sat in ['m01', 'm03']]
    check = meped_day(pd.Timestamp('2025-01-10'), paths, support)
    old = original[original.date == '2025-01-10']
    test = check.merge(old, on=['date', 'band', 'variable'], suffixes=('_new', '_old'))
    np.testing.assert_allclose(test.value_new, test.value_old, rtol=1e-12, atol=1e-12)
    assert len(test) == 4
    VALIDATION['Jan10_MEPED_recomputation_matches_original'] = True
    result = pd.concat([original[original.date.ge(str(START.date()))],
                        meped_day(END, new_paths, support)], ignore_index=True)
    result.to_csv(OUT / 'MEPED_plotted_values.csv', index=False)
    VALIDATION['MEPED_original_fixed_support_preserved'] = True
    return result


def indices():
    legacy = ROOT.parent / 'era5/data'
    path = source(ROOT / 'PCN_AE_three_newyear_cases_20261017/AE_PCN_2025_Dec27_Jan20_1min.csv')
    minute = pd.read_csv(path)
    minute['UTC_time'] = pd.to_datetime(minute.UTC_time, utc=True).dt.tz_localize(None)
    hourly = minute.set_index('UTC_time')[['aeValue', 'PCN']].resample('1h').mean().loc[START:STOP - pd.Timedelta(seconds=1)]
    assert len(hourly) == 360 and hourly.notna().all().all()
    hourly.to_csv(OUT / 'AE_PCN_hourly.csv')
    payload = json.loads(source(legacy / 'GFZ_Kp_20241225_20250115.json').read_text())
    kp = pd.DataFrame(dict(time=pd.to_datetime(payload['datetime'], utc=True).tz_localize(None),
                           Kp=payload['Kp'], status=payload['status']))
    full = pd.read_csv(source(legacy / 'NASA_OMNI_hourly_Kp_Dst_2010_2025.csv'),
                       names=['time', 'Kp_x10', 'Dst_nT'])
    dst = full[['time', 'Dst_nT']].copy()
    dst['time'] = pd.to_datetime(dst.time)
    if dst.time.dt.tz is not None:
        dst['time'] = dst.time.dt.tz_localize(None)
    frames = []
    for frame, name in [(kp, 'Kp_3hourly_GFZ.csv'), (dst, 'Dst_hourly_OMNI.csv')]:
        frame = frame[(frame.time >= START) & (frame.time < STOP)].copy()
        old = pd.read_csv(source(BASE / name), parse_dates=['time'])
        common = frame.merge(old, on='time', suffixes=('_new', '_old'))
        column = 'Kp' if name.startswith('Kp') else 'Dst_nT'
        np.testing.assert_array_equal(common[column + '_new'], common[column + '_old'])
        frame.to_csv(OUT / name, index=False)
        frames.append(frame)
    old = pd.read_csv(source(BASE / 'AE_PCN_hourly.csv'), index_col=0, parse_dates=True)
    shared = hourly.index.intersection(old.index)
    np.testing.assert_allclose(hourly.loc[shared], old.loc[shared], rtol=1e-12, atol=1e-12)
    VALIDATION['overlap_indices_unchanged'] = True
    return hourly, *frames


def edges(values):
    values = np.asarray(values, float)
    mid = (values[:-1] + values[1:]) / 2
    return np.r_[2 * values[0] - mid[0], mid, 2 * values[-1] - mid[-1]]


def zero_white_cmap(lower, upper, name):
    zero = -lower / (upper - lower)
    positions = np.unique(np.r_[np.linspace(0, 1, 513), zero])
    source_positions = np.where(positions <= zero, .5 * positions / zero,
                               .5 + .5 * (positions - zero) / (1 - zero))
    colours = plt.get_cmap('RdBu_r')(source_positions)
    colours[positions == zero] = [1, 1, 1, 1]
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colours)), N=4096)
    cmap.set_bad('white')
    return cmap


def axes_at(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH_CM, y / HEIGHT_CM, w / WIDTH_CM, h / HEIGHT_CM])


def style(ax, title, ylabel='', label=False, field=False, wide=False):
    ax.set_xlim(START, STOP)
    ax.set_xticks(pd.date_range(START, END, freq='2D'))
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %d' if wide else '%b\n%d'))
    ax.tick_params(which='major', direction='in', width=.65, length=3.3,
                   top=True, right=True, colors='black', labelsize=12, pad=3,
                   labelbottom=label)
    ax.tick_params(which='minor', direction='in', width=.5, length=1.8, top=True, right=True)
    ax.set_title(title, loc='left', fontsize=14, fontweight='normal', pad=6.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color='black', rotation=90,
                      ha='center', va='center')
        ax.yaxis.set_label_coords(-.063 if wide else (-.138 if field else -.160), .5)
    if label and not wide:
        ax.set_xlabel('UTC date | 2024–2025', fontsize=12, labelpad=4)
    ax.set_axisbelow(True)
    if not field:
        ax.grid(True, which='major', color='#9ca6b0', linewidth=.65, linestyle=':', alpha=.72)
    for when, color, ls, width, gid in [
        ('2024-12-31 16:18', '#d62728', '--', .85, 'USGS storm commencement'),
        ('2025-01-03 00:30', '#666666', '-.', .8, 'Operational storm end: 80% Dst recovery')]:
        line = ax.axvline(pd.Timestamp(when), color=color, ls=ls, lw=width, zorder=10)
        line.set_gid(gid)


def plot(chem, particle, hourly, kp, dst, previous_checks):
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 12,
        'axes.labelsize': 12, 'axes.linewidth': .65, 'axes.titleweight': 'normal',
        'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54))
    axes = [axes_at(fig, x, y, 8.02, 2.20)
            for y in [16.70, 13.65] for x in [1.70, 11.70]]
    axes.append(axes_at(fig, 1.70, 9.90, 18.02, 2.20))
    axes.extend([axes_at(fig, x, y, 7.60, 3.05)
                 for y in [5.55, 1.65] for x in [1.70, 10.95]])
    axes[0].bar(kp.time + pd.Timedelta(hours=1.5), kp.Kp, width=.1,
                color=COLORS['Kp'], edgecolor=COLORS['Kp'], linewidth=.3, zorder=2)
    axes[0].set(ylim=(0, 9), yticks=[0, 3, 6, 9])
    axes[1].plot(dst.time, dst.Dst_nT, color=COLORS['Dst'], lw=1.6)
    axes[1].set(ylim=(-250, 50), yticks=[-200, -100, 0, 50])
    axes[1].axhline(0, color='.6', lw=.55, alpha=.6)
    axes[2].plot(hourly.index, hourly.aeValue, color=COLORS['AE'], lw=1.5)
    axes[2].set(ylim=(0, 1600), yticks=[0, 800, 1600])
    axes[3].plot(hourly.index, hourly.PCN, color=COLORS['PCN'], lw=1.5)
    axes[3].set(ylim=(-2, 12), yticks=[0, 6, 12])
    for ax, title, ylabel in zip(axes[:4],
            ['(a) Planetary activity', '(b) Ring-current disturbance',
             '(c) Auroral activity', '(d) Polar-cap activity'],
            ['Kp', 'Dst (nT)', 'AE (nT)', 'PCN (mV m⁻¹)']):
        style(ax, title, ylabel)
    handles = []
    for band in ['40-60', '60-80']:
        for channel, ls, energy in [('e1', '-', 40), ('e3', '--', 287)]:
            group = particle[(particle.band == band) & (particle.variable == 'MEPED_' + channel)]
            values = group.set_index('date').reindex(DAYS.strftime('%Y-%m-%d')).value
            line, = axes[4].plot(MID, values, color=COLORS[band], ls=ls, lw=1.65,
                marker='o' if channel == 'e1' else None, ms=3,
                label=f'{band.replace("-", "–")}°N, >{energy} keV')
            handles.append(line)
    # Preserve the original EEP scale; extend only if the added day needs it.
    old_particle = pd.read_csv(BASE / 'MEPED_plotted_values.csv').value
    finite = particle.value[np.isfinite(particle.value)]
    axes[4].set_yscale('log')
    axes[4].set_ylim(min(old_particle.min(), finite.min()) / 1.5,
                     max(old_particle.max(), finite.max()) * 1.5)
    axes[4].set_yticks([1e2, 1e3, 1e4, 1e5])
    style(axes[4], '(e) EEP proxy (MEPED) | cm⁻² s⁻¹ sr⁻¹',
          'Electron flux', label=True, wide=True)
    legend = fig.legend(handles=handles, loc='lower right',
        bbox_to_anchor=(19.72 / WIDTH_CM, 12.25 / HEIGHT_CM), ncol=2,
        frameon=False, fontsize=11, handlelength=1.6, handletextpad=.45,
        columnspacing=1., labelspacing=.30, borderpad=0, borderaxespad=0)
    colour_scales = []
    for index, band, field in [(5, '40-60', 'dO3'), (6, '60-80', 'dO3'),
                                (7, '40-60', 'dT'), (8, '60-80', 'dT')]:
        ax = axes[index]
        table = chem[chem.band == band]
        levels = table.groupby('pressure_hPa').reference_height_km.first().sort_values()
        values = table.pivot(index='date', columns='pressure_hPa', values=field).reindex(
            index=DAYS.strftime('%Y-%m-%d'), columns=levels.index).to_numpy()
        old_scale = next(item for item in previous_checks['individual_panel_colour_scales']
                         if item['panel'] == chr(97 + index))
        lower, upper = old_scale['colour_limits']
        if field == 'dO3':
            cmap = plt.get_cmap('BrBG').copy()
            cmap.set_bad('white')
            norm = TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
        else:
            cmap = zero_white_cmap(lower, upper, 'Figure2_' + band)
            norm = Normalize(vmin=lower, vmax=upper)
        zupper = next(item['display_height_km'][1] for item in previous_checks['thermal_checks']
                      if item['latitude_band'] == band)
        height = levels.to_numpy()
        selected = np.isfinite(height) & (height > 42) & (height < zupper + 8)
        image = ax.pcolormesh(md.date2num(DAYS.append(pd.DatetimeIndex([STOP]))),
            edges(height[selected]), np.ma.masked_invalid(values[:, selected].T),
            shading='flat', norm=norm, cmap=cmap)
        title = f'({chr(97 + index)}) Night {"O₃" if field == "dO3" else "T"} | {band.replace("-", "–")}°N'
        style(ax, title, 'approx. alt. (km)' if index in [5, 7] else '',
              label=index in [7, 8], field=True)
        ax.set_ylim(50, zupper)
        ax.set_yticks([z for z in [50, 60, 70, 80, 90] if z <= zupper])
        ax.tick_params(labelleft=index in [5, 7])
        left = 1.70 if index in [5, 7] else 10.95
        y = 5.55 if index in [5, 6] else 1.65
        bar = fig.colorbar(image, cax=axes_at(fig, left + 7.60 + .25, y, .20, 3.05),
                           ticks=old_scale['colourbar_ticks'], extend='neither')
        bar.ax.tick_params(direction='in', labelsize=10, pad=2, width=.6, length=2.5)
        bar.ax.set_title('ΔO₃ (%)' if field == 'dO3' else 'ΔT (K)',
                         fontsize=10, pad=8, fontweight='normal')
        finite = values[:, (height >= 50) & (height <= zupper)]
        colour_scales.append(dict(panel=chr(97 + index), limits=[lower, upper],
            saturated_cells=int(((finite < lower) | (finite > upper)).sum()),
            native_values_not_clipped=True))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Dec 28 is now exactly at the left boundary. Keep its two-line date label
    # clear of the bottom 50-km tick without moving dates or changing fonts.
    for ax in [axes[7], axes[8]]:
        date_labels = [t for t in ax.get_xticklabels() if t.get_visible()]
        for tick in ax.yaxis.get_major_ticks():
            if tick.label1.get_visible() and np.isclose(tick.get_loc(), ax.get_ylim()[0]):
                if any(tick.label1.get_window_extent(renderer).overlaps(
                        text.get_window_extent(renderer)) for text in date_labels):
                    tick.label1.set_verticalalignment('bottom')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ghost_ticks = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    ghost_ticks.update([tick.label1, tick.label2])
    clipped = []
    for text in fig.findobj(Text):
        if text in ghost_ticks or not text.get_visible() or not text.get_text().strip():
            continue
        assert not re.search(r'[\u3400-\u9fff]', text.get_text())
        box = text.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            clipped.append(text.get_text())
    assert not clipped, clipped
    for ax in [axes[4], axes[7], axes[8]]:
        ticks = [t for t in ax.get_xticklabels() if t.get_visible()]
        for left, right in zip(ticks[:-1], ticks[1:]):
            assert not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))
        for ytick in ax.yaxis.get_major_ticks():
            if ytick.label1.get_visible() and min(ax.get_ylim()) <= ytick.get_loc() <= max(ax.get_ylim()):
                for xtick in ticks:
                    assert not ytick.label1.get_window_extent(renderer).overlaps(
                        xtick.get_window_extent(renderer)), ('Corner tick overlap', ytick.get_loc())
    for below, above in [([2, 3], [0, 1]), ([4], [2, 3]), ([5, 6], [4]), ([7, 8], [5, 6])]:
        for i in below:
            assert axes[i]._left_title.get_window_extent(renderer).y1 < min(axes[j].bbox.y0 for j in above)
    box = legend.get_window_extent(renderer)
    assert box.y0 > axes[4].bbox.y1 and box.y1 < axes[3].bbox.y0
    assert not box.overlaps(axes[4]._left_title.get_window_extent(renderer))
    png = OUT / 'Figure2_storm_activity_EEP_ozone_temperature_Dec28_Jan11.png'
    fig.savefig(png, dpi=600)
    fig.savefig(png.with_name(png.stem + '_preview.png'), dpi=140)
    plt.close(fig)
    with Image.open(png) as image:
        pixels = list(image.size)
    assert abs(pixels[0] - WIDTH_CM / 2.54 * 600) < 2
    return dict(path=str(png), pixels=pixels, width_cm=WIDTH_CM,
                height_cm=HEIGHT_CM, dpi=600, clipped_text=clipped,
                individual_colour_scales=colour_scales)


def main():
    OUT.mkdir(exist_ok=True)
    previous_checks = json.loads(source(BASE / 'data_and_layout_checks.json').read_text(encoding='utf-8'))
    hourly, kp, dst = indices()
    print('Indices complete through Jan 11', flush=True)
    chem = chemistry()
    print('MLS extension calculated; Jan 10 method reproduced', flush=True)
    particle = particles()
    for variable, frame, date_column in [('chemistry', chem, 'date'), ('particles', particle, 'date')]:
        assert frame[date_column].min() == str(START.date())
        assert frame[date_column].max() == str(END.date())
        assert frame[date_column].nunique() == 15
    result = plot(chem, particle, hourly, kp, dst, previous_checks)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in HASHES.items())
    checks = dict(figure=result, period_UTC=[str(START.date()), str(END.date())],
        full_last_day_included=True, x_limits_UTC=[str(START), str(STOP)],
        anomalies_reference='Dec 27-30, 2024; unchanged',
        original_ozone_baseline_threshold_ppmv=.05, original_missing_level_mask_retained=True,
        temperature_and_ozone_colorbars_unchanged=True, layout_fonts_and_event_markers_preserved=True,
        date_tick_step_days=2, no_smoothing=True, no_gap_filling=True,
        validation=VALIDATION,
        Jan11_chemistry_nonmissing=chem[chem.date == str(END.date())][['dO3', 'dT']].count().to_dict(),
        Jan11_particle_nonmissing=int(particle[particle.date == str(END.date())].value.notna().sum()),
        MEPED_original_fixed_support='two_region_diagnostics_20261021/forcing/fixed_support_cells.csv',
        EEP_limitation=previous_checks['EEP_limitation'],
        storm_start=previous_checks['storm_start'], storm_end=previous_checks['storm_end'],
        downloads=DOWNLOADS, input_sha256=HASHES, PDF_export=False)
    (OUT / 'data_and_layout_checks.json').write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figure=result, validation=VALIDATION,
        Jan11_chemistry_nonmissing=checks['Jan11_chemistry_nonmissing'],
        Jan11_particle_nonmissing=checks['Jan11_particle_nonmissing']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
