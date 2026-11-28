"""Two-stage test: upper W1 / vortex response, then vertical extension.

Read-only source diagnostics. Dec 28-Jan 11, reference Dec 28-30.
No weather panels, no inferred W1->W2 transfer, and PNG-only export.
"""
from pathlib import Path
import hashlib
import json
import sys

# Reuse the project's existing Cartopy installation without changing the environment.
_local_packages = Path(__file__).resolve().parent / 'plot_packages'
if _local_packages.exists():
    sys.path.insert(0, str(_local_packages))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from PIL import Image
import cartopy.crs as ccrs

import plot_era5_daily_vortex_boundaries as base
import analyze_vortex_regional_extension as region
from analyze_w1_w2_vortex_20261025 import geometry

ROOT = Path(__file__).resolve().parent
WAVE = ROOT.parent / 'planetary_waves'
OUT = ROOT / 'results' / 'upper_wave_vortex_two_stage_20261026'
COMMON = ROOT / 'results/ERA5_MLS_W1_PV_link_20261025/common_layer_daily.csv'
FLUX = ROOT / 'results/planetary_wave_vortex_mechanism_20261026/Plumb1985_regional_vertical_flux_daily.csv'
MLS = WAVE / 'W1_three_bands_Dec27_Jan20_20261016/MLS_W123_fullheight_Dec27_Jan20.npz'
DAYS = pd.date_range('2024-12-28', '2025-01-11')
PRE = np.asarray(DAYS <= '2024-12-30')
POST = np.asarray((DAYS >= '2025-01-06') & (DAYS <= '2025-01-09'))
ONSET = pd.Timestamp('2024-12-31 16:18')
PURPLE, BLUE, GREEN = '#803a9b', '#167da3', '#16805d'
QA = {}


def wrap(x):
    return (np.asarray(x) + 180) % 360 - 180


def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 12.5, 'axes.titleweight': 'normal',
        'axes.labelsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'axes.linewidth': .8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'savefig.bbox': None,
    })


def time_axis(ax, periods=True, heat=False):
    # Day labels coincide with daily sample centers; edges are half a day away.
    ax.set_xlim(DAYS[0] - pd.Timedelta(hours=12), DAYS[-1] + pd.Timedelta(hours=12))
    ax.set_xticks(pd.date_range('2024-12-29', '2025-01-10', freq='3D'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
    ax.set_xlabel('UTC date | 2024–2025', labelpad=4)
    ax.tick_params(top=True, right=True, length=3)
    if periods:
        ax.axvspan(DAYS[0] - pd.Timedelta(hours=12),
                   pd.Timestamp('2024-12-30 12:00'), color='.75', alpha=.13, lw=0, zorder=-3)
        ax.axvspan(pd.Timestamp('2025-01-05 12:00'),
                   pd.Timestamp('2025-01-09 12:00'), color='#d9ae42', alpha=.11, lw=0, zorder=-3)
    ax.axvline(ONSET, color='white' if heat else '#c74336', lw=1, ls='--', zorder=8)
    if not heat:
        ax.grid(axis='y', color='.85', ls=':', lw=.6, zorder=-4)


def pressure_axis(ax, levels, hide=False):
    ax.set_yscale('log')
    ax.set_yticks(levels)
    ax.set_yticklabels([f'{p:g}' for p in levels])
    ax.minorticks_off()
    if hide:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel('Pressure (hPa)', labelpad=3)


def cell_edges(pressure):
    """Daily centers and geometric pressure edges, without field interpolation."""
    logp = np.log(np.asarray(pressure, dtype=float))
    pedges = np.exp(np.r_[logp[0]-(logp[1]-logp[0])/2,
                         (logp[:-1]+logp[1:])/2,
                         logp[-1]+(logp[-1]-logp[-2])/2])
    dates = mdates.date2num(DAYS.to_pydatetime())
    return np.r_[dates-.5, dates[-1]+.5], pedges


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label,
                  *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            texts += list(legend.get_texts())
    for txt in texts:
        if txt.get_visible() and txt.get_text():
            bb = txt.get_window_extent(renderer)
            if bb.x0 < -.6 or bb.y0 < -.6 or bb.x1 > fig.bbox.width + .6 or bb.y1 > fig.bbox.height + .6:
                outside.append(txt.get_text())
    if outside:
        raise RuntimeError(f'Text outside figure: {name}: {outside}')
    fig.savefig(OUT / name, dpi=600, facecolor='white')
    plt.close(fig)
    with Image.open(OUT / name) as im:
        assert abs(im.width - 4724) <= 1
        size = list(im.size)
        preview = im.copy()
        preview.thumbnail((1500, 1800))
        preview.save(OUT / (Path(name).stem + '_preview.png'))
    QA.setdefault('figures', []).append(dict(file=name, pixels=size,
        width_cm=20, dpi=600, outside_text=outside))


def load_waves():
    common = pd.read_csv(COMMON)
    common = common[common.band == '60-80']
    pressure = np.sort(common.pressure_hPa.unique())
    pressure = pressure[pressure <= 100]
    era = []
    eh = []
    for p in pressure:
        sub = common[common.pressure_hPa == p].set_index('date').loc[DAYS.strftime('%Y-%m-%d')]
        era.append(sub.ERA5_C_real.to_numpy() + 1j * sub.ERA5_C_imag.to_numpy())
        eh.append(sub.ERA5_height_km.to_numpy())
    era = np.stack(era, axis=1)
    eh = np.stack(eh, axis=1)
    m = np.load(MLS)
    ids = [np.flatnonzero(m['dates'].astype('datetime64[D]') == np.datetime64(day.date()))[0] for day in DAYS]
    lat = m['lat']
    iy = (lat >= 60) & (lat <= 80)
    weights = np.cos(np.deg2rad(lat[iy]))
    # Stored MLS least-squares coefficients are a+i*b; use C=a-i*b in Re(C*e^iλ).
    coeff = np.conj(np.take(m['cz'][0, 0, ..., 0], ids, axis=0))
    c = np.average(coeff[:, iy], weights=weights, axis=1)
    z0 = np.take(m['z0'][0, 0], ids, axis=0)
    height = np.average(z0[:, iy], weights=weights, axis=1) / 1000
    anchor = int(np.argmin(abs(height[PRE].mean(0) - 50)))
    sector = np.conj(np.take(m['cz'][0, 1:, ..., 0], ids, axis=1))
    sector = np.average(sector[:, :, iy], weights=weights, axis=2)
    branch = np.conj(np.take(m['cz'][1:3, 0, ..., 0], ids, axis=1))
    branch = np.average(branch[:, :, iy], weights=weights, axis=2)
    sector_error = np.nanpercentile(abs(wrap(np.angle(sector * c.conj()[None], deg=True))), 95, axis=0)
    branch_error = np.nanmax(abs(wrap(np.angle(branch * c.conj()[None], deg=True))), axis=0)
    usable = (abs(c) >= 50) & (sector_error <= 30) & (branch_error <= 45)
    # Keep only the MLS anchor and its native neighbors BELOW it, up to the ERA5 top.
    upper_ids = np.flatnonzero((m['pressure'] >= m['pressure'][anchor] * .999) & (m['pressure'] < 1))
    upper_ids = upper_ids[np.argsort(m['pressure'][upper_ids])]
    combined_p = np.r_[m['pressure'][upper_ids], pressure]
    combined_c = np.c_[c[:, upper_ids], era]
    ratio = abs(combined_c) / abs(combined_c[PRE]).mean(axis=0)
    phase = wrap(np.angle(combined_c * c[:, anchor, None].conj(), deg=True))
    good = (abs(combined_c) >= 50) & usable[:, anchor, None]
    good[:, :len(upper_ids)] &= usable[:, upper_ids]
    phase = np.where(good, phase, np.nan)
    seam = float(np.sqrt(combined_p[len(upper_ids)-1] * pressure[0]))
    rows = []
    for j, p in enumerate(combined_p):
        for i, day in enumerate(DAYS):
            rows.append(dict(date=str(day.date()), source='MLS' if j < len(upper_ids) else 'ERA5',
                pressure_hPa=float(p), C_real=float(combined_c[i, j].real),
                C_imag=float(combined_c[i, j].imag), amplitude_m=float(abs(combined_c[i, j])),
                amplitude_ratio=float(ratio[i, j]),
                trough_longitude_degE=float(wrap(180 - np.angle(combined_c[i, j], deg=True))),
                phase_relative_to_MLS50_deg=float(phase[i, j]), phase_valid=bool(good[i, j])))
    write_csv('W1_vertical_amplitude_phase.csv', rows)
    QA['wave_method'] = {
        'band': '60-80 N; four shared MLS native centers, cosine-weighted complex coefficients before magnitude.',
        'harmonics': 'Absolute daily GPH zonal harmonics, NOT climatological-anomaly harmonics. No temporal smoothing in amplitude/phase panels.',
        'MLS_anchor_pressure_hPa': float(m['pressure'][anchor]),
        'MLS_anchor_pre_height_km': float(height[PRE, anchor].mean()),
        'ERA5_3hPa_pre_height_km': float(eh[PRE, pressure == 3].mean()),
        'phase_convention': 'Z1=Re(C exp(i lambda)); trough longitude=wrap(180-arg(C)); relative phase=arg(C(p)conj(C_MLS50)).',
        'phase_screen': 'Amplitude>=50 m; MLS anchor leave-one-sector p95<=30deg and day/night max<=45deg. MLS upper cells also require their own sensitivity screen; not confidence intervals.',
        'source_seam_pressure_hPa': seam,
        'seam_note': 'Discrete native MLS cells above the geometric midpoint to ERA5 1hPa; no interpolation across the source join.',
        'screened_phase_cells': int((~good).sum()),
    }
    return dict(p=pressure, era=era, height=eh, mls=c[:, anchor],
                mls_good=usable[:, anchor], combined_p=combined_p, ratio=ratio,
                phase=phase, seam=seam)


def load_pv():
    source = np.load(base.CACHE)
    lat, lon = source['lat'], source['lon']
    weights = base.cell_weights(lat, len(lon))
    sec = (lon >= -130) & (lon <= -60)
    data = {}
    rows, summary = [], []
    for theta in [450, 850, 1200]:
        j = np.flatnonzero(source['theta'] == theta)[0]
        fields = np.stack([base.smooth_map(x) for x in source['pv'][:, j]])
        assert np.isfinite(fields).all()
        threshold = float(source['pv_edge'][PRE, j].mean())
        edge = np.full((len(DAYS), len(lon)), np.nan)
        daily_geo = []
        for i, day in enumerate(DAYS):
            core, fraction = region.main_component(fields[i] >= threshold, weights)
            assert not core[0].any(), 'Vortex core reaches 20N input limit'
            for x in range(len(lon)):
                ids = np.flatnonzero(core[:, x])
                if not len(ids):
                    continue
                n = ids[0]
                lo, hi = fields[i, n-1, x], fields[i, n, x]
                edge[i, x] = lat[n-1] + (threshold-lo)/(hi-lo)*(lat[n]-lat[n-1])
            g = geometry(core, lat, lon, weights)
            daily_geo.append(g)
            h = float(np.median(source['iso_z'][i, j][core]) / 1000)
            rows.append(dict(date=str(day.date()), theta_K=theta, threshold_PVU=threshold,
                NA_edge_lat=float(np.nanmean(edge[i, sec])), height_km=h,
                retained_area_fraction=float(fraction), **g))
        edge_lat = np.nanmean(edge[:, sec], axis=1)
        g = pd.DataFrame(daily_geo)
        data[theta] = dict(fields=fields, threshold=threshold, edge_lat=edge_lat,
            shift=edge_lat[PRE].mean()-edge_lat, geometry=g,
            height=float(np.mean([r['height_km'] for r in rows if r['theta_K'] == theta][:3])))
        for name, select in [('Dec28-30', PRE), ('Jan6-9', POST)]:
            summary.append(dict(theta_K=theta, period=name,
                NA_edge_lat=float(edge_lat[select].mean()),
                NA_southward_shift_deg=float(data[theta]['shift'][select].mean()),
                centroid_lat=float(g.centroid_lat[select].mean()),
                centroid_lon=float(np.angle(np.mean(np.exp(1j*np.deg2rad(g.centroid_lon[select]))), deg=True)),
                aspect_ratio=float(g.aspect_ratio[select].mean()),
                height_km=data[theta]['height']))
    write_csv('PV_geometry_daily.csv', rows)
    write_csv('PV_geometry_periods.csv', summary)
    QA['PV_method'] = {
        'boundary': 'Same operational PV value fixed at Dec28-30 mean on each theta surface; NOT a daily reselected threshold.',
        'connectivity': 'Largest longitude-periodic connected high-PV region; spherical area centroid; holes retained.',
        'position': 'Daily centroid plus mean outer boundary latitude in 130-60 W. Shift = mean(pre latitude)-today latitude; positive equatorward.',
        'map': 'PV boundary and centroid of PERIOD-MEAN PV field; time-series use daily boundaries/centroids. Nonlinear operations differ.',
        'upper_layer_choice': '1200 K (~36 km), fully covered by ERA5 p>=1hPa inputs. 1600K omitted because of partial coverage above input top; 1500K omitted because fixed-core contours can contact southern input boundary.',
        'resolution': '2.5deg grid, sigma=.55-grid spatial smoothing before fixed-PV selection. Linear edge interpolation does not increase effective resolution.',
        'interpretation': 'North-American sector excursion is not a rigid translation of the entire vortex. GPH phase and PV geometry are complementary diagnostics, not independent datasets.',
    }
    return source, data


def upper_map(fig, rect, source, pv):
    pc = ccrs.PlateCarree()
    ax = fig.add_axes(rect, projection=ccrs.NorthPolarStereo(central_longitude=-90))
    circle = np.linspace(0, 2*np.pi, 361)
    ax.set_boundary(MplPath(np.c_[np.sin(circle), np.cos(circle)]*.5+.5), transform=ax.transAxes)
    ax.set_extent([-180, 180, 25, 90], crs=pc)
    ax.coastlines(resolution='110m', color='.43', linewidth=.5, zorder=4)
    ax.gridlines(crs=pc, xlocs=[-180, -90, 0, 90], ylocs=[40, 60, 80],
                 color='.4', linewidth=.65, linestyle='--', alpha=.7, zorder=1)
    x = np.r_[source['lon'], 180.]
    lat = source['lat']
    weights = base.cell_weights(lat, len(source['lon']))
    centers = []
    for select, color, linestyle in [(PRE, '.43', '--'), (POST, BLUE, '-')]:
        q = pv['fields'][select].mean(0)
        core, _ = region.main_component(q >= pv['threshold'], weights)
        assert not core[0].any()
        # Remove detached high-PV islands from the contour, matching centroid selection.
        qplot = np.where((q >= pv['threshold']) & ~core, pv['threshold']-1, q)
        if select is POST:
            ax.contourf(x, lat, np.c_[core, core[:, 0]].astype(float), levels=[.5, 1.5],
                        colors=['#dcebf1'], transform=pc, zorder=0)
        ax.contour(x, lat, np.c_[qplot, qplot[:, 0]], levels=[pv['threshold']],
                   colors=[color], linestyles=linestyle, linewidths=1.8, transform=pc, zorder=5)
        g = geometry(core, lat, source['lon'], weights)
        centers.append(g)
        ax.plot(g['centroid_lon'], g['centroid_lat'], 'o', markersize=5,
                markeredgewidth=1.2, markeredgecolor=color,
                markerfacecolor=color if select is POST else 'white', transform=pc, zorder=7)
    a, b = [ax.projection.transform_point(c['centroid_lon'], c['centroid_lat'], pc) for c in centers]
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.1), zorder=6)
    ax.plot(0, 90, '+', color='.3', ms=5, transform=pc, zorder=6)
    for label, xx, yy, ha in [('90°E', .5, 1.025, 'center'), ('90°W', .5, -.065, 'center'),
                              ('180°', -.035, .5, 'right'), ('0°', 1.035, .5, 'left')]:
        ax.text(xx, yy, label, transform=ax.transAxes, ha=ha, va='center', fontsize=11, color='.28')
    for la in [40, 60, 80]:
        ax.text(45, la, f'{la}°N', transform=pc, fontsize=10.5, color='.28', zorder=8)
    ax.set_title('(c) Upper-vortex boundary', loc='left', pad=27)
    ax.text(.5, 1.075, f'1200 K (~{pv["height"]:.0f} km)', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)
    ax.legend(handles=[Line2D([], [], color='.43', ls='--', label='Dec 28–30'),
                       Line2D([], [], color=BLUE, label='Jan 6–9')],
              loc='lower left', bbox_to_anchor=(-.07, -.02), frameon=False,
              ncol=1, handlelength=1.5, handletextpad=.4, fontsize=11)
    return ax


def figure_upper(waves, source, pv):
    fig = plt.figure(figsize=(20/2.54, 20.5/2.54))
    a = fig.add_axes([.12, .62, .36, .285])
    b = fig.add_axes([.61, .62, .36, .285])
    j = np.flatnonzero(waves['p'] == 3)[0]
    ca, cb = waves['mls'], waves['era'][:, j]
    labels = ['MLS ~50 km', f'ERA5 ~{waves["height"][PRE, j].mean():.0f} km']
    for c, color, label in [(ca, PURPLE, labels[0]), (cb, BLUE, labels[1])]:
        a.plot(DAYS, abs(c), color=color, lw=1.9, marker='o', ms=2.8, label=label)
    a.set_ylabel('W1 amplitude (m)', labelpad=4)
    a.set_ylim(0, max(2700, np.ceil(max(abs(ca).max(), abs(cb).max())/500)*500))
    a.set_yticks([0, 1000, 2000])
    a.set_title('(a) Upper-level W1 amplitude', loc='left', pad=9)
    a.legend(loc='upper left', frameon=False, handlelength=1.5, handletextpad=.4, fontsize=11.5)
    for c, color, label in [(ca, PURPLE, labels[0]), (cb, BLUE, labels[1])]:
        phase = wrap(180 - np.angle(c, deg=True))
        if color == PURPLE:
            phase[~waves['mls_good']] = np.nan
        # Break lines across the cyclic dateline rather than joining opposite edges.
        phase = phase.astype(float)
        phase[np.r_[False, abs(np.diff(phase)) > 180]] = np.nan
        b.plot(DAYS, phase, color=color, lw=1.9, marker='o', ms=2.8)
    b.axhspan(-130, -60, color=GREEN, alpha=.08, lw=0, zorder=-4)
    b.set_ylim(-180, 180)
    b.set_yticks([-180, -90, 0, 90, 180])
    b.set_yticklabels(['180°', '90°W', '0°', '90°E', '180°'])
    b.set_ylabel('Trough longitude', labelpad=3)
    b.set_title('(b) Upper-level W1 phase', loc='left', pad=9)
    time_axis(a)
    time_axis(b)
    upper_map(fig, [.12, .12, .36, .35], source, pv[1200])
    d = fig.add_axes([.61, .12, .36, .35])
    d.plot(DAYS, pv[1200]['geometry'].centroid_lat, color=PURPLE, lw=1.8, ls='--', label='Centroid')
    d.plot(DAYS, pv[1200]['edge_lat'], color=BLUE, lw=1.9, marker='o', ms=2.8, label='N. American edge')
    d.set_ylim(40, 90)
    d.set_yticks([40, 50, 60, 70, 80, 90])
    d.set_ylabel('Latitude (°N)', labelpad=3)
    d.set_title('(d) Upper-vortex latitude', loc='left', pad=27)
    d.legend(loc='lower left', frameon=False, handlelength=1.5, handletextpad=.4, fontsize=11.5)
    time_axis(d)
    fig.suptitle('Upper-stratospheric waves and vortex response', fontsize=16, fontweight='normal', y=.985)
    save(fig, 'Figure4A_upper_W1_and_vortex.png')


def colorbar(fig, ax, image, title, ticks):
    pos = ax.get_position()
    ca = fig.add_axes([pos.x1+.014, pos.y0, .012, pos.height])
    cb = fig.colorbar(image, cax=ca, ticks=ticks, extend='neither')
    cb.ax.set_title(title, fontsize=11, pad=5)
    cb.ax.tick_params(labelsize=10.5, length=2, direction='in', pad=2)
    return cb


def figure_extension(waves, pv):
    fig = plt.figure(figsize=(20/2.54, 21.8/2.54))
    a = fig.add_axes([.12, .615, .315, .275])
    b = fig.add_axes([.59, .615, .315, .275])
    # Display discrete pressure cells in log-p: no vertical interpolation.
    cp = waves['combined_p']
    tx, py = cell_edges(cp)
    vmax = max(2., np.ceil(np.nanmax(waves['ratio'])*2)/2)
    im = a.pcolormesh(tx, py, waves['ratio'].T, shading='flat',
                      cmap='RdBu_r', norm=TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax))
    colorbar(fig, a, im, r'$A_1$/pre', [0, 1, 2] + ([vmax] if vmax > 2 else []))
    im = b.pcolormesh(tx, py, waves['phase'].T, shading='flat',
                      cmap='twilight_shifted', vmin=-180, vmax=180)
    colorbar(fig, b, im, 'deg', [-180, -90, 0, 90, 180])
    for ax in [a, b]:
        pressure_axis(ax, [.5, 1, 3, 10, 30, 100], hide=ax is b)
        ax.set_ylim(py[-1], py[0])
        ax.axhline(waves['seam'], color='.2', lw=.8, ls=':', zorder=6)
        time_axis(ax, periods=False, heat=True)
    # Source tags only once; the horizontal dotted join is shared by both panels.
    a.text(.03, .95, 'MLS', transform=a.transAxes, fontsize=10, va='top',
           bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=.8))
    a.text(.03, .78, 'ERA5', transform=a.transAxes, fontsize=10, va='top',
           bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=.8))
    a.set_title('(a) W1 amplitude ratio', loc='left', pad=10)
    b.set_title('(b) Relative W1 phase', loc='left', pad=10)
    c = fig.add_axes([.12, .12, .315, .355])
    for theta, color in [(1200, BLUE), (850, '#ce761f'), (450, GREEN)]:
        v = pv[theta]
        c.plot(DAYS, v['shift'], color=color, lw=1.9, label=f'{theta} K (~{v["height"]:.0f} km)')
    c.axhline(0, color='.45', lw=.8)
    c.set_ylabel('Southward shift (°)', labelpad=3)
    c.set_ylim(-10, 23)
    c.set_yticks([-10, 0, 10, 20])
    c.set_title('(c) N. American PV-edge shift', loc='left', pad=10)
    c.legend(loc='upper left', frameon=False, fontsize=11.2, handlelength=1.3,
             handletextpad=.4, labelspacing=.25)
    time_axis(c)
    d = fig.add_axes([.59, .12, .315, .355])
    flux = pd.read_csv(FLUX)
    flux = flux[(flux.component == 'W1') & (flux.pressure_hPa >= 2) & (flux.pressure_hPa <= 100)]
    f = flux.pivot(index='date', columns='pressure_hPa', values='Fz_m2_s2').loc[DAYS.strftime('%Y-%m-%d')]
    lim = np.ceil(np.nanmax(abs(f.to_numpy())*1000)/5)*5
    tx, py = cell_edges(f.columns.to_numpy())
    im = d.pcolormesh(tx, py, f.to_numpy().T*1000,
        shading='flat', cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    colorbar(fig, d, im, '$10^{-3}$\nm$^2$ s$^{-2}$', [-lim, 0, lim])
    pressure_axis(d, [2, 10, 50, 100])
    d.set_ylim(py[-1], py[0])
    d.set_title('(d) Vertical W1 wave flux', loc='left', pad=10)
    time_axis(d, periods=False, heat=True)
    fig.suptitle('Testing extension into the lower stratosphere', fontsize=16, fontweight='normal', y=.985)
    QA['flux_method'] = {
        'source': str(FLUX), 'method': 'Existing Plumb1985 stationary QG W1 vertical flux; no recomputation.',
        'latitude_band': '45-75N; single-W1 flux under zonal background is longitude-independent.',
        'filter': 'Centered 3-day coefficients and zonal background from buffered data; unsuitable for exact onset/causal lag.',
        'validity': 'U>=5m/s and N2>1e-6/s2, minimum 80% valid area; top1hPa omitted.',
        'sign': 'Positive upward, negative downward. Net single-W1 flux does not separate incident and reflected components.',
        'range_1e3_m2_s2': [float(np.nanmin(f.to_numpy())*1000), float(np.nanmax(f.to_numpy())*1000)],
        'negative_valid_cells': int(np.sum(f.to_numpy() < 0)),
        'claim_limit': 'Amplitude/phase similarity and boundary coevolution alone do not establish downward causality or magnetic-storm attribution.',
    }
    save(fig, 'Figure4B_vertical_extension_test.png')


def notes(waves, pv):
    j = np.flatnonzero(waves['p'] == 3)[0]
    w1pre, w1post = abs(waves['mls'][PRE]).mean(), abs(waves['mls'][POST]).mean()
    ep, eq = abs(waves['era'][PRE, j]).mean(), abs(waves['era'][POST, j]).mean()
    text = f'''# 两阶段图：图注、数据口径与结果边界

两图主时段均为2024年12月28日至2025年1月11日，基准期为12月28–30日；对比期固定为1月6–9日。仅导出PNG，20 cm宽、600 dpi。原六面板图与原脚本均保留。

## 第一幅：上层波动与极涡响应

文件：Figure4A_upper_W1_and_vortex.png。

图注建议：图4A. 上层行星波1与上平流层极涡位置的演变。（a）60°–80°N MLS近50 km层与ERA5 3 hPa层的位势高度W1振幅；（b）对应W1槽位经度，浅绿色水平带表示北美经度扇区（130°–60°W）；（c）1200 K等熵面（约{pv[1200]['height']:.0f} km）上磁暴前12月28–30日与1月6–9日期间平均PV场的极涡边界，灰色虚线和蓝色实线分别表示两个时期，空心点和实心点表示对应高PV主体的面积质心，小箭头连接两期质心位置；（d）同一等熵面上每日极涡质心纬度及北美扇区平均外边界纬度。PV阈值固定为基准期每日诊断边界PV值的平均值，并保留经度周期连接的最大高PV主体。灰色与浅黄色竖向背景分别表示基准期与1月6–9日，红色竖虚线表示磁暴开始时刻（12月31日16:18 UT）。

第一排的振幅、相位是两个独立的子图，未使用双纵轴；不以W1振幅代替实际边界。3 hPa与1200 K高度接近，用于上平流层的谱结构—几何对应，但不是严格同一个垂直面。MLS选取基准期纬向平均高度最接近50 km的原生气压层：{QA['wave_method']['MLS_anchor_pressure_hPa']:.4f} hPa，约{QA['wave_method']['MLS_anchor_pre_height_km']:.2f} km。

## 第二幅：向下延伸检验

文件：Figure4B_vertical_extension_test.png。

图注建议：图4B. 行星波与极涡变化向下平流层延伸的诊断。（a）60°–80°N各气压层的每日W1振幅相对于该层12月28–30日平均振幅的比值；（b）各层W1复系数相对于同日MLS近50 km复系数的相位差，0°表示同相，±180°表示反相，并不直接指示波传播方向；（c）1200、850与450 K等熵面上北美扇区（130°–60°W）PV外边界相对基准期的纬度变化，正值表示向赤道方向伸展；（d）45°–75°N平均的Plumb（1985）定常准地转W1垂直波作用通量，正值向上、负值向下。图（a、b）水平点线表示MLS与ERA5原生层之间的资料接合位置，没有跨资料插值；白色竖虚线表示磁暴开始时刻。图（c）阴影同图4A。

振幅比以每层日振幅的基准期平均作分母，不是复系数先取基准平均后的模，也不是气候态异常；因此每一层基准期的振幅比平均为1。相位差由复系数直接计算，未把不同高度各自的相位强行对齐。MLS相位使用删经度扇区和昼夜分支敏感性筛选，缺测/未通过筛选的格点不填充。MLS与ERA5均使用相同四个原生5°纬度中心及余弦权重，先平均复系数再取振幅。

WAF采用已有诊断，输入含前后缓冲时段并做居中3日平滑；不能用其最早变化时刻推断精确因果滞后。它是净W1通量，不能分离入射波和反射波，也不是经向—高度MLS EP箭头的直接替代。纬带与振幅/相位图不同，图注已分别说明。上层1 hPa边界不计算WAF。没有给负通量区域人为加色或改变符号。

## 新图显示什么

- MLS约50 km W1平均振幅：基准期{w1pre:.1f} m，1月6–9日{w1post:.1f} m；ERA5 3 hPa：{ep:.1f}→{eq:.1f} m。
- 1200 K的北美扇区边界平均向南伸展{pv[1200]['shift'][POST].mean():.2f}°，850 K为{pv[850]['shift'][POST].mean():.2f}°，450 K为{pv[450]['shift'][POST].mean():.2f}°。450 K负值代表向北；不能概括为所有高度同步南下。
- 1200 K质心方向转向北美，与北美边界向南伸展可并列描述；整体质心位移和一个经度扇区的边界伸展不等价。相位—几何一致性是结构对应证据，不自动证明波动单向驱动边界。
- 图4B(d)在当前有效格点中没有净向下的W1通量。因此本图不能声称已显示一条从50 km连续向下到100 hPa的W1净传播通道；不能由此反向否定更高层已发现的反射，因为指标、纬带和波分解不同。
- 第二幅是对跨层链条的检验，而不是既定机制的确认。下一步若要做因果归因，需要波作用/动量收支和上层扰动对照试验；相位同向、时间接近或振幅共同变化不足以完成归因。

## 与原图的调整

保留原文件，不在新图重复500 hPa槽、850 hPa平流、T2m和天气归因。暂不画W2；不是认为W2不重要，而是本次主问题是W1变化与极涡位移及其垂直延伸。W1/W2能量交换应另以收支检验，不能由振幅此消彼长推断。

## 方法参考

- Chandra et al. (2026), https://doi.org/10.1038/s41612-026-01418-4 。借鉴将W1振幅、相位与区域极涡构型分开诊断的思路；本图是原始日GPH谐波，不直接照搬其气候态异常和7日平滑。
- Ning et al. (2026), https://doi.org/10.5194/wcd-7-277-2026 。上层非对称强迫与下层响应需区分空间相位及动力学传播证据；该研究的模式因果证据不等同于本事件已完成归因。
- Plumb (1985) WAF公式实现对照： https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/Plumb1985-ncep-daily-10d-lpf.ncl 。本事件3日滤波与该示例滤波参数不同，已在方法说明中区分。
'''
    (OUT / '两阶段图_图注与结果说明.md').write_text(text, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    waves = load_waves()
    source, pv = load_pv()
    assert np.allclose(waves['ratio'][PRE].mean(axis=0), 1)
    assert np.allclose(wrap(np.angle(np.exp(1j*np.deg2rad([0,90,180,-90])), deg=True)), [0,90,-180,-90])
    # Independent implementation consistency: compare to the existing fixed-PV diagnostics.
    old = pd.read_csv(ROOT / 'results/planetary_wave_vortex_mechanism_20261026/PV_geometry_daily.csv')
    for theta in pv:
        assert np.allclose(pv[theta]['edge_lat'], old[old.theta_K == theta].NA_outer_edge_lat)
    figure_upper(waves, source, pv)
    figure_extension(waves, pv)
    notes(waves, pv)
    QA['event'] = dict(start='2024-12-28', end='2025-01-11', baseline='2024-12-28/2024-12-30',
                       comparison='2025-01-06/2025-01-09', PDF_export=False)
    QA['checks'] = dict(per_level_baseline_ratio_one=True, fixed_PV_edges_match_previous=True,
                        phase_wrap_synthetic_test=True)
    QA['sources'] = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in [MLS, COMMON, FLUX, base.CACHE, Path(__file__)]]
    (OUT / 'method_and_layout_checks.json').write_text(json.dumps(QA, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), figures=QA['figures'], wave=QA['wave_method'],
                         flux=QA['flux_method']), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
