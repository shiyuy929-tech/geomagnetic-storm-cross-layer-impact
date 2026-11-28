"""W1 flux proxy and amplitude in 40–80 N, daily EP vectors in 20–80 N.

Uses the established input fields and diagnostics; only panel ordering,
latitude window and presentation are changed from the previous figure.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.text import Text
from matplotlib.colors import Normalize, SymLogNorm, PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DayLocator
from PIL import Image

import plot_W1_amplitude_flux_daily_Jan02_Jan11_2x5_20261022 as daily

core = daily.core
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figure3_flux_amplitude_EP_nature_20261105'
W, H = 18.0, 14.7
X = [.98]
TW, TH = 6.58, 2.70
TY = [11.36, 7.95, 4.54, 1.13]
MX = [9.50, 12.07, 14.64]
MW, MH = 2.12, 3.66
MY = [10.40, 5.76, 1.12]


def axes(fig, x, y, width, height):
    return fig.add_axes([x / W, y / H, width / W, height / H])


def bar(fig, artist, x, y, height, ticks, label):
    cax = axes(fig, x, y, .16, height)
    cb = fig.colorbar(artist, cax=cax, ticks=ticks)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    cb.ax.tick_params(labelsize=7, length=2, pad=1, direction='in')
    cb.ax.set_title(label, fontsize=7, pad=5)
    return cb


def times(fig, bands):
    result = []
    for row, (band, variable) in enumerate((
        (bands[0], 'net_proxy'), (bands[0], 'amplitude'),
        (bands[1], 'net_proxy'), (bands[1], 'amplitude'))):
            col = 0 if band['lo'] == 40 else 1
            ax = axes(fig, X[0], TY[row], TW, TH)
            if variable == 'net_proxy':
                norm, cmap = SymLogNorm(1, vmin=-200, vmax=200), 'RdBu_r'
                ticks, unit = [-200, -10, -1, 0, 1, 10, 200], r'K m s$^{-1}$'
                ylim, yticks, name = (40, 90), np.arange(40, 91, 10), 'Net W1 proxy'
            else:
                norm, cmap = Normalize(0, band['amplitude_max']), 'viridis'
                ticks = ([0, 500, 1000, 1500, 1800] if col == 0 else
                         [0, 500, 1000, 1500, 2000, 2500])
                unit, ylim, yticks, name = 'm', (20, 90), np.arange(20, 91, 10), 'W1 amplitude'
            img = ax.pcolormesh(core.linked.edges(mdates.date2num(core.DAYS)),
                core.linked.edges(band[variable + '_height']),
                np.ma.masked_invalid(band[variable].T), cmap=cmap, norm=norm,
                shading='flat', zorder=1)
            ax.set_xlim(core.DAYS[0] - pd.Timedelta(hours=12), core.DAYS[-1] + pd.Timedelta(hours=12))
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            ax.set_xticks(daily.DATE_TICKS)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
            ax.xaxis.set_minor_locator(DayLocator())
            ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                           labelsize=7, length=3, pad=2, labelbottom=row == 3, labelleft=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True, length=1.5)
            ax.set_ylabel('Approx. alt. (km)', fontsize=7, labelpad=2)
            if row == 3:
                ax.set_xlabel('UTC date', fontsize=7, labelpad=2)
            ax.axvline(daily.COMMENCEMENT, color='white', ls='--', lw=.65, zorder=4)
            ax.axhline(50, color='white', ls='--', lw=.6, zorder=4)
            letter = chr(97 + row)
            ax.set_title(f'{name} | {band["lo"]}–{band["hi"]}°N', fontsize=7,
                         loc='center', pad=5)
            ax.text(.00, 1.02, letter, fontsize=8, fontweight='bold',
                    transform=ax.transAxes, ha='left', va='bottom')
            if variable == 'net_proxy':
                for reg in core.linked.FOCUS_REGIONS:
                    if tuple(reg['band']) == (band['lo'], band['hi']):
                        start = mdates.date2num(pd.Timestamp(reg['start'])) - .4
                        end = mdates.date2num(pd.Timestamp(reg['end'])) + .4
                        low, high = reg['height']
                        ax.add_patch(Rectangle((start, low), end-start, high-low,
                            fill=False, edgecolor=reg['color'], linewidth=1.2, zorder=8))
            bar(fig, img, X[0] + TW + .28, TY[row], TH, ticks, unit)
            result.append(ax)
    return result


def daily_map(fig, lat, levels, q, day, index, norm, cmap):
    row, col = divmod(index, 3)
    ax = axes(fig, MX[col], MY[row], MW, MH)
    ax.set(xlim=(20, 80), ylim=(20, 90), xticks=[20, 40, 60, 80],
           yticks=np.arange(30, 91, 10))
    ax.tick_params(direction='in', top=True, right=True, labelleft=col == 0,
                   labelbottom=row == 2, labelsize=7, pad=1, length=2.5)
    ax.set_axisbelow(True)
    if col == 0:
        ax.set_ylabel('Approx. alt. (km)', fontsize=7, labelpad=2)
    ax.set_title(f'{day:%b} {day.day}', fontsize=7, pad=3)
    ax.text(.005, 1.02, chr(101 + index), fontsize=8, fontweight='bold',
            transform=ax.transAxes, ha='left', va='bottom')
    # Mask outside the requested range while retaining all native physical values.
    selected = (lat[:, None] >= 20) & (lat[:, None] <= 80)
    p = np.where(selected, q['P'], np.nan)
    w = np.where(selected, q['W'], np.nan)
    options = dict(readable=True, uniform_heads=True, length_bounds_inches=(.030, .060),
                   head_points=2.5, shaft_points=.42)
    artist, keep = core.linked.draw_arrows(ax, lat, q['height'], p, w,
                                          levels, cmap, norm, **options)
    boosted, boosted_keep = core.linked.draw_arrows(ax, lat, q['height'],
        np.where(keep, p, np.nan),
        np.where(keep, w * core.linked.DIRECTION_VERTICAL_BOOST, np.nan),
        levels, cmap, norm, **options)
    if not np.array_equal(keep, boosted_keep):
        daily.restyle_by_native_cell(ax, artist, boosted, lat, q, keep)
    else:
        core.linked.rotate_and_style_arrows(ax, artist, boosted, lat, q['height'],
            q['P'], q['W'], keep, highlight_downward=True, shaft_pts=.48)
    for reg in core.linked.FOCUS_REGIONS:
        if pd.Timestamp(reg['start']) <= day <= pd.Timestamp(reg['end']):
            x0, x1 = reg['latitude']
            y0, y1 = reg['height']
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                         fill=False, edgecolor=reg['color'], linewidth=1.1, zorder=8))
    assert keep[(lat >= 20) & (lat < 40)].any()
    return ax, artist, int(keep.sum())


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
        'axes.linewidth': .6, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    core.linked.FOCUS_REGIONS = [dict(r, height=(52., r['height'][1]),
        end='2025-01-10' if tuple(r['band']) == (60, 80) else r['end'])
        for r in core.linked.FOCUS_REGIONS]
    bands = core.load_tables()[1:]
    lat, pressure, levels, fields, prior = daily.load_fields()
    fig = plt.figure(figsize=(W/2.54, H/2.54))
    panels = times(fig, bands)
    cmap = core.linked.reference_arrow_cmap()
    norm = PowerNorm(gamma=.5, vmin=prior['magnitude_limits'][0], vmax=prior['magnitude_limits'][1])
    counts = []
    for index, (day, q) in enumerate(zip(daily.MAP_DAYS[1:], fields[1:])):
        ax, artist, count = daily_map(fig, lat, levels, q, day, index, norm, cmap)
        panels.append(ax)
        counts.append(count)
    fig.text((MX[0] + (MX[-1] + MW - MX[0]) / 2) / W, .34 / H,
             'Latitude (°N)', ha='center', va='bottom', fontsize=7)
    bar(fig, artist, 17.05, MY[-1], MY[0]+MH-MY[-1], [0, 50, 150, 300, 550],
        '$|F|$ (m$^2$ s$^{-2}$)')
    fig.canvas.draw()
    for item in fig.findobj(match=Text):
        panel=(item.get_text() in tuple('abcdefghijklm') and item.get_fontweight()=='bold')
        item.set_fontsize(8 if panel else 7)
    fig.canvas.draw()
    assert len(panels) == 13
    path = OUT / 'Figure3_W1_mid_high_EP_20-80N.png'
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_name(path.stem + '_preview.png'), dpi=150)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)
    with Image.open(path) as im:
        assert abs(im.width-W/2.54*600) < 2 and abs(im.height-H/2.54*600) < 2
        pixels = im.size
    (OUT / 'figure_settings.json').write_text(json.dumps(dict(width_cm=W, height_cm=H,
        dpi=600, left_panels=['40–60 N W1 flux proxy', '40–60 N W1 amplitude', '60–80 N W1 flux proxy', '60–80 N W1 amplitude'],
        right_panels='three columns by three rows, chronological Jan 3–11',
        latitude_bands=['40–60 N', '60–80 N'], daily_latitude='20–80 N',
        original_data_and_calculations_preserved=True, arrow_counts=counts,
        arrow_display_vertical_factor=100*core.linked.DIRECTION_VERTICAL_BOOST,
        arrow_display_length='compressed, true magnitude shown by colour',
        source_hashes=core.HASHES), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(path), 'pixels': pixels, 'arrow_counts': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
