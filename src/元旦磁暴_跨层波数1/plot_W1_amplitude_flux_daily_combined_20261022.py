"""Four-row W1 amplitude, net heat-flux proxy and daily EP-direction figure.

The upper two rows use the existing all-MLS Dec27-Jan20 tables, with their
original height coordinates and diagnostics unchanged. The lower two rows
reuse the Jan3-Jan10 fields, selected native pressure levels, direction boost,
colour norm and highlighting from the original linked-reflection script.
No phase panels or W1-amplitude contours are included. PNG only.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, SymLogNorm, PowerNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PIL import Image

import plot_linked_reflection_mechanism_20261019 as linked

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'W1_amplitude_flux_daily_combined_Dec28_Jan11_20261022'
TABLES = ROOT / 'W1_three_bands_Dec27_Jan20_20261016'
REFERENCE = ROOT / 'winter_case_comparison_20261012/figures_2025_counterparts'
DAYS = pd.date_range('2024-12-28', '2025-01-11')
MAP_DAYS = pd.date_range('2025-01-03', '2025-01-10')
WIDTH, HEIGHT = 20., 23.40
TIME_X, TIME_W = [1.25, 7.45, 13.65], 4.95
TIME_Y, TIME_H = [17.90, 13.20], 3.50
MAP_X, MAP_W = [1.25, 5.75, 10.25, 14.75], 3.85
MAP_Y, MAP_H = [6.65, 1.05], 4.65
FONT = ['Arial', 'DejaVu Sans']
HASHES, TABLE_CHECKS, MAP_CHECKS, RECORDS, MAP_MASKS = {}, [], [], [], []


def digest(path):
    path = Path(path)
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def axis_at(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])


def load_tables():
    bands = []
    for index, (lo, hi) in enumerate(linked.BANDS):
        values = dict(lo=lo, hi=hi, amplitude_max=[450, 1800, 2500][index])
        for variable in ['amplitude', 'net_proxy']:
            name = f'{variable}_{lo}_{hi}_2025_MLS.csv'
            path = TABLES / name
            digest(path)
            all_days = pd.read_csv(path, index_col='date', parse_dates=True)
            assert DAYS.isin(all_days.index).all()
            frame = all_days.reindex(DAYS)
            old = pd.read_csv(REFERENCE / name, index_col='date', parse_dates=True)
            digest(REFERENCE / name)
            overlap = DAYS.intersection(old.index)
            np.testing.assert_allclose(frame.columns.astype(float), old.columns.astype(float),
                                       rtol=0, atol=1e-10)
            np.testing.assert_allclose(frame.loc[overlap], old.loc[overlap],
                                       rtol=0, atol=1e-8, equal_nan=True)
            values[variable] = frame.to_numpy()
            values[variable + '_height'] = frame.columns.to_numpy(float)
            TABLE_CHECKS.append(dict(band=f'{lo}-{hi}', variable=variable,
                overlap_days_unchanged=len(overlap),
                Jan11_finite_cells=int(np.isfinite(frame.loc['2025-01-11']).sum()),
                source=str(path), no_time_interpolation=True))
            for date, row in frame.iterrows():
                for height, number in row.items():
                    RECORDS.append(dict(date=str(date.date()), band=f'{lo}-{hi}',
                        variable=variable, reference_height_km=float(height), value=float(number)))
        bands.append(values)
    return bands


def time_style(ax, labels):
    ax.set_xlim(DAYS[0] - pd.Timedelta(hours=12), DAYS[-1] + pd.Timedelta(hours=12))
    ax.set_xticks(pd.date_range(DAYS[0], DAYS[-1], freq='2D'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelbottom=labels, labelsize=10, pad=2, length=3)
    if labels:
        ax.set_xlabel('UTC date', fontsize=12, labelpad=3)
    # Preserve the original calendar reference; do not relabel Jan 1 as the
    # measured onset of the geomagnetic storm.
    reference = ax.axvline(pd.Timestamp('2025-01-01'), color='black',
                           ls='--', lw=.85, zorder=6)
    reference.set_path_effects([pe.withStroke(linewidth=1.65, foreground='white'), pe.Normal()])


def colorbar(fig, image, x, y, height, ticks, heading, fontsize=9.5):
    cax = axis_at(fig, x, y, .18, height)
    bar = fig.colorbar(image, cax=cax, ticks=ticks, extend='neither')
    bar.ax.set_title(heading, fontsize=fontsize, fontweight='normal', pad=6)
    bar.ax.tick_params(direction='in', labelsize=9.5, pad=2, length=2.5)
    bar.formatter = FuncFormatter(lambda value, position: f'{value:g}')
    bar.update_ticks()
    return bar


def make_time_rows(fig, bands):
    axes = []
    for row in range(2):
        for col, band in enumerate(bands):
            ax = axis_at(fig, TIME_X[col], TIME_Y[row], TIME_W, TIME_H)
            letter = chr(97 + row * 3 + col)
            if row == 0:
                field, height = band['amplitude'], band['amplitude_height']
                norm, cmap, name = Normalize(0, band['amplitude_max']), 'viridis', 'W1 amplitude'
                ticks = [[0, 150, 300, 450], [0, 500, 1000, 1500, 1800],
                         [0, 500, 1000, 1500, 2000, 2500]][col]
                heading = 'm'
                limits, yticks = (20, 90), [20, 40, 50, 60, 80, 90]
            else:
                field, height = band['net_proxy'], band['net_proxy_height']
                # Same norm and diagnostic as Figure 1, not the separately
                # density-reduced EP-vector magnitude used for arrow colours.
                norm, cmap, name = SymLogNorm(1, vmin=-200, vmax=200), 'RdBu_r', 'Net W1 proxy'
                ticks, heading = [-200, -10, -1, 0, 1, 10, 200], 'K m s⁻¹'
                limits, yticks = (40, 90), [40, 50, 60, 70, 80, 90]
            image = ax.pcolormesh(linked.edges(mdates.date2num(DAYS)), linked.edges(height),
                                  np.ma.masked_invalid(field.T), norm=norm, cmap=cmap,
                                  shading='flat', zorder=1)
            ax.set(ylim=limits, yticks=yticks)
            ax.tick_params(labelleft=col == 0)
            if col == 0:
                ax.set_ylabel('approx. alt. (km)', fontsize=12, labelpad=3)
            ax.axhline(50, color='white' if row == 0 else '#555555',
                       ls='--' if row == 0 else ':', lw=.75, zorder=4)
            ax.set_title(f'({letter}) {name}\n{band["lo"]}–{band["hi"]}°N',
                         loc='left', fontsize=12, fontweight='normal', pad=5)
            time_style(ax, labels=row == 1)
            if row == 1:
                for region in linked.FOCUS_REGIONS:
                    if (band['lo'], band['hi']) != region['band']:
                        continue
                    start = mdates.date2num(pd.Timestamp(region['start'])) - .4
                    end = mdates.date2num(pd.Timestamp(region['end'])) + .4
                    bottom, top = region['height']
                    ax.add_patch(Rectangle((start, bottom), end - start, top - bottom,
                        fill=False, edgecolor=region['color'], linewidth=2, zorder=8))
            colorbar(fig, image, TIME_X[col] + TIME_W + .15,
                     TIME_Y[row], TIME_H, ticks, heading)
            axes.append(ax)
    return axes


def map_panel(fig, lat, pressure, levels, q, date, index, norm, cmap):
    row, col = divmod(index, 4)
    ax = axis_at(fig, MAP_X[col], MAP_Y[row], MAP_W, MAP_H)
    ax.set(xlim=(20, 80), ylim=(20, 90), xticks=[20, 40, 60, 80],
           yticks=[20, 40, 50, 60, 80, 90])
    ax.tick_params(direction='in', top=True, right=True, labelleft=col == 0,
                   labelsize=12, pad=2, length=3)
    if col == 0:
        ax.set_ylabel('approx. alt. (km)', fontsize=12, labelpad=3)
    if row == 1:
        ax.set_xlabel('Latitude (°N)', fontsize=12, labelpad=3)
    ax.set_title(f'({chr(103 + index)}) {date:%b} {date.day}',
                 loc='left', fontsize=12, fontweight='normal', pad=5)
    height = np.nanmedian(q['height'], axis=0)
    negative = np.where(np.isfinite(q['n2']) & (q['n2'] < 0), 1., np.nan)
    ax.pcolormesh(lat, height, negative.T, cmap=ListedColormap(['#6e6e6e']),
                  vmin=0, vmax=1, shading='auto', alpha=.42, zorder=.5)
    for key, color, style, width in [('n2', '#333333', '--', .7),
                                   ('background_qy', '#8c1d40', '-', 1.05)]:
        values = np.where(np.isfinite(q[key]), q[key], np.nan)
        if np.any(values < 0) and np.any(values > 0):
            ax.contour(lat, height, values.T, levels=[0], colors=[color],
                linewidths=width, linestyles=style, alpha=.9, zorder=.8)
    for latitude in [40, 60]:
        ax.axvline(latitude, color='#444444', ls='--', lw=.75, alpha=.65, zorder=1.2)
    ax.axhline(50, color='#222222', ls=':', lw=.85, zorder=1.2)
    options = dict(readable=True, uniform_heads=True,
        length_bounds_inches=linked.old.DENSE_LENGTH_BOUNDS,
        head_points=linked.old.DENSE_HEAD_POINTS, shaft_points=linked.old.DENSE_SHAFT_POINTS)
    artist, keep = linked.draw_arrows(ax, lat, q['height'], q['P'], q['W'],
                                     levels, cmap, norm, **options)
    boosted, keep_boosted = linked.draw_arrows(ax, lat, q['height'],
        q['P'] * keep.astype(float),
        q['W'] * linked.DIRECTION_VERTICAL_BOOST * keep.astype(float),
        levels, cmap, norm, **options)
    assert np.array_equal(keep, keep_boosted)
    linked.rotate_and_style_arrows(ax, artist, boosted, lat, q['height'], q['P'], q['W'],
                                   keep, highlight_downward=True, shaft_pts=1.05)
    # Confirm rotation has not changed original arrow colours/magnitudes. All
    # noncyan patches still carry the unboosted EP-vector magnitude colour.
    unboosted = q['magnitude'][keep]
    for patch, value, vertical, z in zip(artist.fixed_head_patches,
            unboosted, q['vert_reduced'][keep], q['height'][keep]):
        if not (vertical < 0 and z >= 48):
            np.testing.assert_allclose(patch.get_facecolor(), cmap(norm(value)), atol=1e-12)
    for region in linked.FOCUS_REGIONS:
        if pd.Timestamp(region['start']) <= date <= pd.Timestamp(region['end']):
            x0, x1 = region['latitude']
            z0, z1 = region['height']
            ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0,
                fill=False, edgecolor=region['color'], linewidth=2, zorder=8))
    MAP_MASKS.append(keep)
    MAP_CHECKS.append(dict(date=str(date.date()), panel=chr(103 + index),
        displayed_arrows=int(keep.sum()),
        highlighted_downward_arrows=int(((q['vert_reduced'] < 0) & (q['height'] >= 48) & keep).sum()),
        displayed_pressure_hPa=pressure[levels].tolist(),
        colour_from_true_unboosted_magnitude=True, fixed_head_points=linked.old.DENSE_HEAD_POINTS,
        display_length_bounds_inches=list(linked.old.DENSE_LENGTH_BOUNDS),
        display_vertical_factor=100 * linked.DIRECTION_VERTICAL_BOOST,
        arrows_are_not_wave_packet_trajectories=True))
    return ax, artist


def verify(fig, data_axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    decorations = list(fig.legends)
    for ax in fig.axes:
        decorations += [ax._left_title, ax.title, ax.xaxis.label, ax.yaxis.label]
    active = [x for x in decorations if not isinstance(x, Text) or x.get_text().strip()]
    overlaps = []
    for index, first in enumerate(active):
        for second in active[index + 1:]:
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                overlaps.append((str(first), str(second)))
    assert not overlaps, ('Decoration overlaps', overlaps)
    for ax in data_axes:
        assert ax._left_title.get_fontweight() == 'normal'
        low, high = sorted(ax.get_xlim())
        labels = [tick.label1 for tick in ax.xaxis.get_major_ticks()
                  if low <= tick.get_loc() <= high and tick.label1.get_visible()]
        for first, second in zip(labels[:-1], labels[1:]):
            assert not first.get_window_extent(renderer).overlaps(
                second.get_window_extent(renderer)), ('Tick overlap', ax.get_title(loc='left'),
                first.get_text(), second.get_text(), first.get_fontsize(),
                first.get_window_extent(renderer).bounds, second.get_window_extent(renderer).bounds)
    linked.validate_text_frame(fig)
    assert len(data_axes) == 14 and len(fig.axes) == 21
    return dict(clipped_text=[], decoration_overlaps=[], date_tick_overlaps=[],
                subfigure_titles_regular_weight=True)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': FONT, 'font.size': 12,
        'axes.titlesize': 12, 'axes.titleweight': 'normal', 'axes.labelsize': 12,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.linewidth': .7,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.facecolor': 'white'})
    bands = load_tables()
    lat, pressure, levels, all_fields, prior, _ = linked.load_daily_and_band_data()
    for path in [linked.SOURCE, linked.PREVIOUS, linked.PREVIOUS_SETTINGS]:
        digest(path)
    assert not levels[np.argmin(abs(pressure - .1))]
    fields = [all_fields[linked.DAYS.get_loc(date)] for date in MAP_DAYS]
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54))
    axes = make_time_rows(fig, bands)
    norm = PowerNorm(gamma=.5, vmin=prior['magnitude_limits'][0], vmax=prior['magnitude_limits'][1])
    cmap = linked.reference_arrow_cmap()
    for index, (date, q) in enumerate(zip(MAP_DAYS, fields)):
        ax, artist = map_panel(fig, lat, pressure, levels, q, date, index, norm, cmap)
        axes.append(ax)
    colorbar(fig, artist, 18.85, MAP_Y[1], MAP_Y[0] + MAP_H - MAP_Y[1],
             [0, 50, 150, 300, 550], '$|\\mathbf{F}|$\n(m² s⁻²)')
    fig.legend(handles=[
        Line2D([], [], color='#00e5ff', marker='>', markersize=7, lw=1.8,
            markeredgecolor='#002b49', label='Downward vector ($F_z < 0$)'),
        Patch(facecolor='#7a7a7a', alpha=.45, edgecolor='#404040',
              linestyle='--', label='$n^2 < 0$'),
        Line2D([], [], color='#8c1d40', lw=1.4, label='$\\bar q_y = 0$')],
        loc='center', bbox_to_anchor=(.5, 22.98 / HEIGHT), ncol=3,
        frameon=False, fontsize=11, handlelength=1.5, columnspacing=1.1,
        handletextpad=.4, borderpad=0, borderaxespad=0)
    for text in fig.findobj(Text):
        text.set_fontfamily(FONT)
    qa = verify(fig, axes)
    path = OUT / 'W1_amplitude_net_proxy_daily_EP_Dec28_Jan11.png'
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_name(path.stem + '_preview.png'), dpi=140)
    plt.close(fig)
    with Image.open(path) as item:
        pixels = list(item.size)
    assert abs(pixels[0] - WIDTH / 2.54 * 600) < 2
    assert abs(pixels[1] - HEIGHT / 2.54 * 600) < 2
    pd.DataFrame(RECORDS).to_csv(OUT / 'time_height_values.csv', index=False)
    np.savez_compressed(OUT / 'daily_EP_original_fields_and_display_masks.npz',
        dates=MAP_DAYS.values, latitude=lat, pressure_hPa=pressure,
        selected_pressure_indices=np.flatnonzero(levels), displayed_masks=np.stack(MAP_MASKS),
        **{key: np.stack([q[key] for q in fields]) for key in
           ['P', 'W', 'Q', 'height', 'magnitude', 'n2', 'background_qy']})
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
               for name, value in HASHES.items())
    checks = dict(figure=str(path), pixels=pixels, width_cm=WIDTH, height_cm=HEIGHT,
        dpi=600, PDF_export=False, layout='four rows: 3, 3, 4, 4',
        time_height_dates=[str(DAYS[0].date()), str(DAYS[-1].date())],
        daily_arrow_dates=[str(day.date()) for day in MAP_DAYS],
        date_labels='%b then %d, two lines, two-day step',
        time_height_data='existing all-MLS joint local-time fit tables; no ERA5 splice',
        amplitude_interpolation='only the existing complex-coefficient vertical display interpolation retained; no new interpolation',
        net_proxy='joint-fit geostrophic W1 heat-flux proxy in K m/s; not separated down-going EP flux',
        daily_EP='original paired-local-time branch diagnostic; same native pressure levels and colour normalization',
        January1_line='original calendar reference, not the measured storm commencement',
        unchanged_time_height_overlap=TABLE_CHECKS, unchanged_original_EP_fields=True,
        daily_arrow_checks=MAP_CHECKS, focus_regions=linked.FOCUS_REGIONS,
        focus_box_linewidth=2, W1_amplitude_contours_removed=True,
        phase_panels_removed=True, no_temporal_smoothing_or_missing_data_filling=True,
        arrow_magnitude_limits=prior['magnitude_limits'], original_sources_sha256=HASHES, **qa)
    (OUT / 'data_and_layout_checks.json').write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figure=str(path), pixels=pixels, layout=checks['layout'],
        QA=qa, Jan11_coverage=[x for x in TABLE_CHECKS],
        arrow_counts=[x['displayed_arrows'] for x in MAP_CHECKS]), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
