"""Keep the established upper rows and show Jan 2--11 as two-by-five maps.

Reuse the original native MLS coefficients and EP diagnostic. No new temporal
or vertical interpolation, background overlays, legend, white box outlines,
PDF output, or alteration of the upper-row data is introduced.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image, ImageChops

import plot_W1_amplitude_flux_daily_combined_20261022 as core
from daily_W1_EP_Jan03_Jan18_20261020.plot_daily_Jan03_Jan18 import (
    restyle_by_native_cell, direction_error,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'W1_amplitude_flux_daily_Jan02_Jan11_2x5_20261022'
BASELINE = ROOT / 'W1_amplitude_flux_daily_taller_upper_no_legend_Dec28_Jan11_20261022'
SOURCE = ROOT / 'W1_three_bands_Dec27_Jan20_20261016/MLS_W123_fullheight_Dec27_Jan20.npz'
MAP_DAYS = pd.date_range('2025-01-02', '2025-01-11')
DATE_TICKS = pd.date_range('2024-12-29', '2025-01-11', freq='3D')
COMMENCEMENT = pd.Timestamp('2024-12-31 16:18:00')
WIDTH, HEIGHT = 20., 22.7
MAP_X = [1.25 + 3.58 * col for col in range(5)]
MAP_W, MAP_H = 3.03, 4.65
MAP_Y = [6.65, 1.05]
CHECKS, MASKS = [], []


def load_fields():
    core.digest(SOURCE)
    core.digest(core.linked.SOURCE)
    core.digest(core.linked.PREVIOUS)
    core.digest(core.linked.PREVIOUS_SETTINGS)
    with np.load(SOURCE) as native:
        d = {key: native[key] for key in native.files}
    with np.load(core.linked.SOURCE) as original:
        dates = pd.DatetimeIndex(d['dates'])
        old_dates = pd.DatetimeIndex(original['dates'])
        overlap = MAP_DAYS.intersection(old_dates)
        assert len(overlap) == 9
        for key in ['cz', 'ct', 'z0', 't0', 'counts']:
            np.testing.assert_array_equal(
                np.take(d[key], dates.get_indexer(overlap), axis=2),
                np.take(original[key], old_dates.get_indexer(overlap), axis=2),
                err_msg=f'Changed original coefficients: {key}')
        np.testing.assert_array_equal(d['lat'], original['lat'])
        np.testing.assert_array_equal(d['pressure'], original['pressure'])
    assert (dates.get_indexer(MAP_DAYS) >= 0).all()
    lat, pressure = d['lat'], d['pressure']
    with np.load(core.linked.PREVIOUS) as frozen:
        levels = np.zeros(len(pressure), bool)
        levels[frozen['displayed_pressure_indices']] = True
    levels[np.argmin(abs(pressure - .1))] = False
    assert levels.sum() == 14
    prior = json.loads(core.linked.PREVIOUS_SETTINGS.read_text(encoding='utf-8'))
    z, t, z0, t0 = core.linked.extract(d, branch=-1, sector=0)
    fields = []
    for day in MAP_DAYS:
        i = dates.get_loc(day)
        q = core.linked.flux(z[i, ..., 0], t[i, ..., 0], z0[i], t0[i],
                             lat, pressure, wave=1, points=3)
        q['height'] = z0[i] / 1000
        q['mer_reduced'], q['vert_reduced'], q['magnitude'] = (
            core.linked.reduced_components(q['P'], q['W'], lat))
        # Retain native background diagnostics in the archive, but do not plot them.
        bg = core.linked.background(z0[i], t0[i], lat, pressure, points=5)
        q['n2'], q['background_qy'] = bg['n2'], bg['qy']
        fields.append(q)
    with np.load(BASELINE / 'daily_EP_original_fields_and_display_masks.npz') as old:
        old_dates = pd.DatetimeIndex(old['dates'])
        for day in old_dates:
            new_i, old_i = MAP_DAYS.get_loc(day), old_dates.get_loc(day)
            for key in ['P', 'W', 'Q', 'height', 'magnitude', 'n2', 'background_qy']:
                np.testing.assert_allclose(fields[new_i][key], old[key][old_i],
                                           rtol=1e-10, atol=1e-20, equal_nan=True,
                                           err_msg=f'{day}: changed {key}')
    return lat, pressure, levels, fields, prior


def upper_rows(fig, bands):
    axes = core.make_time_rows(fig, bands)
    for index, ax in enumerate(axes):
        ax.set_title(ax.get_title(loc='left').split('\n')[0],
                     loc='left', fontsize=12, fontweight='normal', pad=5)
        ax.set_xticks(DATE_TICKS)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if index < 3:
            ax.set_yticks(np.arange(20, 91, 10))
        for line in ax.lines:
            yy, xx = list(line.get_ydata(orig=True)), list(line.get_xdata(orig=True))
            if yy == [50, 50]:
                line.set(color='white', linestyle='--', linewidth=.75)
                line.set_path_effects([])
            if len(xx) == 2 and xx[0] == xx[1] == pd.Timestamp('2025-01-01'):
                line.set_xdata([COMMENCEMENT, COMMENCEMENT])
                line.set(color='white', linestyle='--', linewidth=.85)
                line.set_path_effects([])
    headers = [fig.text((x + core.TIME_W / 2) / WIDTH, .98,
        f'{band["lo"]}°–{band["hi"]}°N', ha='center', va='center',
        fontsize=14, fontweight='normal') for x, band in zip(core.TIME_X, bands)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_cm = 2.54 / fig.dpi
    top = max(ax._left_title.get_window_extent(renderer).y1 for ax in axes[:3]) * to_cm
    header_height = max(h.get_window_extent(renderer).height for h in headers) * to_cm
    for header in headers:
        header.set_y((top + .08 + header_height / 2) / HEIGHT)
    return axes, headers


def map_panel(fig, lat, pressure, levels, q, day, index, norm, cmap):
    row, col = divmod(index, 5)
    ax = core.axis_at(fig, MAP_X[col], MAP_Y[row], MAP_W, MAP_H)
    ax.set(xlim=(20, 80), ylim=(20, 90), xticks=[20, 40, 60, 80],
           yticks=np.arange(30, 91, 10))
    ax.tick_params(direction='in', top=True, right=True, labelleft=col == 0,
                   labelsize=12, pad=2, length=3)
    if col == 0:
        ax.set_ylabel('approx. alt. (km)', fontsize=12, labelpad=3)
    if row == 1:
        ax.set_xlabel('Latitude (°N)', fontsize=12, labelpad=3)
    ax.set_title(f'({chr(103 + index)}) {day:%b} {day.day}',
                 loc='left', fontsize=12, fontweight='normal', pad=5)
    options = dict(readable=True, uniform_heads=True,
        length_bounds_inches=core.linked.old.DENSE_LENGTH_BOUNDS,
        head_points=core.linked.old.DENSE_HEAD_POINTS,
        shaft_points=core.linked.old.DENSE_SHAFT_POINTS)
    artist, keep = core.linked.draw_arrows(ax, lat, q['height'], q['P'], q['W'],
                                          levels, cmap, norm, **options)
    boosted, boosted_keep = core.linked.draw_arrows(ax, lat, q['height'],
        q['P'] * keep.astype(float),
        q['W'] * core.linked.DIRECTION_VERTICAL_BOOST * keep.astype(float),
        levels, cmap, norm, **options)
    fallback = not np.array_equal(keep, boosted_keep)
    if fallback:
        restyle_by_native_cell(ax, artist, boosted, lat, q, keep)
    else:
        core.linked.rotate_and_style_arrows(ax, artist, boosted, lat, q['height'],
            q['P'], q['W'], keep, highlight_downward=True, shaft_pts=1.05)
    error = direction_error(ax, artist, q, keep)
    for patch, cell in zip(artist.fixed_head_patches, np.argwhere(keep)):
        i, j = cell
        downward = q['vert_reduced'][i, j] < 0 and q['height'][i, j] >= 48
        expected = '#00e5ff' if downward else cmap(norm(q['magnitude'][i, j]))
        np.testing.assert_allclose(patch.get_facecolor(), core.plt.matplotlib.colors.to_rgba(expected))
    for region in core.linked.FOCUS_REGIONS:
        if pd.Timestamp(region['start']) <= day <= pd.Timestamp(region['end']):
            x0, x1 = region['latitude']
            z0, z1 = region['height']
            ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False,
                edgecolor=region['color'], linewidth=2, zorder=8))
    assert len(ax.collections) == 0 and len(ax.lines) == 0
    MASKS.append(keep)
    CHECKS.append(dict(date=str(day.date()), panel=chr(103 + index),
        displayed_arrows=int(keep.sum()), boundary_mask_fallback=fallback,
        max_direction_error=error, background_overlays=0,
        displayed_pressure_hPa=pressure[levels].tolist()))
    return ax, artist


def validate(fig, axes, headers):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    core.linked.validate_text_frame(fig)
    assert len(axes) == 16 and len(fig.axes) == 23 and not fig.legends
    decorations = headers + [text for ax in fig.axes for text in
        [ax._left_title, ax.title, ax.xaxis.label, ax.yaxis.label] if text.get_text().strip()]
    for i, a in enumerate(decorations):
        for b in decorations[i + 1:]:
            assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), (
                'Decoration overlap', a.get_text(), b.get_text())
    # Check tick labels across adjacent panels as well as within each panel.
    for row_axes in [axes[:3], axes[3:6], axes[6:11], axes[11:16]]:
        labels = [tick.label1 for ax in row_axes for tick in ax.xaxis.get_major_ticks()
                  if tick.label1.get_visible() and ax.get_xlim()[0] <= tick.get_loc() <= ax.get_xlim()[1]]
        labels.sort(key=lambda label: label.get_window_extent(renderer).x0)
        for a, b in zip(labels, labels[1:]):
            assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), (
                'Tick overlap', a.get_text(), b.get_text())
    for ax in axes:
        assert ax._left_title.get_fontweight() == 'normal'
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                assert patch.get_linewidth() == 2 and patch.get_path_effects() == []
    return dict(clipped_text=[], decoration_overlaps=[], tick_overlaps=[],
                arrow_direction_and_colour_checks_passed=True)


def main():
    OUT.mkdir(exist_ok=True)
    core.HEIGHT, core.TIME_Y, core.TIME_H = HEIGHT, [17.70, 13.20], 3.80
    plt.rcParams.update({'font.family': core.FONT, 'font.size': 12,
        'axes.titlesize': 12, 'axes.titleweight': 'normal', 'axes.labelsize': 12,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.linewidth': .7,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.facecolor': 'white'})
    bands = core.load_tables()
    lat, pressure, levels, fields, prior = load_fields()
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54))
    axes, headers = upper_rows(fig, bands)
    norm = PowerNorm(gamma=.5, vmin=prior['magnitude_limits'][0],
                     vmax=prior['magnitude_limits'][1])
    cmap = core.linked.reference_arrow_cmap()
    for index, (day, q) in enumerate(zip(MAP_DAYS, fields)):
        ax, artist = map_panel(fig, lat, pressure, levels, q, day, index, norm, cmap)
        axes.append(ax)
    core.colorbar(fig, artist, 18.85, MAP_Y[1], MAP_Y[0] + MAP_H - MAP_Y[1],
                  [0, 50, 150, 300, 550], '$|\\mathbf{F}|$\n(m² s⁻²)')
    for text in fig.findobj(Text):
        text.set_fontfamily(core.FONT)
    qa = validate(fig, axes, headers)
    path = OUT / 'W1_amplitude_net_proxy_daily_EP_Dec28_Jan11_10days.png'
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_name(path.stem + '_preview.png'), dpi=140)
    plt.close(fig)
    with Image.open(path) as image:
        pixels = list(image.size)
        assert abs(image.width - WIDTH / 2.54 * 600) < 2
        assert abs(image.height - HEIGHT / 2.54 * 600) < 2
        with Image.open(BASELINE / 'W1_amplitude_net_proxy_daily_EP_Dec28_Jan11.png') as before:
            crop = (0, 0, image.width, round((HEIGHT - 12.) / 2.54 * 600))
            upper_pixels_unchanged = ImageChops.difference(
                before.convert('RGB').crop(crop), image.convert('RGB').crop(crop)).getbbox() is None
    core.pd.DataFrame(core.RECORDS).to_csv(OUT / 'time_height_values.csv', index=False)
    csv_name = 'time_height_values.csv'
    assert hashlib.sha256((OUT / csv_name).read_bytes()).digest() == hashlib.sha256(
        (BASELINE / csv_name).read_bytes()).digest()
    np.savez_compressed(OUT / 'daily_EP_original_fields_and_display_masks.npz',
        dates=MAP_DAYS.values, latitude=lat, pressure_hPa=pressure,
        selected_pressure_indices=np.flatnonzero(levels), displayed_masks=np.stack(MASKS),
        **{key: np.stack([q[key] for q in fields]) for key in
           ['P', 'W', 'Q', 'height', 'magnitude', 'n2', 'background_qy']})
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
               for name, value in core.HASHES.items())
    checks = dict(figure=str(path), pixels=pixels, width_cm=WIDTH, height_cm=HEIGHT,
        dpi=600, PDF_export=False, layout='four rows: 3, 3, 5, 5',
        arrow_dates=[str(day.date()) for day in MAP_DAYS],
        upper_rows_original_values_unchanged=True,
        upper_rows_pixels_identical_to_previous=upper_pixels_unchanged,
        original_Jan03_Jan10_EP_fields_identical=True,
        native_Jan02_Jan10_coefficients_identical=True,
        added_Jan11_source=str(SOURCE), native_levels_unchanged=True,
        no_missing_data_filling=True, temporal_or_vertical_interpolation_added=False,
        map_width_cm=MAP_W, map_height_cm=MAP_H, map_column_gap_cm=.55,
        map_x_positions_cm=MAP_X, map_y_positions_cm=MAP_Y,
        arrow_display_vertical_factor=400, arrow_magnitude_limits=prior['magnitude_limits'],
        arrow_colour_gamma=.5, arrow_length_mapping='existing nonlinear display compression',
        display_masks='recomputed only for narrower panel-edge clipping; no spatial subsampling',
        focus_boxes=core.linked.FOCUS_REGIONS, focus_box_linewidth_pt=2,
        focus_box_white_outline=False, background_overlays=False, top_legend=False,
        daily_panel_checks=CHECKS, sources_sha256=core.HASHES, **qa)
    (OUT / 'data_and_layout_checks.json').write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figure=str(path), pixels=pixels, layout=checks['layout'],
        upper_rows_pixels_identical_to_previous=upper_pixels_unchanged,
        arrow_counts=[c['displayed_arrows'] for c in CHECKS], QA=qa), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
