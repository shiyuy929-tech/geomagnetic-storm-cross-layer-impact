"""Six-panel Figure 3: thermal balance -> absolute wind -> PV attribution.

The spatial fourth column is omitted, not replaced with a simplified proof.
All calculation windows/operators are identical to the audited Figure 3.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from PIL import Image

import plot_mechanism_chain_barriers_20261022 as m


OUT = m.HERE / 'mechanism_chain_2x3_new_qy_outline_Dec28_Jan11_20261023'
WIDTH, HEIGHT = 20., 15.5
XS, WS = [1.10, 7.57, 14.37], [4.92, 4.92, 5.28]
YS, PANEL_H = [8.65, 1.60], 4.8
AXIS_LABEL_SIZE = 13
ORIGINAL_WIND_PANEL = m.wind_panel
ORIGINAL_PV_PANEL = m.pv_panel
ORIGINAL_ZERO_CONTOUR = m.zero_contour
ORIGINAL_CALENDAR = m.calendar
LT_ROWS = []
QY_SIGN_ROWS, QY_AUDIT, QY_BOUNDARY_ROWS = [], [], []
NEW_QY_FACE = 'none'
NEW_QY_EDGE = '#b51365'


def zero_contour(ax, x, z, values, color, linestyle='-', linewidth=1.):
    if color == m.MAGENTA:
        # Show only new negative native cells, not the full qy=0 contour.
        return None
    return ORIGINAL_ZERO_CONTOUR(ax, x, z, values, color, linestyle, linewidth)


def calendar(ax, row, events):
    ORIGINAL_CALENDAR(ax, row, events)
    for line in list(ax.lines):
        if line.get_gid() == 'storm_end':
            line.remove()
    ax.tick_params(axis='x', labelsize=12)


def panel_title(ax, letter, heading, subtitle):
    # Latitude is a shared row heading, not repeated in six panel titles.
    ax.set_title(f'({letter}) {heading}', loc='left', fontsize=14,
                 fontweight='normal', pad=7)


def row_headings(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    center_x = (XS[0] + XS[-1] + WS[-1]) / 2 / WIDTH
    headings = []
    for row, conf in enumerate(m.REGIONS.values()):
        top = max(ax._left_title.get_window_extent(renderer).y1
                  for ax in axes[3 * row:3 * row + 3]) / fig.dpi * 2.54
        headings.append(fig.text(center_x, (top + .10) / HEIGHT,
            f'{conf["band"][0]}–{conf["band"][1]}°N', ha='center', va='bottom',
            fontsize=13, fontweight='normal'))
    return headings


def wind_panel(*args):
    ax, data, conf, region, letter = args[:5]
    ax._qy_region, ax._qy_panel = region, letter
    image = ORIGINAL_WIND_PANEL(*args)
    for patch in list(args[0].patches):
        patch.remove()
    qy, height = m.band_mean(data, 'qy', conf['band'])
    valid = np.isfinite(qy).all(0) & (height >= 45)
    baseline_known = np.isfinite(qy[m.PRE]).all(0)
    pre_nonnegative = baseline_known & (qy[m.PRE] >= 0).all(0)
    # Highlight changes only in the independently selected response layers.
    # Daily means on the onset date mix pre- and post-onset samples; start
    # highlighting at the first complete UTC day after the event timestamp.
    onset = pd.Timestamp(args[7]['storm_start']['time_UTC'])
    after_onset = m.DAYS > onset.normalize()
    lower, upper = conf['attribution_layer']
    focus = valid & (height >= lower) & (height <= upper)
    newly_negative = (after_onset[:, None] & focus[None, :]
                      & pre_nonnegative[None, :] & (qy < 0))
    xedges = m.md.date2num(m.DAYS.append(
        pd.DatetimeIndex([m.DAYS[-1] + pd.Timedelta(days=1)])))
    zedges = m.base.edges(height)
    cells = []
    for it, j in zip(*np.where(newly_negative)):
        cells.append([(xedges[it], zedges[j]), (xedges[it + 1], zedges[j]),
                      (xedges[it + 1], zedges[j + 1]), (xedges[it], zedges[j + 1])])
    if cells:
        overlay = PolyCollection(cells, facecolors=[NEW_QY_FACE],
            edgecolors=[NEW_QY_EDGE], linewidths=0, hatch='///', zorder=3)
        overlay.set_gid('newly-negative-qy-focus')
        ax.add_collection(overlay)
    # Trace only exposed cell edges: adjacent marked cells form one region
    # without internal grid boxes, interpolation, smoothing, or a white halo.
    boundary = []
    for it, j in zip(*np.where(newly_negative)):
        x0, x1, z0, z1 = xedges[it], xedges[it + 1], zedges[j], zedges[j + 1]
        if it == 0 or not newly_negative[it - 1, j]:
            boundary.append([(x0, z0), (x0, z1)])
        if it == newly_negative.shape[0] - 1 or not newly_negative[it + 1, j]:
            boundary.append([(x1, z0), (x1, z1)])
        if j == 0 or not newly_negative[it, j - 1]:
            boundary.append([(x0, z0), (x1, z0)])
        if j == newly_negative.shape[1] - 1 or not newly_negative[it, j + 1]:
            boundary.append([(x0, z1), (x1, z1)])
    if boundary:
        outline = LineCollection(boundary, colors=NEW_QY_EDGE, linewidths=1.15,
            capstyle='butt', joinstyle='miter', zorder=5)
        outline.set_gid('newly-negative-qy-outline')
        ax.add_collection(outline)
    for segment, endpoints in enumerate(boundary):
        for vertex, (date_number, z) in enumerate(endpoints):
            QY_BOUNDARY_ROWS.append(dict(region=region, panel=letter,
                segment=segment, vertex=vertex,
                time_UTC=m.md.num2date(date_number).isoformat(),
                reference_height_km=float(z)))
    for j in np.flatnonzero(valid):
        for it, day in enumerate(m.DAYS):
            QY_SIGN_ROWS.append(dict(region=region, panel=letter,
                date=str(day.date()), pressure_hPa=float(data['pressure_hPa'][j]),
                reference_height_km=float(height[j]),
                absolute_qy_1e11=float(qy[it, j] * 1e11),
                negative=bool(qy[it, j] < 0),
                all_three_baseline_days_nonnegative=bool(pre_nonnegative[j]),
                newly_negative_band_mean=bool(pre_nonnegative[j] and qy[it, j] < 0),
                highlighted_new_negative_cell=bool(newly_negative[it, j])))
    QY_AUDIT.append(dict(region=region, panel=letter,
        latitude_band=list(conf['band']), absolute_total_qy=True,
        averaging='cosine-latitude weighted mean on native pressure levels',
        valid_height_range_km=[float(height[valid].min()), float(height[valid].max())],
        qy_zero_contour_displayed=False,
        highlight_style='transparent magenta hatching and exposed native-cell boundary only',
        highlight_facecolor=NEW_QY_FACE, outline_color=NEW_QY_EDGE,
        outline_linewidth_pt=1.15, boundary_segment_count=len(boundary),
        no_vertical_extrapolation=True, no_smoothing=True,
        highlighted_cell_count=int(newly_negative.sum()),
        highlight_focus_centre_height_km=[lower, upper],
        highlight_first_date=str(m.DAYS[after_onset][0].date()),
        highlight_rule='all three PRE days qy>=0; full post-onset UTC day qy<0; valid native levels in the response-layer focus',
        highlight_geometry='native daily cells and pressure-level midpoint edges; no extrapolation or fitted boundary',
        pre_mean_qy_min_1e11=float(np.min(qy[m.PRE].mean(0)[valid]) * 1e11),
        post_mean_qy_min_1e11=float(np.min(qy[m.POST].mean(0)[valid]) * 1e11)))
    return image


def pv_panel(ax, data, conf, region, letter):
    ORIGINAL_PV_PANEL(ax, data, conf, region, letter)
    for artist in list(ax.collections):
        if isinstance(artist, PolyCollection):
            artist.remove()
    ax.set_xlabel('Δqᵧ (10⁻¹¹ m⁻¹ s⁻¹)', fontsize=12, labelpad=3)
    for branch, setting in enumerate(data['LT']):
        q, height = m.band_mean(dict(data, qy=setting['qy']), 'qy', conf['band'])
        delta = (q[m.POST].mean(0) - q[m.PRE].mean(0)) * 1e11
        for j in np.flatnonzero(np.isfinite(delta) & (height >= 40)):
            LT_ROWS.append(dict(region=region, branch=branch,
                pressure_hPa=float(data['pressure_hPa'][j]),
                reference_height_km=float(height[j]), delta_qy_1e11=float(delta[j])))


def legend(fig):
    handles = [Line2D([], [], color=m.PURPLE, lw=1.1, label='$U_g = 0$'),
        Patch(facecolor=NEW_QY_FACE, edgecolor=NEW_QY_EDGE, lw=1.15, hatch='///',
              label=r'New $\overline{q}_y < 0$'),
        Line2D([], [], color=m.DARK, lw=1.5, marker='o', ms=3, label='Total Δqᵧ'),
        Line2D([], [], color=m.BLUE, lw=1.5, ls='--', label='Meridional Δqᵧ'),
        Line2D([], [], color=m.RED, lw=1.5, ls='-.', label='Vertical Δqᵧ')]
    return fig.legend(handles=handles, loc='upper center',
        bbox_to_anchor=(.515, (HEIGHT - .13) / HEIGHT), ncol=5, frameon=False,
        fontsize=13, handlelength=1.25, handletextpad=.3, columnspacing=.5,
        labelspacing=.3, borderpad=0, borderaxespad=0)


def verify(fig, axes, bars, key, headings):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
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
    outside, tick_collisions = [], []
    for text in fig.findobj(Text):
        if not text.get_visible() or text in inactive or not text.get_text().strip():
            continue
        assert not re.search(r'[\u3400-\u9fff]', text.get_text())
        b = text.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5:
            outside.append(text.get_text())
    assert not outside, ('Clipped text', outside)
    for ax in axes:
        assert ax.xaxis.label.get_fontsize() == AXIS_LABEL_SIZE
        assert ax.yaxis.label.get_fontsize() == AXIS_LABEL_SIZE
        low, high = ax.get_xlim()
        ticks = [t.label1 for t in ax.xaxis.get_major_ticks()
                 if low <= t.get_loc() <= high and t.label1.get_visible()]
        for a, b in zip(ticks[:-1], ticks[1:]):
            if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)):
                tick_collisions.append((ax.get_title(loc='left'), a.get_text(), b.get_text()))
    assert not tick_collisions, ('Tick overlap', tick_collisions)
    decorations = [key, *headings]
    for ax in axes:
        decorations.extend([ax._left_title, ax.xaxis.label, ax.yaxis.label])
    for bar in bars:
        decorations.extend([bar.ax.title, bar.ax.yaxis.label])
        unit = bar.ax.yaxis.label.get_window_extent(renderer)
        for tick in bar.ax.get_yticklabels():
            assert not unit.overlaps(tick.get_window_extent(renderer)), 'Colorbar unit overlap'
    for i, a in enumerate(decorations):
        if isinstance(a, Text) and not a.get_text():
            continue
        for b in decorations[i + 1:]:
            if isinstance(b, Text) and not b.get_text():
                continue
            assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)), \
                ('Decoration overlap', str(a), str(b))
    for col in range(3):
        below = axes[3 + col]._left_title.get_window_extent(renderer)
        above = axes[col].xaxis.label.get_window_extent(renderer)
        if axes[col].get_xlabel():
            assert below.y1 < above.y0
        for tick in axes[col].get_xticklabels():
            if tick.get_visible():
                assert not below.overlaps(tick.get_window_extent(renderer)), 'Row/date overlap'
        row_heading = headings[1].get_window_extent(renderer)
        if axes[col].get_xlabel():
            assert not row_heading.overlaps(above), 'Latitude/upper xlabel overlap'
        for tick in axes[col].get_xticklabels():
            if tick.get_visible():
                assert not row_heading.overlaps(tick.get_window_extent(renderer)), 'Latitude/date overlap'
    heading_bounds = []
    for row, heading in enumerate(headings):
        b = heading.get_window_extent(renderer)
        for ax in axes[3 * row:3 * row + 3]:
            assert b.y0 > ax._left_title.get_window_extent(renderer).y1, 'Row latitude below panel title'
        heading_bounds.append((np.array(b.bounds) * 2.54 / fig.dpi).tolist())
    legend_gap = (key.get_window_extent(renderer).y0
                  - headings[0].get_window_extent(renderer).y1) * 2.54 / fig.dpi
    assert 0 <= legend_gap <= .4, ('Legend/row-heading gap', legend_gap)
    return dict(clipped_text=outside, tick_overlap=tick_collisions,
        decoration_overlap=False, six_panels=True, regular_titles=True,
        profile_own_y_ticks_due_to_lower_native_top=True,
        shared_row_latitude_bounds_cm=heading_bounds, legend_to_row_heading_gap_cm=legend_gap,
        subplot_height_cm=PANEL_H, row_axes_gap_cm=YS[0] - YS[1] - PANEL_H,
        panel_title_fontsize=14, panel_title_pad_pt=7, colorbar_unit_fontsize=11.5,
        x_y_axis_title_fontsize=AXIS_LABEL_SIZE,
        column_axes_gaps_cm=[XS[i+1]-XS[i]-WS[i] for i in range(2)],
        legend_fontsize=13)


def main():
    OUT.mkdir(exist_ok=True)
    m.WIDTH, m.HEIGHT, m.PANEL_H = WIDTH, HEIGHT, PANEL_H
    m.zero_contour = zero_contour
    m.calendar = calendar
    m.title = panel_title
    m.response_time_box = lambda ax, conf: None
    plt.rcParams.update({'font.family': m.FONT, 'font.size': 12, 'axes.labelsize': AXIS_LABEL_SIZE,
        'axes.titlesize': 12, 'axes.titleweight': 'normal', 'axes.linewidth': .65,
        'hatch.linewidth': .65,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'savefig.facecolor': 'white'})
    events = json.loads(m.EVENTS_PATH.read_text(encoding='utf-8'))
    data = m.load()
    masks = m.n2_masks(data)
    fig = plt.figure(figsize=(WIDTH / 2.54, HEIGHT / 2.54))
    axes = [m.axes_at(fig, x, y, w, PANEL_H) for y in YS for x, w in zip(XS, WS)]
    bars, manifest = [], []
    for row, (region, conf) in enumerate(m.REGIONS.items()):
        a, b, c = axes[3 * row:3 * row + 3]
        letters = [chr(97 + 3 * row + j) for j in range(3)]
        gi, top = m.gradient_panel(a, data, conf, region, letters[0], row, events)
        ui = wind_panel(b, data, conf, region, letters[1], row, top, events, masks)
        pv_panel(c, data, conf, region, letters[2])
        bars.extend([m.colorbar(fig, gi, XS[0]+WS[0]+.16, YS[row], 'ΔTᵧ', 'K / 1000 km', [-12, 0, 12]),
                     m.colorbar(fig, ui, XS[1]+WS[1]+.16, YS[row], '$U_g$', 'm s⁻¹', [-40, 0, 40, 80])])
        for bar in bars[-2:]:
            bar.ax.yaxis.label.set_fontsize(11.5)
        for letter, meaning in zip(letters, ['thermal_gradient_and_shear_consistency',
                'absolute_background_wind_evolution', 'PV_gradient_term_attribution']):
            manifest.append(dict(panel=letter, region=region, diagnostic=meaning,
                latitude_band=f'{conf["band"][0]}-{conf["band"][1]}N'))
    for ax in axes:
        ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    headings = row_headings(fig, axes)
    key = legend(fig)
    geometry = verify(fig, axes, bars, key, headings)
    png = OUT / 'Figure3_mechanism_chain_2x3_new_qy_outline_Dec28_Jan11.png'
    fig.savefig(png, dpi=600)
    fig.savefig(png.with_name(png.stem + '_preview.png'), dpi=160)
    plt.close(fig)
    with Image.open(png) as im:
        pixels = list(im.size)
    assert abs(pixels[0] - WIDTH / 2.54 * 600) < 2
    for name, rows in [('plotted_daily_diagnostics.csv', m.DAILY_ROWS),
        ('qy_contributions.csv', m.PROFILE_ROWS), ('native_coverage.csv', m.COVERAGE),
        ('new_n2_calculation_audit_not_plotted.csv', m.NEW_ROWS),
        ('diagnostic_metrics.csv', m.METRICS), ('LT_profile_sensitivity.csv', LT_ROWS),
        ('new_qy_native_boundary_vertices.csv', QY_BOUNDARY_ROWS),
        ('absolute_qy_sign_audit.csv', QY_SIGN_ROWS),
        ('panel_manifest.csv', manifest)]:
        frame = pd.DataFrame(rows)
        if name == 'native_coverage.csv':
            frame = frame.drop_duplicates()
        frame.to_csv(OUT / name, index=False)
    assert all(hashlib.sha256(m.Path(path).read_bytes()).hexdigest() == digest
               for path, digest in m.base.HASHES.items())
    checks = dict(figure=str(png), pixels=pixels, width_cm=WIDTH, height_cm=HEIGHT, dpi=600,
        period_display=['2024-12-28', '2025-01-11'], baseline=['2024-12-28', '2024-12-30'],
        comparison=['2025-01-06', '2025-01-09'], baseline_days=3, comparison_days=4,
        comparison_definition='difference of arithmetic daily means; not sums',
        column_order=['temperature_gradient_and_shear', 'absolute_Ug_evolution', 'Delta_qy_attribution'],
        order_rationale='show diagnosed wind change before its meridional/vertical PV-gradient consequences',
        no_absolute_wind_integration_from_thermal_wind=True,
        shared_latitude_labels='one centred label above each row; no repeated panel latitude subtitles',
        subplot_height_cm=PANEL_H, previous_subplot_height_cm=4.,
        daily_ticks='daily cell centres12UTC; labels every3days fromDec29; font12',
        latitude_support='40-60N; high-latitude complete background60-70N, not60-80N',
        native_operator='unchanged5-point geostrophic background and additive qy terms',
        native_qy_top='profile ends near85km; not extrapolated to wind/temperature topnear94km',
        absolute_qy_overlay=QY_AUDIT,
        newly_negative_qy_interpretation='observed post-onset sign change relative to three baseline days; no unique storm-causation claim',
        omitted_from_main=['spatial fourth column', 'EP arrows', 'new n2 marks', 'qy zero contours',
                           'response boxes', 'LT branch range envelope'],
        LT_sensitivity='numeric branch comparison in separate CSV; not a confidence interval',
        no_independent_causal_closure=True, no_reflection_claim_from_zero_wind_alone=True,
        thermal_wind_is_shared_input_consistency=True,
        geometry_checks=geometry, panel_manifest=manifest,
        storm_start=events['storm_start'], storm_end=events['storm_end'],
        displayed_event_markers=['storm_start'],storm_end_marker_displayed=False,
        input_sha256=m.base.HASHES, PDF_export=False)
    (OUT / 'data_and_layout_checks.json').write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(figure=str(png), pixels=pixels, geometry=geometry),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
