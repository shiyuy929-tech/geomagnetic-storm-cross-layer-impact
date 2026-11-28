"""Five-panel Figure5: actual wind/temperature above three weather maps.

The averaging box is user-selected110--70W,40--60N. Panel(b) uses the same
longitude interval with its latitude axis retained at25--80N. No causal
stratospheric attribution is inferred from this plot.
"""
from pathlib import Path
import sys, json, hashlib

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'plot_packages'), str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from PIL import Image
import analyze_upper_vortex_followup_20261029 as base
import plot_figure5_w2w3_troposphere_impact as weather

OUT = ROOT/'results/Figure5_snow_region_actual_20261031'
WIDTH, HEIGHT = 20., 18.55
WIND_LIMIT = 15.
ANALYSIS_BOX = (-110., -70., 40., 60.)
DAYS = pd.date_range('2024-12-28', '2025-01-11')
PLOT_DAYS = DAYS + pd.Timedelta(hours=12)
CHECKS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_box(stations):
    assert len(stations) and (stations.snowfall_accumulation_mm > 0).all()
    raw = [float(stations.longitude.min()), float(stations.longitude.max()),
           float(stations.latitude.min()), float(stations.latitude.max())]
    box = ANALYSIS_BOX
    inside = stations.longitude.between(box[0], box[1]) & stations.latitude.between(box[2], box[3])
    CHECKS.update(snow_station_count=len(stations), positive_snow_station_bounds=raw,
        panel_a_box=list(box), panel_b_longitudes=list(box[:2]),
        box_rule='user-selected110--70W,40--60N; same longitudes in panels(a,b)',
        snow_stations_inside_panel_a_box_count=int(inside.sum()),
        box_is_official_NOAA_polygon=False)
    return box


def weighted_mean(a, weights):
    good = np.isfinite(a)
    denominator = np.sum(good*weights, axis=(-2, -1))
    return np.divide(np.sum(np.where(good, a, 0)*weights, axis=(-2, -1)), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 0)


def analyse(box):
    base.START = np.datetime64('2024-12-28')
    base.END = np.datetime64('2025-01-11')
    times, days, p, lat, lon, fields = base.load()
    np.testing.assert_array_equal(days, DAYS.values.astype('datetime64[D]'))
    assert len(times) == 4*len(DAYS)
    sp = base.read_sp(times, lat, lon)
    above_ground = sp[:, None] >= p[None, :, None, None]*100
    v = np.where(above_ground, fields['v'], np.nan).astype(float)
    fixed = np.isfinite(v).all(axis=0)
    region = base.roi(lat, lon, box)
    area = np.cos(np.deg2rad(lat))[:, None]*region
    weights = area[None]*fixed
    actual_v = base.daily(weighted_mean(v, weights[None]))
    coverage = weights.sum(axis=(-2, -1))/area.sum()

    j850 = int(np.flatnonzero(p == 850)[0])
    t850 = np.where(above_ground[:, j850], fields['t'][:, j850], np.nan).astype(float)
    v850 = v[:, j850]
    lon_mask = (lon >= box[0]) & (lon <= box[1])
    support = np.isfinite(t850).all(0) & np.isfinite(v850).all(0) & lon_mask[None]
    den = support.sum(-1)
    def zonal_mean(a):
        return np.divide(np.sum(np.where(support, a, 0), axis=-1), den,
                         out=np.full(a.shape[:2], np.nan), where=den > 0)
    yi = np.flatnonzero((lat >= 25) & (lat <= 80))[::-1]
    temperature = base.daily(zonal_mean(t850))[:, yi]-273.15
    hov_v = base.daily(zonal_mean(v850))[:, yi]
    assert np.isfinite(temperature).all() and np.isfinite(hov_v).all()
    result = dict(time=days, pressure_hPa=p, actual_V=actual_v,
        pressure_fixed_area_fraction=coverage, hov_latitude=lat[yi],
        T850_C=temperature, V850=hov_v,
        hov_fixed_longitude_fraction=(den/lon_mask.sum())[yi])
    np.savez_compressed(OUT/'snow_region_actual_fields.npz', **result)
    rows = [dict(date=str(day), pressure_hPa=float(pp), V_m_s=actual_v[i, j],
                 fixed_area_fraction=coverage[j])
            for i, day in enumerate(days) for j, pp in enumerate(p)]
    pd.DataFrame(rows).to_csv(OUT/'actual_meridional_wind_daily.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([dict(date=str(day), latitude_N=float(yy), T850_C=temperature[i, j], V850_m_s=hov_v[i, j])
        for i, day in enumerate(days) for j, yy in enumerate(lat[yi])]).to_csv(
            OUT/'actual_temperature_and_wind_Hovmoller.csv', index=False, encoding='utf-8-sig')
    CHECKS.update(native_grid_degrees=2.5, hours_between_samples=6,
        native_time_UTC=[str(times[0]), str(times[-1])], pressure_hPa=p.tolist(),
        panels_a_b_use_actual_values=True, baseline_subtraction_in_panels_a_b=False,
        mask_subterranean_samples_before_average=True, fixed_support_over_entire_period=True,
        panel_a_pressure_fixed_area_fraction=coverage.tolist(),
        actual_V_range_m_s=[float(np.nanmin(actual_v)), float(np.nanmax(actual_v))],
        panel_a_colour_range=[-WIND_LIMIT, WIND_LIMIT],
        panel_a_fraction_outside_colour_range=float(np.mean(abs(actual_v) > WIND_LIMIT)),
        actual_T850_range_C=[float(temperature.min()), float(temperature.max())],
        source_sha256={str(path): digest(path) for path in dict.fromkeys(base.INPUTS)})
    return result


def heading(ax, title, subtitle, single_line=False):
    separator = ' | ' if single_line else '\n'
    ax.set_title(title+separator+subtitle, loc='left', fontsize=13.5, fontweight='normal', pad=7)


def colourbar(fig, ax, artist, ticks, unit, horizontal=False):
    position = ax.get_position()
    if horizontal:
        cax = fig.add_axes([position.x0, position.y0-.90/HEIGHT, position.width, .20/HEIGHT])
        cb = fig.colorbar(artist, cax=cax, ticks=ticks, extend='neither', orientation='horizontal')
        cb.set_label(unit, fontsize=11.5, labelpad=2)
    else:
        cax = fig.add_axes([position.x1+.35/WIDTH, position.y0, .18/WIDTH, position.height])
        cb = fig.colorbar(artist, cax=cax, ticks=ticks, extend='neither')
        cb.ax.set_title(unit, fontsize=12, pad=9)
    cb.ax.tick_params(direction='in', labelsize=11.5, length=2.7, pad=2)
    return cb, cax


def plot(data, maps, box):
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'], 'font.size':13,
        'axes.labelsize':13, 'axes.titlesize':13.5, 'axes.titleweight':'normal',
        'xtick.labelsize':13, 'ytick.labelsize':13, 'axes.linewidth':.75,
        'xtick.direction':'in', 'ytick.direction':'in'})
    fig = plt.figure(figsize=(WIDTH/2.54, HEIGHT/2.54))
    def axes(x, y, w, h, **kwargs):
        return fig.add_axes([x/WIDTH, y/HEIGHT, w/WIDTH, h/HEIGHT], **kwargs)
    a = axes(1.7, 14.45, 16.75, 3.2)
    b = axes(1.7, 10.45, 16.75, 3.2, sharex=a)
    bar_pairs = []
    xe = mdates.date2num(pd.date_range(DAYS[0], DAYS[-1]+pd.Timedelta(days=1)))

    im = a.pcolormesh(xe, weather.edges(np.log(data['pressure_hPa'])), data['actual_V'].T,
        cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-WIND_LIMIT, vmax=WIND_LIMIT), shading='flat')
    a.set_ylim(np.log(1000), np.log(1))
    ticks = np.array([1, 3, 10, 30, 100, 300, 1000])
    a.set_yticks(np.log(ticks)); a.set_yticklabels([f'{x:g}' for x in ticks])
    a.set_ylabel('pressure (hPa)', labelpad=2)
    heading(a, '(a) Meridional wind', f'{abs(box[1]):g}–{abs(box[0]):g}°W, {box[2]:g}–{box[3]:g}°N', single_line=True)
    _, ca = colourbar(fig, a, im, [-15, -10, -5, 0, 5, 10, 15], r'm s$^{-1}$')
    bar_pairs.append((a, ca))

    low = min(-5., 5*np.floor(data['T850_C'].min()/5))
    high = max(5., 5*np.ceil(data['T850_C'].max()/5))
    im = b.pcolormesh(xe, weather.edges(data['hov_latitude']), data['T850_C'].T,
        cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=low, vmax=high), shading='flat')
    b.set_ylim(25, 80); b.set_yticks([30, 40, 50, 60, 70, 80])
    b.set_ylabel('Latitude (°N)', labelpad=4)
    contours = b.contour(PLOT_DAYS, data['hov_latitude'], data['T850_C'].T,
        levels=[-10, 0], colors=['#444444','#222222'], linestyles=['--','-'], linewidths=[.75,.85])
    b.clabel(contours, fmt={-10:'−10°C', 0:'0°C'}, fontsize=11.5, inline=True, inline_spacing=3)
    yi = np.array([int(np.flatnonzero(data['hov_latitude'] == x)[0]) for x in [30,40,50,60,70]])
    ti = np.arange(len(PLOT_DAYS))
    xx, yy = np.meshgrid(PLOT_DAYS[ti].to_pydatetime(), data['hov_latitude'][yi])
    qu = b.quiver(xx, yy, np.zeros(xx.shape, dtype=float), data['V850'][np.ix_(ti, yi)].T,
        angles='uv', scale_units='inches', scale=60, width=.0015, headwidth=3.5,
        color='#222222', zorder=8)
    b.quiverkey(qu, .80, 1.055, 10, r'10 m s$^{-1}$', labelpos='E', coordinates='axes',
        fontproperties={'size':12})
    heading(b, '(b) Cold-air evolution', f'850 hPa | {abs(box[1]):g}–{abs(box[0]):g}°W', single_line=True)
    cticks = sorted(set([low, float(10*np.ceil(low/20)), 0., float(10*np.floor(high/20)), high]))
    _, ca = colourbar(fig, b, im, cticks, '°C')
    bar_pairs.append((b, ca))
    for ax in [a, b]:
        ax.set_xlim(DAYS[0], DAYS[-1]+pd.Timedelta(days=1))
        ax.set_xticks(pd.date_range('2024-12-29','2025-01-10',freq='3D')+pd.Timedelta(hours=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(direction='in', top=True, right=True, length=3, pad=3)
    a.tick_params(labelbottom=False)
    b.set_xlabel('UTC date | 2024–2025', labelpad=4)
    assert a.get_shared_x_axes().joined(a, b)

    c = axes(1.7, 2.8, 5.1, 5.2, projection=weather.PC)
    d = axes(7.8, 2.8, 5.1, 5.2, projection=weather.PC)
    e = axes(13.9, 2.8, 5.1, 5.2, projection=weather.PC)
    for ax, left in [(c,True), (d,False), (e,False)]:
        weather.map_style(ax, left); ax.tick_params(labelsize=13)
    im = c.pcolormesh(maps['lon'], maps['lat'], maps['dz500'], cmap='RdBu_r',
        norm=TwoSlopeNorm(vcenter=0,vmin=-240,vmax=240), shading='auto', transform=weather.PC, zorder=2)
    c.plot(maps['lon'], maps['edge_pre'], color='#555555', ls='--', lw=1.3, transform=weather.PC, zorder=8)
    c.plot(maps['lon'], maps['edge_main'], color='#7b3294', lw=1.7, transform=weather.PC, zorder=9)
    heading(c, '(c) Trough and envelope', 'Jan 7–9 | 500 hPa')
    _, ca = colourbar(fig, c, im, [-240,-120,0,120,240], r'$\Delta Z_{500}$ (m)', True)
    bar_pairs.append((c, ca))
    c.legend(handles=[Line2D([],[],color='#555555',ls='--',lw=1.3,label='Dec 28–30'),
                      Line2D([],[],color='#7b3294',lw=1.7,label='Jan 7–9')],
        loc='upper center', bbox_to_anchor=(.5,-.42), ncol=2, frameon=False, fontsize=11.5,
        handlelength=1.2, handletextpad=.25, columnspacing=.55, borderaxespad=0, borderpad=0)
    im = d.pcolormesh(maps['lon'], maps['lat'], maps['adv850'], cmap='RdBu_r',
        norm=TwoSlopeNorm(vcenter=0,vmin=-10,vmax=10), shading='auto', transform=weather.PC, zorder=2)
    iy = np.flatnonzero((maps['lat'] >= 22) & (maps['lat'] <= 82))[::2]
    ix = np.flatnonzero((maps['lon'] >= -140) & (maps['lon'] <= -50))[::2]
    qu = d.quiver(maps['lon'][ix], maps['lat'][iy], maps['u850'][np.ix_(iy,ix)],
        maps['v850'][np.ix_(iy,ix)], transform=weather.PC, scale=180, width=.0034,
        headwidth=3.4, color='#222222', zorder=8)
    d.quiverkey(qu,.3,-.46,10,r'10 m s$^{-1}$',labelpos='E',coordinates='axes',fontproperties={'size':12})
    heading(d, '(d) Cold-air transport', 'Jan 7–9 | 850 hPa')
    _, ca = colourbar(fig, d, im, [-10,-5,0,5,10], r'$-\mathbf{V}\cdot\nabla T$ (K day$^{-1}$)', True)
    bar_pairs.append((d, ca))
    im = e.pcolormesh(maps['s_lon'], maps['s_lat'], maps['surface'], cmap='RdBu_r',
        norm=TwoSlopeNorm(vcenter=0,vmin=-12,vmax=12), shading='auto', transform=weather.PC, zorder=2)
    snow = e.scatter(maps['snow'].longitude, maps['snow'].latitude, s=20, marker=(6,2,0),
        color='#f7f9fa', linewidths=.6, transform=weather.PC, zorder=12)
    snow.set_path_effects([pe.Stroke(linewidth=1.15,foreground='#50606b'),pe.Normal()])
    heading(e, '(e) Surface cold & snow', 'Jan 9–11 | 2 m temperature')
    _, ca = colourbar(fig, e, im, [-12,-6,0,6,12], r'$\Delta T_{2m}$ (K)', True)
    bar_pairs.append((e, ca))
    e.legend(handles=[Line2D([],[],color='#f7f9fa',marker=(6,2,0),mec='#f7f9fa',mew=.6,
        ms=np.sqrt(20),ls='none',label='Snow: Jan 9–11',
        path_effects=[pe.Stroke(linewidth=1.15,foreground='#50606b'),pe.Normal()])],
        loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,fontsize=11.5,
        handlelength=.8,handletextpad=.25,borderpad=0,borderaxespad=0)

    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    clipped = []; intrusions = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip(): continue
        bb = text.get_window_extent(renderer)
        if bb.x0 < -.5 or bb.y0 < -.5 or bb.x1 > fig.bbox.width+.5 or bb.y1 > fig.bbox.height+.5:
            clipped.append(text.get_text())
    for owner in fig.axes:
        for text in [owner.xaxis.label,owner.yaxis.label,owner.title,owner._left_title]:
            if not text.get_visible() or not text.get_text().strip(): continue
            bb = text.get_window_extent(renderer)
            for target in [a,b,c,d,e]:
                if owner is not target and bb.overlaps(target.get_window_extent(renderer)):
                    intrusions.append(text.get_text())
    for ax in [b, d]:
        for artist in ax.collections:
            if isinstance(artist, Quiver):
                xy = artist.get_offsets(); xl,xr = sorted(ax.get_xlim()); yl,yr = sorted(ax.get_ylim())
                assert np.any((xy[:,0]>=xl)&(xy[:,0]<=xr)&(xy[:,1]>=yl)&(xy[:,1]<=yr))
    for ax in [b,c,d,e]:
        labels = [x for x in ax.get_xticklabels() if x.get_visible() and x.get_text()]
        assert all(not l.get_window_extent(renderer).overlaps(r.get_window_extent(renderer))
                   for l,r in zip(labels,labels[1:])), 'Tick label overlap'
    for ax, ca in bar_pairs:
        aa, cc = ax.get_position(), ca.get_position()
        if ax in [a,b]:
            np.testing.assert_allclose([aa.y0,aa.height], [cc.y0,cc.height], rtol=0, atol=1e-12)
        else:
            np.testing.assert_allclose([aa.x0,aa.width], [cc.x0,cc.width], rtol=0, atol=1e-12)
    row12_gap=(a.get_position().y0-b.get_position().y1)*HEIGHT
    row23_gap=(b.get_position().y0-c.get_position().y1)*HEIGHT
    np.testing.assert_allclose([row12_gap,row23_gap], [.80,2.45], atol=1e-12)
    fig.savefig(OUT/'Figure5_snow_region_actual_preview.png',dpi=150,facecolor='white')
    assert not clipped, clipped
    assert not intrusions, intrusions
    path = OUT/'Figure5_snow_region_actual.png'
    fig.savefig(path,dpi=600,facecolor='white')
    with Image.open(path) as image:
        assert abs(image.width-WIDTH/2.54*600)<2 and abs(image.height-HEIGHT/2.54*600)<2
        pixels=list(image.size)
    CHECKS.update(figure_width_cm=WIDTH,figure_height_cm=HEIGHT,DPI=600,pixels=pixels,
        subplot_count=5,layout='one full-width panel in row1; one full-width panel in row2; three maps in row3',
        panels_a_b_share_x=True,panels_a_b_single_line_titles=True,
        body_font_size_pt=13,title_font_size_pt=13.5,colourbar_tick_font_size_pt=11.5,
        date_format='%b %d',daily_cells='00--24UTC; labels and contour/wind samples at12UTC',
        rectangular_colourbars=True,colourbar_extend='neither',
        colourbar_orientations=['vertical','vertical','horizontal','horizontal','horizontal'],
        vertical_colourbars_match_axes_height=True,horizontal_colourbars_match_axes_width=True,
        first_second_row_gap_cm=row12_gap,second_third_row_gap_cm=row23_gap,
        top_row_panel_heights_cm=[3.2,3.2],
        top_row_panel_widths_cm=[16.75,16.75],third_row_panel_widths_cm=[5.1,5.1,5.1],
        third_row_column_gaps_cm=[1.,1.],
        temperature_colour_range_C=[low,high],clipped_text=[],labels_intruding_into_other_panels=[],PDF_generated=False)
    plt.close(fig)
    return path


def write_note(box):
    text=f'''# 图5：降雪区域实际风温与天气图

- 以原实际风场版本为基础，删除原图(c)的梯度/风/平流曲线。
- 按用户最新指定范围，
  图(a)采用{abs(box[1]):g}–{abs(box[0]):g}°W、{box[2]:g}–{box[3]:g}°N；
  图(b)采用相同经度范围，纬度轴保持25–80°N。
- 第一行图(a)是实际经向风V，单位m/s，负值向南。使用1–1000hPa全部已有层，
  剔除地下气压面后，在每层全时段固定有效格点上作cos纬度加权平均。
- 第二行图(b)是850hPa实际温度，单位°C，细线为−10°C和0°C实际等温线，
  箭头为同层实际经向风，向下表示向南，不表示垂直下降或气团轨迹。
- 图(a、b)没有减去磁暴前均值。它们共用x轴，日期标签在图(b)下方单行显示，子图标题也均为单行。
- 原(d–f)放在第三行并连续编号为(c–e)，地图数据与原版一致：
  1月7–9日500hPa高度变化和低高度包络、1月7–9日850hPa温度平流与风，
  以及1月9–11日地面温度距平与白色雪花站点。
- 第三行的高度变化仍相对12月28–30日，地面温度距平仍相对1991–2020逐日气候态。
- 所有色标均为矩形。图(a、b)色标位于右侧并与对应子图等高，单位上移；
  图(c–e)色标位于各自下方并与子图等宽。风色标±15m/s，超范围取端点颜色，原值完整保留。
- 前两行子图高度均为3.2cm，两行图框间隔0.8cm；第二、三行图框间隔保持2.45cm。
  画布宽20cm、高18.55cm。
- 图(a)统计区域与末图降雪站点的显示范围分别设置；图(a)采用用户指定的40–60°N，雪花仍显示原观测位置。
- 地形覆盖比例、实际日值、资料指纹与导出检查保存在同目录CSV、NPZ、method_checks.json。
- 本图展示实际风温及天气过程，未单独估计平流层对对流层的因果贡献。
'''
    (OUT/'图5新版说明.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous = ROOT/'results/Figure5_meridional_bridge_20261030/Figure5_meridional_bridge.png'
    before = digest(previous) if previous.exists() else None
    weather.OUT = OUT/'weather_cache'; weather.OUT.mkdir(exist_ok=True)
    maps,_ = weather.load_data()
    box = analysis_box(maps['snow'])
    data = analyse(box)
    path = plot(data,maps,box)
    assert before == (digest(previous) if previous.exists() else None)
    CHECKS.update(script_sha256=digest(Path(__file__)),
        weather_source_sha256={str(p):digest(p) for p in [weather.ALIGNED,weather.SURFACE,weather.BOUNDARY,weather.SNOW_DAILY]},
        previous_figure_unchanged=True,weather_method_checks=weather.CHECKS)
    (OUT/'method_checks.json').write_text(json.dumps(CHECKS,ensure_ascii=False,indent=2),encoding='utf-8')
    write_note(box)
    print('Panel(a) averaging box:',box,flush=True)
    print('Saved:',path,flush=True)


if __name__ == '__main__':
    main()
