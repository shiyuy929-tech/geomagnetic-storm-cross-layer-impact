"""Three-row Figure5: actual Zwest-Zeast, actual V, advection, T2m/snow.

Uses existing native daily diagnostic caches without changing their values.
The two upper panels share UTC time; the maps share latitude. PNG only.
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
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PIL import Image
import plot_figure5_w2w3_troposphere_impact as weather

OUT = ROOT/'results/Figure5_height_wind_weather_20261101'
REGION = (-130.,-60.,40.,60.)
REGION_FIELDS = OUT/'regional_fields_60_130W'
Z_INPUT = REGION_FIELDS/'actual_height_fields.npz'
V_INPUT = REGION_FIELDS/'snow_region_actual_fields.npz'
WIDTH, HEIGHT = 20., 15.8
DAYS = pd.date_range('2024-12-28','2025-01-11')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_region_fields():
    """Recompute the selected region with the original native-data operators."""
    checks_path=REGION_FIELDS/'regional_fields_checks.json'
    if Z_INPUT.exists() and V_INPUT.exists() and checks_path.exists():
        checks=json.loads(checks_path.read_text(encoding='utf-8'))
        assert checks['region_degE_degN']==list(REGION)
        assert all(sha(path)==digest for path,digest in checks['derived_sha256'].items())
        return checks
    import contextlib, io
    import plot_actual_west_east_height_20261031 as height_builder
    import plot_figure5_snow_region_actual_20261031 as wind_builder
    REGION_FIELDS.mkdir(parents=True,exist_ok=True)
    height_builder.OUT=REGION_FIELDS
    height_builder.WEST,height_builder.EAST,height_builder.SOUTH,height_builder.NORTH=REGION
    wind_builder.OUT=REGION_FIELDS
    wind_builder.CHECKS.clear()
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        z,z_checks=height_builder.analyse()
        v=wind_builder.analyse(REGION)
    (REGION_FIELDS/'recalculation_log.txt').write_text(log.getvalue(),encoding='utf-8')
    np.testing.assert_array_equal(z['time'],v['time'])
    np.testing.assert_array_equal(z['pressure_hPa'],v['pressure_hPa'])
    checks=dict(region_degE_degN=list(REGION),
        west_minus_east_definition='Z(130W)-Z(60W); identical fixed paired above-ground latitude support',
        height_calculation=z_checks,wind_calculation=wind_builder.CHECKS,
        source_operators_unchanged=True,
        derived_sha256={str(path):sha(path) for path in [Z_INPUT,V_INPUT]})
    checks_path.write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    return checks


def axes(fig,x,y,w,h,**kwargs):
    return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT],**kwargs)


def bar(fig,owner,im,ticks,unit,labelpad=1):
    pos=owner.get_position()
    ca=fig.add_axes([pos.x1+.30/WIDTH,pos.y0,.18/WIDTH,pos.height])
    cb=fig.colorbar(im,cax=ca,ticks=ticks,extend='neither')
    cb.set_label(unit,fontsize=12,rotation=90,labelpad=labelpad)
    cb.ax.yaxis.set_label_position('right')
    cb.ax.tick_params(direction='in',labelsize=11.5,length=2.7,pad=2)
    return ca


def pressure_axis(ax):
    ticks=np.array([1,3,10,30,100,300,1000])
    ax.set_ylim(np.log(1000),np.log(1))
    ax.set_yticks(np.log(ticks)); ax.set_yticklabels([f'{x:g}' for x in ticks])
    ax.set_ylabel('pressure (hPa)',fontsize=13,labelpad=2)
    ax.set_xlim(DAYS[0],DAYS[-1]+pd.Timedelta(days=1))
    ax.set_xticks(pd.date_range('2024-12-29','2025-01-10',freq='3D')+pd.Timedelta(hours=12))
    ax.set_xticks(DAYS+pd.Timedelta(hours=12),minor=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(direction='in',top=True,right=True,length=3,pad=3)
    ax.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,length=3)


def plot(z,v,maps):
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':13,
        'axes.labelsize':13,'axes.titlesize':13.5,'axes.titleweight':'normal',
        'xtick.labelsize':13,'ytick.labelsize':13,'axes.linewidth':.75,
        'xtick.direction':'in','ytick.direction':'in'})
    fig=plt.figure(figsize=(WIDTH/2.54,HEIGHT/2.54))
    a=axes(fig,2.0,12.3,16.0,2.6)
    b=axes(fig,2.0,8.9,16.0,2.6,sharex=a)
    c=axes(fig,2.0,1.3,7.1,5.3,projection=weather.PC)
    d=axes(fig,10.9,1.3,7.1,5.3,projection=weather.PC,sharey=c)
    xe=mdates.date2num(pd.date_range(DAYS[0],DAYS[-1]+pd.Timedelta(days=1)))
    bars=[]

    # Keep zero white with the requested asymmetric limits. A linear colourbar
    # makes every250m tick equally spaced even in the shorter upper panel.
    zero=250./1250.
    positions=np.unique(np.r_[np.linspace(0,1,513),zero])
    colours=plt.get_cmap('RdBu_r')(np.where(positions<=zero,.5*positions/zero,
        .5+.5*(positions-zero)/(1-zero)))
    colours[np.isclose(positions,zero)]=[1,1,1,1]
    cmap=LinearSegmentedColormap.from_list('height_difference_zero_white',list(zip(positions,colours)),N=4096)
    cmap.set_bad('#e5e5e5')
    im=a.pcolormesh(xe,weather.edges(np.log(z['pressure_hPa'])),
        np.ma.masked_invalid(z['west_minus_east_m'].T),cmap=cmap,
        norm=Normalize(vmin=-250,vmax=1000),shading='flat')
    pressure_axis(a)
    a.set_title('(a) West–east height difference | 60–130°W, 40–60°N',loc='left',pad=7)
    bars.append(bar(fig,a,im,np.arange(-250,1001,250),'m'))
    a.tick_params(labelbottom=False)

    im=b.pcolormesh(xe,weather.edges(np.log(v['pressure_hPa'])),
        np.ma.masked_invalid(v['actual_V'].T),cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-15,vcenter=0,vmax=10),shading='flat')
    pressure_axis(b)
    b.set_title('(b) Meridional wind | 60–130°W, 40–60°N',loc='left',pad=7)
    b.set_xlabel('UTC date | 2024–2025',labelpad=4)
    bars.append(bar(fig,b,im,[-15,-10,-5,0,5,10],r'm s$^{-1}$'))

    for ax in [c,d]:
        weather.map_style(ax,ax is c)
        ax.tick_params(labelsize=13,direction='in',top=True,right=True,length=3,pad=3)
        ax.set_xticks(np.arange(-140.,-49.,20.),crs=weather.PC)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position: f'{abs(value):.0f}'))
        ax.tick_params(axis='x',labelsize=12)
        ax.set_xlabel('Longitude (°W)',fontsize=13,labelpad=4)
    c.set_ylabel('Latitude (°N)',fontsize=13,rotation=90,ha='center',va='center')
    c.yaxis.set_label_coords(-.22,.5)

    # Limit the surface panel to the native data cells: the northernmost
    # centre is75.2351N; its outer cell edge is76.1873N. Do not extend its
    # raster to82N or extrapolate missing polar temperatures.
    in_lon=(maps['s_lon']>=-140)&(maps['s_lon']<=-50)
    native_valid=np.isfinite(maps['surface'][:,in_lon]).any(axis=1)
    order=np.argsort(maps['s_lat'])
    lat=maps['s_lat'][order]
    lat_edges=weather.edges(lat)
    good_indices=np.flatnonzero(native_valid[order])
    d_lat_limits=(max(22.,float(lat_edges[good_indices[0]])),
                  min(82.,float(lat_edges[good_indices[-1]+1])))
    d.set_extent([-140.,-50.,*d_lat_limits],crs=weather.PC)
    d.set_aspect('auto')
    c.set_extent([-140.,-50.,*d_lat_limits],crs=weather.PC)
    c.set_aspect('auto')
    d.tick_params(labelleft=False)
    im=c.pcolormesh(maps['lon'],maps['lat'],maps['adv850'],cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-10,vcenter=0,vmax=10),shading='auto',
        transform=weather.PC,zorder=2)
    yi=np.flatnonzero((maps['lat']>=22)&(maps['lat']<=82))[::2]
    xi=np.flatnonzero((maps['lon']>=-140)&(maps['lon']<=-50))[::2]
    q=c.quiver(maps['lon'][xi],maps['lat'][yi],maps['u850'][np.ix_(yi,xi)],
        maps['v850'][np.ix_(yi,xi)],transform=weather.PC,scale=180,
        width=.0020,headwidth=3.4,color='#222222',zorder=8)
    wind_key=c.quiverkey(q,.72,1.15,10,r'10 m s$^{-1}$',labelpos='E',coordinates='axes',
                        fontproperties={'size':12})
    c.set_title('(c) Cold-air transport\nJan 7–9 | 850 hPa',loc='left',pad=7)
    bars.append(bar(fig,c,im,[-10,-5,0,5,10],r'K day$^{-1}$',labelpad=-2))

    im=d.pcolormesh(maps['s_lon'],maps['s_lat'],maps['surface'],cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12),shading='auto',
        transform=weather.PC,zorder=2)
    d.set_title('(d) Temperature Change\nJan 9–11 | 2 m',loc='left',pad=7)
    bars.append(bar(fig,d,im,[-12,-6,0,6,12],r'$\Delta T_{2m}$ (K)'))

    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    bb=c._left_title.get_window_extent(renderer)
    second_line_y=bb.y0+bb.height/4
    wind_key.Y=c.transAxes.inverted().transform((0,second_line_y))[1]
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    wind_bb=wind_key.text.get_window_extent(renderer)
    assert wind_bb.y0>c.bbox.y1, 'Wind key must be above map'
    for owner,legend_bb,text in [(c,wind_bb,'Jan 7–9 | 850 hPa')]:
        second_width=renderer.get_text_width_height_descent(text,owner._left_title.get_fontproperties(),False)[0]
        assert legend_bb.x0>owner._left_title.get_window_extent(renderer).x0+second_width, 'Legend overlaps second title line'
        assert legend_bb.x1<=owner.bbox.x1+1., 'Legend exceeds map right edge'
    ghost=set()
    for owner in fig.axes:
        for axis,lim in [(owner.xaxis,owner.get_xlim()),(owner.yaxis,owner.get_ylim())]:
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not min(lim)<=tick.get_loc()<=max(lim): ghost.update([tick.label1,tick.label2])
    clipped=[];intrusions=[]
    for t in fig.findobj(Text):
        if t in ghost or not t.get_visible() or not t.get_text().strip():continue
        bb=t.get_window_extent(renderer)
        if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:
            clipped.append(t.get_text())
    for owner in [a,b,c,d]:
        for t in [owner.xaxis.label,owner.yaxis.label,owner._left_title]:
            if not t.get_visible() or not t.get_text().strip():continue
            bb=t.get_window_extent(renderer)
            for other in [a,b,c,d]:
                if owner is not other and bb.overlaps(other.get_window_extent(renderer)):
                    intrusions.append(t.get_text())
    assert not clipped,clipped
    assert not intrusions,intrusions
    for owner in [c,d]:
        assert not b.xaxis.label.get_window_extent(renderer).overlaps(owner._left_title.get_window_extent(renderer)), 'Time-axis label overlaps map title'
    assert a.get_shared_x_axes().joined(a,b)
    assert c.get_shared_y_axes().joined(c,d)
    assert len(fig.axes)==8
    for owner,ca in zip([a,b,c,d],bars):
        aa,cc=owner.get_position(),ca.get_position()
        np.testing.assert_allclose([aa.y0,aa.height],[cc.y0,cc.height],atol=1e-12)
        assert cc.x0>aa.x1
        np.testing.assert_allclose(aa.height*HEIGHT,2.6 if owner in [a,b] else 5.3,atol=1e-12)
        assert ca.yaxis.label.get_rotation()==90
        tick_labels=[t for t in ca.get_yticklabels() if t.get_visible() and t.get_text()]
        assert all(not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
            for first,second in zip(tick_labels,tick_labels[1:])), 'Overlapping colourbar ticks'
    np.testing.assert_allclose(d.get_extent(weather.PC)[2:],d_lat_limits,atol=1e-8)
    np.testing.assert_allclose(c.get_ylim(),d.get_ylim(),atol=1e-12)
    np.testing.assert_allclose(np.diff(d.get_xticks()),20.,atol=1e-12)
    for owner in [b,c,d]:
        labels=[t for t in owner.get_xticklabels() if t.get_visible() and t.get_text()]
        assert all(not l.get_window_extent(renderer).overlaps(r.get_window_extent(renderer))
                   for l,r in zip(labels,labels[1:])), 'Overlapping x ticks'
    fig.savefig(OUT/'Figure5_height_wind_weather_preview.png',dpi=140,facecolor='white')
    path=OUT/'Figure5_height_wind_weather.png'
    fig.savefig(path,dpi=600,facecolor='white')
    with Image.open(path) as pic:pixels=list(pic.size)
    assert abs(pixels[0]-WIDTH/2.54*600)<2
    plt.close(fig)
    return dict(figure_path=str(path),pixels=pixels,width_cm=WIDTH,height_cm=HEIGHT,dpi=600,
        subplot_count=4,layout='three rows: full-width a, full-width b, side-by-side c and d',
        shared_x_pairs=[['a','b','UTC time']],shared_y_pairs=[['c','d','latitude']],
        time_tick_step_days=1,time_label_step_days=3,time_ticks_at='daily cell centres, 12 UTC',
        panel_heights_cm={'a':2.6,'b':2.6,'c':5.3,'d':5.3},map_longitude_tick_step_degrees=20,
        map_y_axis_title='Latitude (°N)',panel_d_latitude_limits_N=list(d_lat_limits),
        panel_d_latitude_limit_rule='native finite-temperature cell edges, intersected with original map extent',
        legend_position='wind key above panel c, to the right of its second title line; no snow markers',
        colourbar_label_pad_pt={'a':1,'b':1,'c':-2,'d':1},
        second_to_third_row_gap_cm=2.3,
        colourbar_labels='vertical, on right of each colourbar',
        right_rectangular_colorbars=True,colorbars_match_panel_heights=True,
        text_outside_canvas=clipped,labels_intruding_into_other_panels=intrusions,PDF_generated=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    region_checks=ensure_region_fields()
    hashes={str(p):sha(p) for p in [Z_INPUT,V_INPUT,weather.ALIGNED,weather.SURFACE,
                                  weather.BOUNDARY,weather.SNOW_DAILY]}
    with np.load(Z_INPUT) as f:z={k:f[k] for k in f.files}
    with np.load(V_INPUT) as f:v={k:f[k] for k in f.files}
    np.testing.assert_array_equal(z['time'],v['time'])
    np.testing.assert_array_equal(z['time'],DAYS.values.astype('datetime64[D]'))
    np.testing.assert_array_equal(z['pressure_hPa'],v['pressure_hPa'])
    np.testing.assert_allclose(z['west_minus_east_m'],z['Z_west_m']-z['Z_east_m'],equal_nan=True)
    weather.OUT=OUT/'weather_cache';weather.OUT.mkdir(exist_ok=True)
    maps,_=weather.load_data()
    checks=plot(z,v,maps)
    assert all(sha(p)==h for p,h in hashes.items())
    checks.update(source_sha256=hashes,all_existing_input_values_unchanged=True,
        pressure_levels_hPa=z['pressure_hPa'].tolist(),
        panel_a_definition='actual Z(130W)-Z(60W), cos-latitude mean over40--60N; no prestorm subtraction',
        panel_a_meridians_degE=list(REGION[:2]),panel_b_region_degE_degN=list(REGION),
        regional_recalculation=region_checks,
        panel_a_same_paired_fixed_above_ground_latitudes_on_both_meridians=True,
        panel_a_paired_latitude_fraction=z['fixed_latitude_fraction'].tolist(),
        panel_a_colour_limits_m=[-250,1000],panel_a_colour_ticks_m=list(range(-250,1001,250)),
        panel_a_colour_mapping='linear scale with zero-white colour map',panel_b_colour_limits_m_s=[-15,10],
        panel_a_native_values_above_colour_max_count=int(np.sum(z['west_minus_east_m']>1000)),
        removed_panels=['850hPa temperature latitude-time','500hPa trough map'],
        map_date_windows={'c':'2025-01-07 through2025-01-09','d':'2025-01-09 through2025-01-11'},
        surface_anomaly_reference='1991--2020 daily climatology',
        surface_change_definition='mean Jan 9–11 NCEP R1 2 m temperature anomaly relative to corresponding 1991–2020 daily climatology',
        snow_markers='none',
        panel_c_colour_variable='actual horizontal temperature advection, -V dot grad(T)',
        weather_checks=weather.CHECKS)
    (OUT/'method_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'图5四行版说明.md').write_text('''# 图5四行版

1. 图(a)为40–60°N余弦纬度加权的实际西减东位势高度差：Z(130°W)−Z(60°W)，标题标注60–130°W，不是相对磁暴前的异常；范围1–1000hPa，原生数据不插值。两侧采用全时段均位于地表以上的共同纬度点，低层支持比例见method_checks.json。1000hPa若没有共同有效纬点则留为灰色。
2. 图(b)为实际经向风，区域为60–130°W、40–60°N，按该区域计算，色标为−15至10m/s。
3. 删除原图(b)的850hPa温度纬度–时间图，以及原第三行第一幅500hPa槽图。
4. 图(c)为1月7–9日850hPa实际水平温度平流和水平风，单位K/day和m/s；图(d)为1月9–11日NCEP 2m气温相对1991–2020逐日气候态的平均距平，不标示降雪站点。
5. 三行布局：图(a)、图(b)各占一行，共享UTC时间轴，高度压缩至各2.6cm；图(c)、图(d)在第三行并排，共享纬度轴，高度增至各5.3cm。地图经度刻度步长20°，横轴单位统一写在Longitude (°W)标题中；公共纵轴标题为Latitude (°N)，右图保留刻度线而省略重复纬度标签。所有颜色条标签均旋转90°放在对应颜色条右侧，颜色条与子图等高。
6. 图(a)色标为−250至1000m，刻度间隔250m，采用线性色标并保持零值白色；超出色标范围的原始数值保留并显示为端点颜色。两幅地图的公共纬度上限收至图(d)有效温度格点的外边缘约76.1873°N，不外推北侧空白区域。风速图例放在图(c)子图标题第二行的右侧；图(d)不画降雪标志或降雪图例。图(d)标题为Temperature Change，第二行为Jan 9–11 | 2 m。图(c)颜色条标签间隔单独缩至−2pt，其余为1pt。第二行与第三行的图框间距由2.5cm缩至2.3cm，画布高度相应由16.0cm缩至15.8cm。
7. 原始资料与此前70–110°W诊断缓存保留。上两幅图采用原始ERA5资料及既有计算方法重算60–130°W区域，结果另存regional_fields_60_130W；下两幅地图数据不变。输出20cm宽、600dpi PNG，不输出PDF。
8. 图(a)、图(b)每天均绘制时间刻度线，刻度位置为每日数据方块中心12 UTC；日期标签仍只标Dec 29、Jan 01、Jan 04、Jan 07和Jan 10。
''',encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
