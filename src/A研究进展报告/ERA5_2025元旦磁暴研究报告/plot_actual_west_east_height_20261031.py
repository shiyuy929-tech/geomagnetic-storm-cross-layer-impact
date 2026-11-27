"""Actual ERA5 height on the two sides of the existing cold-flow corridor.

No harmonic reconstruction, prestorm subtraction, or smoothing is applied.
The west/east meridians (110W and70W) and latitude band40--60N are inherited
from the user's existing wind corridor, before looking at the height signs.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'plot_packages'),str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from matplotlib.text import Text
from PIL import Image
import analyze_upper_vortex_followup_20261029 as base

OUT=ROOT/'results/actual_west_east_height_20261031'
WEST,EAST,SOUTH,NORTH=-110.,-70.,40.,60.
WIDTH,HEIGHT=20.,13.6

def edges(a):
    a=np.asarray(a,dtype=float)
    return np.r_[a[0]-(a[1]-a[0])/2,(a[1:]+a[:-1])/2,a[-1]+(a[-1]-a[-2])/2]

def area_mean(a,w):
    den=w.sum(axis=-1)
    return np.divide(np.sum(np.where(w>0,a,0)*w,axis=-1),den,
        out=np.full(a.shape[:-1],np.nan),where=den>0)

def analyse():
    OUT.mkdir(parents=True,exist_ok=True)
    base.START=np.datetime64('2024-12-28');base.END=np.datetime64('2025-01-11')
    times,days,p,lat,lon,d=base.load()
    sp=base.read_sp(times,lat,lon)
    jwest=int(np.flatnonzero(np.isclose(lon,WEST))[0])
    jeast=int(np.flatnonzero(np.isclose(lon,EAST))[0])
    yi=np.flatnonzero((lat>=SOUTH)&(lat<=NORTH))
    weight=np.cos(np.deg2rad(lat[yi]))
    # Identical latitude points on both meridians, fixed over all60samples.
    # Underground pressure levels are excluded before any averaging.
    good_w=sp[:,None,yi,jwest]>=p[None,:,None]*100
    good_e=sp[:,None,yi,jeast]>=p[None,:,None]*100
    support=(good_w&good_e).all(axis=0)
    w=support*weight[None]
    zw=d['z'][:,:,yi,jwest].astype(float)/base.G
    ze=d['z'][:,:,yi,jeast].astype(float)/base.G
    west=base.daily(area_mean(zw,w[None]))
    east=base.daily(area_mean(ze,w[None]))
    delta=west-east
    np.testing.assert_allclose(delta,base.daily(area_mean(zw-ze,w[None])),equal_nan=True,atol=1e-10)
    fraction=w.sum(-1)/weight.sum()
    rows=[dict(date=str(day),pressure_hPa=float(pp),Z_west_m=west[i,j],
        Z_east_m=east[i,j],Z_west_minus_east_m=delta[i,j],
        paired_fixed_latitude_fraction=fraction[j],paired_latitude_count=int(support[j].sum()))
        for i,day in enumerate(days) for j,pp in enumerate(p)]
    daily=pd.DataFrame(rows)
    daily.to_csv(OUT/'actual_height_daily.csv',index=False,encoding='utf-8-sig')
    windows={'pre':('2024-12-28','2024-12-30'),'early':('2025-01-01','2025-01-03'),
        'main':('2025-01-07','2025-01-09'),'late':('2025-01-10','2025-01-11')}
    periods=[]
    for name,limits in windows.items():
        use=base.window(days,limits)
        for j,pp in enumerate(p):
            if not support[j].any():continue
            periods.append(dict(period=name,pressure_hPa=float(pp),
                Z_west_m=float(west[use,j].mean()),Z_east_m=float(east[use,j].mean()),
                west_minus_east_m=float(delta[use,j].mean()),
                west_higher_day_count=int(np.sum(delta[use,j]>0)),day_count=int(use.sum()),
                fixed_latitude_fraction=float(fraction[j])))
    periods=pd.DataFrame(periods)
    periods.to_csv(OUT/'actual_height_periods.csv',index=False,encoding='utf-8-sig')
    f=dict(time=days,pressure_hPa=p,latitude=lat[yi],paired_fixed_support=support,
        fixed_latitude_fraction=fraction,Z_west_m=west,Z_east_m=east,west_minus_east_m=delta)
    np.savez_compressed(OUT/'actual_height_fields.npz',**f)
    qa=dict(longitudes_degrees_east=[WEST,EAST],latitude_band_N=[SOUTH,NORTH],
        native_time_UTC=[str(times[0]),str(times[-1])],native_grid_degrees=2.5,
        averaging='four6-hourly values per UTCday; cos(latitude) area weights',
        variable='actual full ERA5geopotential divided by9.80665; no wave decomposition',
        subtract_prestorm=False,subtract_zonal_mean=False,normalization=False,smoothing=False,
        support='same paired latitude points on both meridians; above terrain at every sample',
        fixed_latitude_fraction=dict(zip([str(float(x)) for x in p],fraction.tolist())),
        display_pressure_hPa=[100,1000],all_native_pressure_levels_saved=True,
        source_sha256={str(v):hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in dict.fromkeys(base.INPUTS)})
    print(periods[(periods.pressure_hPa.isin([100,200,300,500,700,850,925,1000]))&
        (periods.period.isin(['pre','main']))].to_string(index=False),flush=True)
    return f,qa

def plot(f,qa):
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,
        'axes.labelsize':12,'axes.titlesize':13,'axes.titleweight':'normal',
        'xtick.labelsize':12,'ytick.labelsize':12,'axes.linewidth':.8,
        'xtick.direction':'in','ytick.direction':'in'})
    fig=plt.figure(figsize=(WIDTH/2.54,HEIGHT/2.54))
    days=pd.to_datetime(f['time'])
    x=mdates.date2num(pd.date_range(days[0],days[-1]+pd.Timedelta(days=1)))
    select=(f['pressure_hPa']>=100)&(f['pressure_hPa']<=1000)
    p=f['pressure_hPa'][select]
    zw=f['Z_west_m'][:,select].T;ze=f['Z_east_m'][:,select].T
    dd=f['west_minus_east_m'][:,select].T
    zlo=1000*np.floor(np.nanmin([zw,ze])/1000)
    zhi=1000*np.ceil(np.nanmax([zw,ze])/1000)
    dlo=50*np.floor(np.nanmin(dd)/50);dhi=50*np.ceil(np.nanmax(dd)/50)
    if not dlo<0<dhi:
        norm=plt.Normalize(vmin=dlo,vmax=dhi)
    else:norm=TwoSlopeNorm(vmin=dlo,vcenter=0,vmax=dhi)
    ticks=[100,200,300,500,700,1000]
    axes=[];bars=[]
    for k,(y,data,title) in enumerate([(9.65,zw,'(a) West: 110°W'),
        (5.75,ze,'(b) East: 70°W'),(1.85,dd,'(c) West minus east')]):
        ax=fig.add_axes([1.8/WIDTH,y/HEIGHT,15.8/WIDTH,3.05/HEIGHT],
            sharex=axes[0] if axes else None)
        if k<2:
            cmap=plt.get_cmap('viridis').copy()
            im=ax.pcolormesh(x,edges(np.log(p)),data,cmap=cmap,vmin=zlo,vmax=zhi,shading='flat')
            cbticks=np.linspace(zlo,zhi,5)
        else:
            cmap=plt.get_cmap('RdBu_r').copy()
            im=ax.pcolormesh(x,edges(np.log(p)),data,cmap=cmap,norm=norm,shading='flat')
            cbticks=np.unique([dlo,0,250,dhi])
        ax.set_facecolor('#e5e5e5')
        ax.set_ylim(np.log(1000),np.log(100))
        ax.set_yticks(np.log(ticks));ax.set_yticklabels(ticks)
        ax.set_ylabel('pressure (hPa)',labelpad=5)
        ax.set_title(title,loc='left',pad=7)
        if k==0:ax.text(1,1.035,'40–60°N',ha='right',va='bottom',transform=ax.transAxes,fontsize=12)
        ax.set_xlim(x[0],x[-1])
        ax.set_xticks(mdates.date2num(pd.date_range('2024-12-29','2025-01-10',freq='3D')+pd.Timedelta(hours=12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.axvline(pd.Timestamp('2024-12-31 16:18'),color='#d74735',ls='--',lw=.9)
        ax.tick_params(direction='in',top=True,right=True,length=3,pad=3,labelbottom=(k==2))
        if k==2:ax.set_xlabel('UTC date | 2024–2025',labelpad=5)
        pos=ax.get_position()
        ca=fig.add_axes([pos.x1+.30/WIDTH,pos.y0,.22/WIDTH,pos.height])
        cb=fig.colorbar(im,cax=ca,ticks=cbticks,extend='neither')
        cb.ax.set_title('m',fontsize=12,pad=7)
        cb.ax.tick_params(direction='in',labelsize=10.5,length=2.5,pad=2)
        axes.append(ax);bars.append(ca)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();cw,ch=fig.canvas.get_width_height()
    outside=[];overlapping_colourbar_ticks=[]
    for bar in bars:
        labels=[t for t in bar.get_yticklabels() if t.get_visible() and t.get_text()]
        boxes=[t.get_window_extent(renderer) for t in labels]
        for i in range(len(boxes)-1):
            if boxes[i].overlaps(boxes[i+1]):
                overlapping_colourbar_ticks.append([labels[i].get_text(),labels[i+1].get_text()])
    assert not overlapping_colourbar_ticks,('Overlapping colourbar tick labels',overlapping_colourbar_ticks)
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        box=t.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1>cw+1 or box.y1>ch+1:
            outside.append(dict(text=t.get_text(),bbox=box.extents.tolist()))
    assert not outside,('Text outside canvas',outside)
    output=OUT/'actual_west_east_geopotential_height.png'
    fig.savefig(output,dpi=600)
    with Image.open(output) as im:
        assert abs(im.width-4724)<=1
        size=im.size
        preview=im.copy();preview.thumbnail((1550,1550));preview.save(OUT/'actual_west_east_geopotential_height_preview.png')
    plt.close(fig)
    qa.update(figure_path=str(output),width_cm=WIDTH,dpi=600,pixel_size=size,
        colour_Z_range_m=[zlo,zhi],colour_difference_range_m=[dlo,dhi],
        rectangular_colourbars=True,colourbars_match_axes_height=True,text_outside_canvas=outside,
        overlapping_colourbar_ticks=overlapping_colourbar_ticks)
    (OUT/'checks.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'readme_中文.md').write_text('''# 北美西高东低的实际位势高度诊断

图(a,b)为ERA5完整位势高度场在110°W、70°W两条经线、40°–60°N内的逐日平均，单位m；未作W1分解、磁暴前相减、纬向平均相减或平滑。两图使用同一色标。图(c)为同日、同层的西侧减东侧：正值表示西高东低，负值表示西低东高。它是两侧实际高度差，不是相对磁暴前的变化。

时期2024年12月28日至2025年1月11日，每日00/06/12/18UTC四个样本平均；每天方格的中心为12UTC。红虚线沿用原研究的2024年12月31日16:18UTC事件时刻。

本图聚焦100–1000hPa的下平流层和对流层。完整原生气压层数据都保存在actual_height_fields.npz和CSV中。两侧采用完全相同的纬度点和cos纬度权重；各层只使用两侧均在地表以上、全期有效的共同纬点。灰色代表该层没有共同有效纬点，支持比例见checks.json/CSV。低层支持范围收窄由地形造成，未将地下压力层当成实际大气。

110°W和70°W是此前70°–110°W南向风统计区的西、东端，纬度40°–60°N也沿用该区。区域选择在查看本图高度差正负之前固定。该图检验此通道两侧的高度配置，不把任何一条经线强行定义为高压脊或低压槽的中心。

actual_height_periods.csv列出磁暴前(12月28–30日)、早期(1月1–3日)、主要期(1月7–9日)和末期(1月10–11日)的实际高度差。该表用于读取数值；未另外画重复的剖面曲线。
''',encoding='utf-8')
    print('SAVED',output,flush=True)

def main():
    f,qa=analyse();plot(f,qa)

if __name__=='__main__':main()
