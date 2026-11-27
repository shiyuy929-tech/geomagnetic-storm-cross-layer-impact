"""Nature-sized rendering of the existing height, wind and weather figure."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'plot_packages'), str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from PIL import Image
import plot_figure5_height_wind_weather_20261101 as source

weather = source.weather
OUT = ROOT/'results/Figure5_height_wind_weather_180mm_20261108'
W, H = 18., 15.3
COLORBARS=[]


def axis(fig, x, y, w, h, **kw):
    return fig.add_axes([x/W,y/H,w/W,h/H], **kw)


def bar(fig, ax, im, ticks, label):
    p=ax.get_position()
    ca=fig.add_axes([p.x1+.20/W,p.y0,.16/W,p.height])
    cb=fig.colorbar(im,cax=ca,ticks=ticks,extend='neither')
    cb.set_label(label, fontsize=7, labelpad=2)
    cb.ax.tick_params(direction='in', labelsize=7, length=2.2, pad=1)
    COLORBARS.append(cb)
    return cb


def letter(ax, value):
    if value in ('c', 'd'):
        ax.text(-.005, 1.035, value, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8, fontweight='bold', clip_on=False)
    else:
        ax.text(.0, 1.035, value, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8, fontweight='bold', clip_on=False)


def pressure_axis(ax, labels):
    ticks=np.array([1,3,10,30,100,300,1000])
    ax.set_ylim(np.log(1000),np.log(1))
    ax.set_yticks(np.log(ticks))
    ax.set_yticklabels([f'{x:g}' for x in ticks])
    ax.set_ylabel('Pressure (hPa)',labelpad=2)
    ax.set_xlim(source.DAYS[0],source.DAYS[-1]+pd.Timedelta(days=1))
    ax.set_xticks(pd.date_range('2024-12-29','2025-01-10',freq='3D')+pd.Timedelta(hours=12))
    ax.set_xticks(source.DAYS+pd.Timedelta(hours=12),minor=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(which='major',direction='in',top=True,right=True,
                   length=3,pad=2,labelbottom=labels)
    ax.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,length=1.6)
    if labels: ax.set_xlabel('UTC date | 2024–2025',labelpad=2)


def plot(z,v,maps):
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,
        'axes.titlesize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.6,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(W/2.54,H/2.54))
    a=axis(fig,1.38,11.32,15.06,2.25)
    b=axis(fig,1.38,8.45,15.06,2.25,sharex=a)
    c=axis(fig,1.38,1.50,6.35,5.50,projection=weather.PC)
    d=axis(fig,10.17,1.50,6.27,5.50,projection=weather.PC,sharey=c)
    xe=mdates.date2num(pd.date_range(source.DAYS[0],source.DAYS[-1]+pd.Timedelta(days=1)))
    zero=250./1250.
    pos=np.unique(np.r_[np.linspace(0,1,513),zero])
    colors=plt.get_cmap('RdBu_r')(np.where(pos<=zero,.5*pos/zero,
        .5+.5*(pos-zero)/(1-zero)))
    colors[np.isclose(pos,zero)]=[1,1,1,1]
    cmap=LinearSegmentedColormap.from_list('height_difference_zero_white',list(zip(pos,colors)),N=4096)
    cmap.set_bad('#e5e5e5')
    im=a.pcolormesh(xe,weather.edges(np.log(z['pressure_hPa'])),
        np.ma.masked_invalid(z['west_minus_east_m'].T),cmap=cmap,
        norm=Normalize(-250,1000),shading='flat')
    pressure_axis(a,False)
    a.set_title('Height difference (130°W − 60°W)',pad=4)
    letter(a,'a')
    bar(fig,a,im,np.arange(-250,1001,250),'m').set_label('m', fontsize=7, labelpad=.25)

    im=b.pcolormesh(xe,weather.edges(np.log(v['pressure_hPa'])),
        np.ma.masked_invalid(v['actual_V'].T),cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-15,vcenter=0,vmax=10),shading='flat')
    pressure_axis(b,True)
    b.set_title('Meridional wind',pad=4)
    letter(b,'b')
    bar(fig,b,im,[-15,-10,-5,0,5,10],r'm s$^{-1}$')

    in_lon=(maps['s_lon']>=-140)&(maps['s_lon']<=-50)
    native_valid=np.isfinite(maps['surface'][:,in_lon]).any(axis=1)
    order=np.argsort(maps['s_lat']); lat=maps['s_lat'][order]
    lat_edges=weather.edges(lat); good=np.flatnonzero(native_valid[order])
    limits=(max(22.,float(lat_edges[good[0]])),min(82.,float(lat_edges[good[-1]+1])))
    for ax,left in [(c,True),(d,False)]:
        weather.map_style(ax,left)
        ax.set_extent([-140.,-50.,*limits],crs=weather.PC)
        ax.set_aspect('auto')
        ax.set_xticks(np.arange(-140.,-49.,20.),crs=weather.PC)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{abs(value):.0f}'))
        ax.tick_params(direction='in',top=True,right=True,bottom=True,left=True,
                       labelsize=7,length=3.2,width=.7,pad=2,labelleft=True)
        ax.xaxis.set_zorder(20)
        ax.yaxis.set_zorder(20)
        for spine in ax.spines.values(): spine.set_zorder(21)
        ax.set_xlabel('Longitude (°W)',labelpad=2)
    c.set_ylabel('Latitude (°N)',labelpad=3)
    im=c.pcolormesh(maps['lon'],maps['lat'],maps['adv850'],cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-10,vcenter=0,vmax=10),shading='auto',
        transform=weather.PC,zorder=2)
    yi=np.flatnonzero((maps['lat']>=22)&(maps['lat']<=82))[::2]
    xi=np.flatnonzero((maps['lon']>=-140)&(maps['lon']<=-50))[::2]
    q=c.quiver(maps['lon'][xi],maps['lat'][yi],maps['u850'][np.ix_(yi,xi)],
        maps['v850'][np.ix_(yi,xi)],transform=weather.PC,scale=180,
        width=.0032,headwidth=3.4,color='#1c1c1c',zorder=8)
    c.set_title('850 hPa cold-air transport',loc='center',pad=6)
    letter(c,'c')
    bar(fig,c,im,[-10,-5,0,5,10],r'K day$^{-1}$').set_label(r'K day$^{-1}$',fontsize=10,labelpad=.25)
    im=d.pcolormesh(maps['s_lon'],maps['s_lat'],maps['surface'],cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12),shading='auto',
        transform=weather.PC,zorder=2)
    d.set_title('2 m temperature anomaly',loc='center',pad=6)
    letter(d,'d')
    bar(fig,d,im,[-12,-6,0,6,12],r'$\Delta T_{2m}$ (K)')
    fig.canvas.draw()
    # Enforce the requested type sizes, including tick and colourbar labels
    # inherited from the earlier map renderer.
    for item in fig.findobj(match=Text):
        is_panel_letter=(item.get_text() in ('a','b','c','d') and
                         item.get_fontweight()=='bold')
        item.set_fontsize(11 if is_panel_letter else 10)
        item.set_fontfamily('Arial')
    for cb in COLORBARS:
        cb.ax.tick_params(labelsize=9)
        cb.ax.yaxis.label.set_fontsize(9)
        cb.ax.xaxis.label.set_fontsize(9)
        cb.ax.title.set_fontsize(9)
    for ax in (a,b,c,d):
        ax.title.set_fontsize(10)
    fig.set_size_inches(W/2.54,H/2.54)
    # Keep the final width 180 mm and fit the content with 2 mm outer margins.
    margin=1/25.4
    margin_right=1/25.4
    for _ in range(8):
        fig.canvas.draw()
        tight=fig.get_tightbbox(fig.canvas.get_renderer())
        delta=180/25.4-(tight.width+margin+margin_right)
        if abs(delta)<.0005: break
        fig.set_size_inches(fig.get_figwidth()+delta,fig.get_figheight(),forward=True)
    fig.canvas.draw()
    tight=fig.get_tightbbox(fig.canvas.get_renderer())
    export_bbox=Bbox.from_extents(tight.x0-margin,tight.y0-margin,
                                 tight.x0-margin+180/25.4,tight.y1+margin)
    assert export_bbox.height*25.4 <= 170
    print('export_mm',round(export_bbox.width*25.4,2),round(export_bbox.height*25.4,2))
    path=OUT/'Figure5_height_wind_weather_nature.png'
    fig.savefig(path,dpi=600,bbox_inches=export_bbox,pad_inches=0)
    fig.savefig(path.with_suffix('.svg'),dpi=600,bbox_inches=export_bbox,pad_inches=0)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source.ensure_region_fields()
    with np.load(source.Z_INPUT) as f:z={k:f[k] for k in f.files}
    with np.load(source.V_INPUT) as f:v={k:f[k] for k in f.files}
    np.testing.assert_array_equal(z['time'],v['time'])
    np.testing.assert_array_equal(z['pressure_hPa'],v['pressure_hPa'])
    np.testing.assert_allclose(z['west_minus_east_m'],z['Z_west_m']-z['Z_east_m'],equal_nan=True)
    weather.OUT=OUT/'weather_cache';weather.OUT.mkdir(exist_ok=True)
    maps,_=weather.load_data()
    path=plot(z,v,maps)
    with Image.open(path) as image: size_mm=[round(x/600*25.4,2) for x in image.size]
    (OUT/'figure_settings.json').write_text(json.dumps({
      'size_mm':size_mm,'dpi':600,'ordinary_font_pt':10,'panel_letters_pt':11,'colorbar_font_pt':9,'outer_margin_mm':{'left':1,'right':1,'top':1,'bottom':1},
      'map_arrow_width':.0032,'previous_arrow_width':.0020,'arrow_scale':180,
      'displayed_snow_markers':False,'surface_reference':'1991–2020 daily climatology',
      'data_and_calculations':'original regional caches and weather.load_data',
      'source_hashes':{str(p):source.sha(p) for p in [source.Z_INPUT,source.V_INPUT,weather.ALIGNED,weather.SURFACE]}},
      ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',path)


if __name__=='__main__':
    main()
