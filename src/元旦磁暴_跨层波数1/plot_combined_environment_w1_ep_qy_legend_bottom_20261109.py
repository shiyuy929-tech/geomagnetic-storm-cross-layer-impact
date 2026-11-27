"""Regional propagation environment and W1 diagnostics with daily EP vectors.

Data, calculation methods and focus-region masks are reused from the audited
source scripts; this script only arranges and labels the existing diagnostics.
"""
from pathlib import Path
import sys, json, inspect
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, SymLogNorm, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.dates import DayLocator
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'two_region_mechanism_profiles_20261021'))
import plot_figure3_flux_amplitude_EP_nature_20261105 as wave
import plot_mechanism_chain_2x3_20261022 as env

m = env.m
W, H = 20.0, 16.90
OUT = ROOT / 'combined_environment_W1_EP_qy_legend_bottom_20261109'
XS, PANEL_W, PANEL_H = [1.05, 6.20], 3.68, 3.05
ROWS = [13.28, 9.18, 5.33, 1.48]
MAP_X, MAP_W, MAP_H = [11.78, 15.22], 3.02, 4.4833
MAP_Y = [11.8467, 6.6634, 1.48]
BLUE = '#53B8E5'
COLORBARS = []

def axes(fig,x,y,w,h):
    return fig.add_axes([x/W,y/H,w/W,h/H])

def title(ax,letter,heading,subtitle):
    ax.set_title(f'{heading} | {subtitle}',fontsize=7,pad=4,loc='center',x=.5)
    ax.text(-.08,1.07 if letter in 'efgh' else 1.035,letter,transform=ax.transAxes,fontsize=8,
            fontweight='bold',ha='left',va='bottom')

def calendar(ax,row,events):
    env.calendar(ax,row,events)
    ax.tick_params(axis='x',which='major',labelsize=7,pad=1,length=2.5,
                   labelbottom=row==3)
    ax.tick_params(axis='x',which='minor',length=1.3)
    ax.set_xlabel('UTC date' if row==3 else '',fontsize=7,labelpad=2)

def colourbar(fig,img,x,y,h,ticks,unit):
    bar=fig.colorbar(img,cax=axes(fig,x,y,.12,h),ticks=ticks)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
    bar.ax.tick_params(direction='in',labelsize=7,pad=1,length=2)
    bar.set_label(unit,fontsize=7,labelpad=2)
    COLORBARS.append(bar)
    return bar

def wave_panel(fig,ax,band,kind,row,col):
    core=wave.core
    if kind=='net_proxy':
        norm,cmap=SymLogNorm(1,vmin=-200,vmax=200),'RdBu_r'
        ticks=[-200,-10,-1,0,1,10,200]
        unit=r'K m s$^{-1}$'
        ylim,yticks,name=(40,90),np.arange(40,91,10),'Net W1 proxy'
    else:
        norm,cmap=Normalize(0,band['amplitude_max']),'viridis'
        ticks=([0,500,1000,1500,1800] if col==0 else
               [0,500,1000,1500,2000,2500])
        unit='m'
        ylim,yticks,name=(20,90),np.arange(20,91,10),'W1 amplitude'
    # Both source products are daily means. Place each daily cell at its
    # 00--24 UTC interval so its centre and event markers match the PV panels.
    day_centres=core.DAYS+pd.Timedelta(hours=12)
    img=ax.pcolormesh(core.linked.edges(mdates.date2num(day_centres)),
        core.linked.edges(band[kind+'_height']),
        np.ma.masked_invalid(band[kind].T),cmap=cmap,norm=norm,shading='flat')
    ax.set_xlim(core.DAYS[0],core.DAYS[-1]+pd.Timedelta(days=1))
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(wave.daily.DATE_TICKS+pd.Timedelta(hours=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,
                   labelsize=7,length=2.5,pad=1,labelbottom=row==3)
    ax.tick_params(axis='x',which='minor',direction='in',top=True,length=1.3)
    ax.set_ylabel('Approx. alt. (km)' if col==0 else '',fontsize=7,labelpad=4)
    if row==3:ax.set_xlabel('UTC date',fontsize=7,labelpad=2)
    ax.axvline(wave.daily.COMMENCEMENT,color='white',ls='--',lw=.65,zorder=4)
    ax.axhline(50,color='white',ls='--',lw=.6,zorder=4)
    title(ax,chr(97+row*2+col),name,f'{band["lo"]}–{band["hi"]}°N')
    if kind=='net_proxy':
        for reg in core.linked.FOCUS_REGIONS:
            if tuple(reg['band'])==(band['lo'],band['hi']):
                start=mdates.date2num(pd.Timestamp(reg['start']))
                end=mdates.date2num(pd.Timestamp(reg['end'])+pd.Timedelta(days=1))
                low,high=reg['height']
                ax.add_patch(Rectangle((start,low),end-start,high-low,fill=False,
                    edgecolor=reg['color'],linewidth=1.1,zorder=8))
    colourbar(fig,img,XS[col]+PANEL_W+.10,ROWS[row],PANEL_H,ticks,
              unit if col==1 else '')


def draw_qy_mask(ax, lat, data):
    # Each shaded patch represents one native latitude/height grid cell.
    # Non-finite qy is left unshaded, not classified as positive.
    qy = np.asarray(data['background_qy'])
    z = np.asarray(data['height'])
    xedges = wave.core.linked.edges(lat)
    cells = []
    for j in range(len(lat)):
        for k in range(1, z.shape[1] - 1):
            if not np.isfinite(qy[j, k]) or qy[j, k] >= 0:
                continue
            if not np.all(np.isfinite(z[j, k-1:k+2])):
                continue
            z0 = (z[j, k-1] + z[j, k]) / 2
            z1 = (z[j, k] + z[j, k+1]) / 2
            cells.append([(xedges[j],z0),(xedges[j+1],z0),
                          (xedges[j+1],z1),(xedges[j],z1)])
    # Fill and outline use the same interpolation of the valid daily field,
    # so the dark contour is the boundary of the grey area.
    x = np.broadcast_to(lat[:, None], z.shape)
    field = np.ma.masked_invalid(qy)
    ax.contourf(x, z, field, levels=[-np.inf, 0], colors=['#c4c9d0'],
                alpha=.65, corner_mask=False, zorder=1)
    ax.contour(x, z, field, levels=[0], corner_mask=False,
               colors=['#555b63'], linewidths=.65, zorder=2)
    # Show missing q_y on the native grid, without extrapolating across
    # latitude or height. This is separate from valid nonnegative q_y.
    missing = []
    for j in range(len(lat)):
        for k in range(1, z.shape[1] - 1):
            if np.isfinite(qy[j, k]) or not np.all(np.isfinite(z[j, k-1:k+2])):
                continue
            z0=(z[j,k-1]+z[j,k])/2
            z1=(z[j,k]+z[j,k+1])/2
            missing.append([(xedges[j],z0),(xedges[j+1],z0),
                            (xedges[j+1],z1),(xedges[j],z1)])
    ax.add_collection(PolyCollection(missing,facecolor='none',edgecolor='#8794a0',
                      linewidths=0,hatch='///',alpha=.38,zorder=.8,clip_on=True))
    return len(cells)

# Reuse the validated vector rendering, changing only its grid indexing.
_src = inspect.getsource(wave.daily_map).replace('divmod(index, 3)', 'divmod(index, 2)')
exec(_src, wave.__dict__)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7,
        'axes.labelsize':7,'axes.linewidth':.55,'hatch.linewidth':.5,
        'xtick.direction':'in','ytick.direction':'in','svg.fonttype':'none'})
    wave.W,wave.H=W,H
    wave.MX,wave.MW,wave.MH,wave.MY=MAP_X,MAP_W,MAP_H,MAP_Y
    m.WIDTH,m.HEIGHT,m.PANEL_H=W,H,PANEL_H
    m.zero_contour=env.zero_contour
    m.calendar=calendar
    m.title=title
    m.response_time_box=lambda *args:None
    env.NEW_QY_EDGE,env.NEW_QY_FACE=BLUE,'none'
    events=json.loads(m.EVENTS_PATH.read_text(encoding='utf8'))
    data=m.load()
    masks=m.n2_masks(data)
    wave.core.linked.FOCUS_REGIONS=[dict(r,height=(52.,r['height'][1]),
        end='2025-01-10' if tuple(r['band'])==(60,80) else r['end'])
        for r in wave.core.linked.FOCUS_REGIONS]
    bands=wave.core.load_tables()[1:]
    lat,pressure,levels,fields,prior=wave.daily.load_fields()
    fig=plt.figure(figsize=(18.0/2.54,(H*18.0/W)/2.54))
    for col,(region,conf) in enumerate(m.REGIONS.items()):
        for row,kind in enumerate(['gradient','wind','net_proxy','amplitude']):
            ax=axes(fig,XS[col],ROWS[row],PANEL_W,PANEL_H)
            if kind=='gradient':
                img,top=m.gradient_panel(ax,data,conf,region,chr(97+2*row+col),row,events)
                colourbar(fig,img,XS[col]+PANEL_W+.10,ROWS[row],PANEL_H,
                          [-12,-6,0,6,12],r'K / 1000 km' if col==1 else '')
            elif kind=='wind':
                grad,height=m.band_mean(data,'T_gradient',conf['band'])
                top=float(height[np.isfinite(grad).all(0)].max())
                img=env.wind_panel(ax,data,conf,region,chr(97+2*row+col),row,top,events,masks)
                colourbar(fig,img,XS[col]+PANEL_W+.10,ROWS[row],PANEL_H,
                          [-40,-20,0,40,80],r'm s$^{-1}$' if col==1 else '')
            else:
                wave_panel(fig,ax,bands[col],kind,row,col)
            if kind in ['gradient','wind']:
                ax.set_ylabel('Approx. alt. (km)' if col==0 else '',fontsize=7,labelpad=4)
                ax.tick_params(axis='y',labelleft=True,labelsize=7,length=2.5,pad=1)
                for text in ax.texts:
                    if text.get_gid()=='shear-label':text.set_fontsize(7)
    cmap=wave.core.linked.reference_arrow_cmap()
    norm=PowerNorm(gamma=.5,vmin=prior['magnitude_limits'][0],vmax=prior['magnitude_limits'][1])
    counts=[]
    selected=[(day,q) for day,q in zip(wave.daily.MAP_DAYS,fields) if pd.Timestamp('2025-01-06') <= day <= pd.Timestamp('2025-01-11')]
    assert len(selected)==6
    mask_counts=[]
    for i,(day,q) in enumerate(selected):
        ax,artist,count=wave.daily_map(fig,lat,levels,q,day,i,norm,cmap)
        ax.set_ylabel('')
        mask_counts.append(draw_qy_mask(ax,lat,q))
        # Keep all longitude ticks while preventing adjacent 80° and 20° labels
        # from touching after compacting the columns.
        if i//2==2:
            for tick,value in zip(ax.get_xticklabels(),ax.get_xticks()):
                if value==20:tick.set_ha('left')
                elif value==80:tick.set_ha('right')
        for t in ax.texts:
            if t.get_text() in list('efghijklm') and t.get_fontweight()=='bold':
                t.set_text(chr(105+i))
        counts.append(count)
    lat_label=fig.text((MAP_X[0]+(MAP_X[-1]+MAP_W-MAP_X[0])/2)/W,.76/H,
             'Latitude (°N)',ha='center',va='bottom',fontsize=7)
    fig.text(11.15/W,(MAP_Y[-1]+(MAP_Y[0]+MAP_H-MAP_Y[-1])/2)/H,
             'Approx. alt. (km)',rotation=90,ha='center',va='center',fontsize=7)
    cax=axes(fig,18.43,MAP_Y[-1],.12,MAP_Y[0]+MAP_H-MAP_Y[-1])
    epbar=fig.colorbar(artist,cax=cax,ticks=[0,50,150,300,550])
    epbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos:f'{value:g}'))
    epbar.ax.tick_params(labelsize=7,length=2,pad=1,direction='in')
    epbar.set_label(r'$|F|$ (m$^2$ s$^{-2}$)',fontsize=7,labelpad=2.5)
    COLORBARS.append(epbar)
    fig.legend(handles=[Line2D([],[],color=m.PURPLE,lw=1,label=r'$U_g=0$'),
        Patch(facecolor='none',edgecolor=BLUE,hatch='///',label=r'New $\overline{q}_y<0$')],
        loc='upper center',bbox_to_anchor=(((XS[0]+XS[1]+PANEL_W)/2)/W,(ROWS[1]+PANEL_H+.91)/H),ncol=2,frameon=False,
        fontsize=7,handlelength=1.2,handletextpad=.3,columnspacing=.8,
        borderaxespad=0)
    right_legend=fig.legend(handles=[Patch(facecolor='#c4c9d0',edgecolor='#555b63',
                              label=r'$\overline{q}_y<0$'),
                        Patch(facecolor='none',edgecolor='#a9b2bb',hatch='///',
                              label=r'No $\overline{q}_y$ data')],
        loc='lower center',bbox_to_anchor=(((MAP_X[0]+MAP_X[-1]+MAP_W)/2)/W,.26/H),
        ncol=2,frameon=False,fontsize=7,handlelength=1.2,handletextpad=.3,
        columnspacing=.8,borderaxespad=0)
    fig.canvas.draw()
    for t in fig.findobj(match=Text):
        if t.get_gid()=='shear-label':t.set_fontsize(7)
        elif t.get_fontweight()=='bold' and t.get_text() in list('abcdefghijklmnopq'):
            t.set_fontsize(9)
        else:t.set_fontsize(8)
    for legend in fig.legends:
        for t in legend.get_texts(): t.set_fontsize(7)
    for bar in COLORBARS:
        bar.ax.tick_params(labelsize=7)
        bar.ax.title.set_fontsize(7)
        bar.ax.xaxis.label.set_fontsize(7)
        bar.ax.yaxis.label.set_fontsize(7)
        for label in bar.ax.get_xticklabels()+bar.ax.get_yticklabels(): label.set_fontsize(7)
    # Keep the exported width at 180 mm while placing the rightmost rendered
    # text within the fixed 180 mm width after increasing label spacing.
    for _ in range(8):
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        delta = 180/25.4 - (tight.x1 + .47/25.4)
        if abs(delta) < .0005:
            break
        fig.set_size_inches(fig.get_figwidth()+delta,fig.get_figheight(),forward=True)
    fig.canvas.draw()
    path=OUT/'environment_W1_EP_combined.png'
    # Crop only the bottom edge, retaining a 1 mm margin beneath the lowest text.
    bottom = max(0, fig.get_tightbbox(fig.canvas.get_renderer()).y0 + .074/25.4)
    export_box = Bbox.from_extents(0,bottom,180/25.4,fig.get_figheight())
    fig.savefig(path,dpi=600,bbox_inches=export_box,pad_inches=0)
    fig.savefig(OUT/'environment_W1_EP_combined.svg',bbox_inches=export_box,pad_inches=0)
    with Image.open(path) as im:
        assert abs(im.width-18.0/2.54*600)<2
        size=im.size
    plt.close(fig)
    (OUT/'figure_settings.json').write_text(json.dumps({'size_cm':[18.0,(fig.get_figheight()-bottom)*2.54],'dpi':600,
      'background_qy_mask':'new negative band mean, preperiod Dec28-30; selected 52-64 km or 65-85 km layers only',
      'background_latitudes':['40–60 N','60–70 N'],
      'wave_latitudes':['40–60 N','60–80 N'],
      'colorbars':'each left panel uses its original range; one shared EP colorbar on right',
      'W1_amplitude_color_limits_m':[[0,1800],[0,2500]],
      'ordinary_text_pt':8,'colorbar_text_pt':7,'panel_letters_pt':9,'legend_text_pt':7,
      'daily_EP_dates':'Jan 6–11','arrow_counts':counts,'negative_qy_cell_counts':mask_counts,
      'missing_qy':'Native-cell light grey hatch; no extrapolation. Missing q_y does not imply positive or negative values.',
      'pixels':size},
      ensure_ascii=False,indent=2),encoding='utf8')
    print(path)
    print('pixels',size,'counts',counts)

if __name__=='__main__':main()
