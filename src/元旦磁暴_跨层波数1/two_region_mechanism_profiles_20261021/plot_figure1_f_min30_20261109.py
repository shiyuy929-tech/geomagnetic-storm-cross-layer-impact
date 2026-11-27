"""Five time series at left and four independent chemistry panels at right."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedLocator
from PIL import Image
import plot_figure2_indices_mainphase_20261101 as src

HERE=Path(__file__).resolve().parent
OUT=HERE/'figure1_f_min30_20261109'
WIDTH,HEIGHT=18.0,14.7
LX,LW=1.35,9.4
RX,RW=12.0,9.0*3.5/6.5
DAYS,MID,START,STOP=src.DAYS,src.MID,src.START,src.STOP

def axis(fig,x,y,w,h,**kwargs):
    return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT],**kwargs)

def decorate(ax,title,ylabel='',show_dates=False,single_line=True,field=False,end=None):
    ax.set_xlim(START,STOP)
    ax.xaxis.set_major_locator(FixedLocator(md.date2num(MID[::2])))
    ax.xaxis.set_minor_locator(FixedLocator(md.date2num(MID[1::2])))
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %d' if single_line else '%b\n%d'))
    ax.tick_params(axis='both',which='major',direction='in',top=True,right=True,
                   length=2.8,width=.5,labelsize=7,pad=2,labelbottom=show_dates)
    ax.tick_params(axis='x',which='minor',direction='in',top=True,bottom=True,length=1.5,width=.4)
    letter,body=title[1],title[4:].strip()
    ax.text(0,1.04,letter,transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    if field:
        ax.text(.5,1.04,body,transform=ax.transAxes,fontsize=7,ha='center',va='bottom',gid='panel-title')
    if ylabel:ax.set_ylabel(ylabel,fontsize=7,labelpad=3)
    ax.set_axisbelow(not field)
    if not field:ax.grid(True,which='major',color='#9ca6b0',lw=.35,ls=':',alpha=.48)
    ax.axvline(src.SSC,color='#d62728',ls='--',lw=.8,zorder=10)

def main():
    OUT.mkdir(exist_ok=True)
    chem,particles,hourly,kp,dst,palette,hashes,minima,end=src.load()
    from ozone_night_daily_candidate_20261109 import calculate
    ozone_daily = calculate()
    ozone_daily.to_csv(OUT/'night_ozone_daily_means_and_counts.csv',index=False)
    chem['pressure_key']=chem.pressure_hPa.round(9)
    chem=chem.merge(ozone_daily[['date','band','pressure_key','dO3_daily_mean']],on=['date','band','pressure_key'],validate='one_to_one')
    chem['dO3']=chem['dO3_daily_mean']
    print('Daily mean O3: valid rows',chem.dO3.notna().sum(),'of',len(chem))
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7,
                         'axes.linewidth':.55,'savefig.facecolor':'white','svg.fonttype':'none'})
    fig=plt.figure(figsize=(WIDTH/2.54,HEIGHT/2.54))
    series=[('Dst','Ring-current disturbance',dst.time,dst.Dst_nT,(-250,50),[-200,-100,0],'Dst (nT)'),
            ('Kp','Planetary activity',kp.time+pd.Timedelta(hours=1.5),kp.Kp,(0,9),[0,3,6,9],'Kp'),
            ('AE','Auroral activity',hourly.index,hourly.aeValue,(0,1600),[0,800,1600],'AE (nT)'),
            ('PCN','Polar-cap activity',hourly.index,hourly.PCN,(-2,12),[0,6,12],'PCN (mV m⁻¹)')]
    left=[]
    for i,(name,title,x,y,ylim,ticks,ylabel) in enumerate(series):
        ax=axis(fig,LX,[12.10,9.87,7.64,5.41][i],LW,1.70,
                **({'sharex':left[0]} if left else {}))
        left.append(ax)
        if name=='Kp':
            edges=pd.DatetimeIndex(kp.time).append(pd.DatetimeIndex([kp.time.iloc[-1]+pd.Timedelta(hours=3)]))
            assert (kp.time.diff().dropna()==pd.Timedelta(hours=3)).all()
            ax.stairs(kp.Kp.to_numpy(),md.date2num(edges),color='#0000ff',linewidth=.65,baseline=None)
        else:
            ax.plot(x,y,color='#0000ff',lw=.65)
        ax.set_ylim(*ylim);ax.set_yticks(ticks)
        decorate(ax,f'({chr(97+i)}) {title}',ylabel,end=end)
        for text in ax.texts:
            text.set_y(1.015)

    eep=axis(fig,LX,1.40,LW,3.15,sharex=left[0])
    handles=[]
    for band in ['40-60','60-80']:
        for channel,ls,energy in [('e1','-',40),('e3','--',287)]:
            group=particles[(particles.band==band)&(particles.variable=='MEPED_'+channel)]
            values=group.set_index('date').reindex(DAYS.strftime('%Y-%m-%d')).value
            line,=eep.plot(MID,values,color=src.COLORS[band],ls=ls,lw=1.05,
                 marker='o',ms=2.4,markeredgewidth=.3,
                 label=f'{band.replace("-","–")}°N, >{energy} keV')
            if channel=='e3':
                line.set_dashes([2.0,1.2])
            handles.append(line)
    vals=particles.value.dropna()
    eep.set_yscale('log');eep.set_ylim(vals.min()/1.5,vals.max()*1.5)
    eep.set_yticks([1e2,1e3,1e4,1e5])
    from matplotlib.ticker import NullLocator
    eep.yaxis.set_minor_locator(NullLocator())
    decorate(eep,'(e) EEP proxy','Electron flux\n(cm⁻² s⁻¹ sr⁻¹)',show_dates=True,end=end)
    # Keep the panel heading close to the axis so the outside legend clears it.
    for text in eep.texts:
        text.set_y(1.0)
    eep.tick_params(axis='x',labelsize=7)
    eep.set_xlabel('UTC date | 2024–2025',fontsize=7,labelpad=3)
    legend=fig.legend(handles=handles,loc='upper right',bbox_to_anchor=((LX+LW+.10)/WIDTH,5.50/HEIGHT),
        ncol=2,frameon=False,fontsize=8,
        handlelength=2.0,handletextpad=.25,columnspacing=.4,labelspacing=.12,
        borderpad=.15,borderaxespad=.15)

    right=[]
    colorbars=[]
    for n,(band,field,ypos) in enumerate([('40-60','dO3',11.1875),('60-80','dO3',7.925),
                                          ('40-60','dT',4.6625),('60-80','dT',1.40)]):
        ax=axis(fig,RX,ypos,RW,2.6125,**({'sharex':right[0]} if right else {}))
        right.append(ax)
        tab=chem[chem.band.eq(band)]
        levels=tab.groupby('pressure_hPa').reference_height_km.first().sort_values()
        val=tab.pivot(index='date',columns='pressure_hPa',values=field).reindex(
            index=DAYS.strftime('%Y-%m-%d'),columns=levels.index).to_numpy()
        valid=tab.groupby('pressure_hPa').dO3.count().gt(0)
        ztop=float(levels[valid.reindex(levels.index)].max())
        height=levels.to_numpy()
        select=np.isfinite(height)&(height>42)&(height<ztop+8)
        old=next(s for s in palette['individual_panel_colour_scales'] if s['panel']==chr(102+n))
        low,high=old['colour_limits']
        colorbar_ticks=old['colourbar_ticks']
        if band=='60-80' and field=='dO3':
            high=100
            colorbar_ticks=[-70,-35,0,50,100]
        if band=='40-60' and field=='dO3':
            low=-30
            colorbar_ticks=[-30,-15,0,50,100]
        if field=='dO3':
            cmap=plt.get_cmap('BrBG').copy();cmap.set_bad('white')
            norm=Normalize(vmin=low,vmax=high,clip=True) if low>=0 else TwoSlopeNorm(vmin=low,vcenter=0,vmax=high)
        else:
            cmap=src.original.zero_white_cmap(low,high,'Figure1_temp_'+band)
            norm=Normalize(vmin=low,vmax=high)
        mesh=ax.pcolormesh(md.date2num(DAYS.append(pd.DatetimeIndex([STOP]))),
            src.original.edges(height[select]),np.ma.masked_invalid(val[:,select].T),
            shading='flat',norm=norm,cmap=cmap)
        name='Night O₃' if field=='dO3' else 'Night T'
        decorate(ax,f'({chr(102+n)}) {name} | {band.replace("-","–")}°N',
                 '',show_dates=n==3,single_line=False,field=True,end=end)
        ax.set_ylim(50,ztop);ax.set_yticks([z for z in [50,60,70,80,90] if z<=ztop])
        ax.tick_params(axis='x',labelsize=7)
        if n==3:ax.set_xlabel('UTC date | 2024–2025',fontsize=7,labelpad=3)
        bar=fig.colorbar(mesh,cax=axis(fig,RX+RW+.19,ypos,.15,2.6125),
                         ticks=colorbar_ticks,extend='neither')
        colorbars.append(bar)
        bar.ax.tick_params(direction='in',labelsize=7,pad=1.5,width=.5,length=2)
        bar.set_label('ΔO₃ (%)' if field=='dO3' else 'ΔT (K)',fontsize=9,labelpad=1.5)

    fig.text(11.20/ WIDTH,7.60/HEIGHT,'Approx. alt. (km)',rotation=90,
             fontsize=7,ha='center',va='center')

    # The 10 pt labels need fewer labeled dates at this physical panel width;
    # daily tick marks remain visible as minor ticks.
    labeled_days=pd.date_range('2024-12-29','2025-01-10',freq='3D')+pd.Timedelta(hours=12)
    eep.xaxis.set_major_locator(FixedLocator(md.date2num(labeled_days)))
    eep.xaxis.set_minor_locator(FixedLocator(md.date2num(MID)))
    right[-1].xaxis.set_major_locator(FixedLocator(md.date2num(labeled_days)))
    right[-1].xaxis.set_minor_locator(FixedLocator(md.date2num(MID)))

    from matplotlib.text import Text
    for text in fig.findobj(match=Text):
        panel_letter = text.get_text() in tuple('abcdefghi') and text.get_fontweight() == 'bold'
        text.set_fontsize(11 if panel_letter else 10)
        text.set_fontweight('bold' if panel_letter else 'normal')
    for subplot in [*left, eep, *right]:
        subplot.tick_params(axis='both',which='both',labelsize=9)
    for subplot in left:
        subplot.yaxis.label.set_fontsize(9)
        subplot.yaxis.labelpad=5
    for bar in colorbars:
        bar.ax.tick_params(labelsize=8)
        bar.ax.title.set_fontsize(9)
        bar.ax.xaxis.label.set_fontsize(9)
        bar.ax.yaxis.label.set_fontsize(9)
        bar.ax.yaxis.get_offset_text().set_fontsize(9)
    for text in legend.get_texts(): text.set_fontsize(8)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    assert abs(left[0].get_position().y1 - right[0].get_position().y1) < 1e-10
    assert abs(eep.get_position().y0 - right[-1].get_position().y0) < 1e-10
    for group in (left, right):
        gaps=[group[i].get_position().y0-group[i+1].get_position().y1
              for i in range(len(group)-1)]
        assert np.ptp(gaps) < 1e-10
    # Keep the bottom of the outside legend 0.6 mm above panel e.
    target_bottom = eep.bbox.y1 + fig.dpi * 0.6 / 25.4
    offset = (target_bottom - legend.get_window_extent(renderer).y0) / fig.bbox.height
    legend.set_bbox_to_anchor(((LX+LW+.10)/WIDTH,5.50/HEIGHT + offset),
                              transform=fig.transFigure)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    assert all(ax.bbox.height>left[0].bbox.height for ax in right)
    assert legend.get_window_extent(renderer).y0 > eep.bbox.y1
    assert legend.get_window_extent(renderer).y1 < left[-1].bbox.y0
    assert not any(legend.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in eep.texts)
    for name,h in hashes.items():assert src.sha(src.SOURCE/name)==h
    from matplotlib.transforms import Bbox
    # Retain point sizes while fitting visible artwork plus 2 mm padding
    # to an exact 180 mm output width.  The resulting height remains below
    # 170 mm, and no raster rescaling is applied.
    margin_in = 1 / 25.4
    left_margin_in = 1 / 25.4
    target_width_in = 180 / 25.4
    for _ in range(12):
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        delta = target_width_in - (tight.width + left_margin_in + margin_in)
        if abs(delta) < 0.0001:
            break
        fig.set_size_inches(fig.get_figwidth() + delta, fig.get_figheight(), forward=True)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    export = Bbox.from_extents(tight.x0-left_margin_in, tight.y0-margin_in,
                               tight.x0-left_margin_in+target_width_in, tight.y1+margin_in)
    assert export.height * 25.4 <= 170
    path=OUT/'Figure1_ozone_dailymean_candidate.png'
    fig.savefig(path,dpi=600,bbox_inches=export,pad_inches=0)
    fig.savefig(OUT/'Figure1_ozone_dailymean_candidate.svg',bbox_inches=export,pad_inches=0)
    with Image.open(path) as im:
        assert abs(im.width - target_width_in * 600) < 2
        print('pixels:', im.size, 'dpi:', im.info.get('dpi'))
    settings={'width_mm':180,'height_mm':round(export.height*25.4,2),'dpi':600,
        'padding_mm':{'left':1,'right':1,'top':1,'bottom':1},'ordinary_text_pt':10,'index_yaxis_title_pt':9,'index_yaxis_labelpad_pt':5,'axis_tick_text_pt':9,'bold_panel_letters_pt':11,'colorbar_tick_pt':8,'colorbar_label_pt':9,'index_line_colour':'blue','index_line_width_pt':0.65,'Kp_rendering':'native 3-hour interval stairs','electron_telescope':'0 degree','electron_curve_line_width_pt':1.05,'electron_markers':{'e1': 'circle','e3':'circle'},'electron_axis':'logarithmic',
        'storm_sudden_commencement_utc':str(src.SSC),'storm_marker_style':'red dashed, 0.8 pt','ozone_method':'nighttime SZA > 95 degrees, ozone-only QC and arithmetic mean, Dec 28-30 baseline; ozone recalculated','left_subplot_titles':False,
        'right_subplot_titles':'centred','pdf_generated':False}
    (OUT/'figure_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
    print('OUTPUT:',path)

if __name__=='__main__':main()
