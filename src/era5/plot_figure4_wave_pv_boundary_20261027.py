"""Upper-wave orientation, independent PV geometry, and an open PV budget.

Uses 6-hourly ERA5 for the horizontal-advection diagnosis. No W1 amplitude or
vertical-flux panels, no assumption that geometric correspondence is causal.
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
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
import cartopy.crs as ccrs
import plot_upper_wave_vortex_two_stage_20261026 as shared

OUT = ROOT/'results/Figure4_wave_PV_boundary_20261027'
DAYS, PRE = shared.DAYS, shared.PRE
EVENT = np.asarray((DAYS >= '2025-01-07') & (DAYS <= '2025-01-09'))
PC = ccrs.PlateCarree()
PURPLE, BLUE, ORANGE, GREEN = '#803a9b', '#167da3', '#ce761f', '#16805d'

def time_axis(ax, show=True):
    ax.set_xlim(DAYS[0]-pd.Timedelta(hours=12), DAYS[-1]+pd.Timedelta(hours=12))
    ax.set_xticks(pd.date_range('2024-12-29','2025-01-10',freq='3D'))
    ax.set_xticklabels([d.strftime('%b\n%d') for d in pd.date_range('2024-12-29','2025-01-10',freq='3D')])
    ax.axvspan(pd.Timestamp('2025-01-05 12:00'),pd.Timestamp('2025-01-10 12:00'),color='#d9ae42',alpha=.12,lw=0,zorder=-5)
    ax.axvline(shared.ONSET,color='#c74336',ls='--',lw=.8,zorder=2)
    ax.grid(axis='y',color='.85',ls=':',lw=.6,zorder=-3)
    ax.tick_params(top=True,right=True,length=3)
    if show: ax.set_xlabel('UTC date | 2024–2025',labelpad=3)
    else: ax.tick_params(labelbottom=False)

def orientation(waves):
    paths=[]
    for name,c,good in [('MLS ~50 km',waves['mls'],waves['mls_good']),
                        ('ERA5 3 hPa',waves['era'][:,np.flatnonzero(waves['p']==3)[0]],np.ones(len(DAYS),bool))]:
        p=shared.wrap(180-np.angle(c,deg=True)).astype(float)
        p[~good]=np.nan
        # Never connect the +/-180° seam; retain each observation as a point.
        for i in range(1,len(p)):
            if np.isfinite(p[i-1:i+1]).all() and abs(p[i]-p[i-1])>180:
                paths.append((name,i))
        yield name,p

def daily_outer_boundaries(source,pv,theta):
    """Reproduce panel (b)'s interpolated outer boundary at each longitude."""
    la,lo=source['lat'],source['lon']
    weights=shared.base.cell_weights(la,len(lo));threshold=pv['threshold']
    edges=np.full((len(DAYS),len(lo)),np.nan)
    for i,q in enumerate(pv['fields']):
        core,_=shared.region.main_component(q>=threshold,weights)
        if core[0].any(): raise ValueError(f'{DAYS[i]} high-PV core reaches southern data boundary')
        for j in range(len(lo)):
            ids=np.flatnonzero(core[:,j])
            if not len(ids): continue
            k=ids[0]
            if k==0: raise ValueError('Cannot interpolate boundary at the southernmost latitude')
            low,high=q[k-1,j],q[k,j]
            if not np.isfinite([low,high]).all() or high<=low:
                raise ValueError(f'Invalid PV crossing at {DAYS[i]}, lon={lo[j]}')
            edges[i,j]=la[k-1]+(threshold-low)/(high-low)*(la[k]-la[k-1])
    sector=(lo>=-130)&(lo<=-60)
    np.testing.assert_allclose(np.nanmean(edges[:,sector],axis=1),pv['edge_lat'],rtol=0,atol=1e-10)
    if not np.isfinite(edges[PRE|EVENT]).all():
        raise ValueError('Period maps need complete daily boundaries at all longitudes; do not bridge gaps')
    for name,select in [('Dec28-30',PRE),('Jan7-9',EVENT)]:
        map_sector=float(edges[select].mean(axis=0)[sector].mean())
        daily_sector=float(pv['edge_lat'][select].mean())
        np.testing.assert_allclose(map_sector,daily_sector,rtol=0,atol=1e-10)
        shared.QA.setdefault('panel_b_maps_consistency',{}).setdefault(str(theta),{})[name]={
            'map_sector_mean_degN':map_sector,'panel_b_period_mean_degN':daily_sector,
            'difference_deg':map_sector-daily_sector,'days':int(select.sum()),
            'sector_longitudes':int(sector.sum())}
    pd.DataFrame(edges,index=DAYS,columns=[f'{x:g}' for x in lo]).to_csv(
        OUT/f'PV_{theta}K_daily_outer_boundary_by_longitude.csv',index_label='date',encoding='utf-8-sig')
    pd.DataFrame({'longitude_degE':lo,'Dec28_30':edges[PRE].mean(0),'Jan07_09':edges[EVENT].mean(0)}).to_csv(
        OUT/f'PV_{theta}K_period_mean_outer_boundary.csv',index=False,encoding='utf-8-sig')
    return edges

def mean_centroid(geometry,selection):
    """Equal-day mean of daily centroid unit vectors on the sphere."""
    g=geometry.loc[selection]
    phi=np.deg2rad(g.centroid_lat.to_numpy());lam=np.deg2rad(g.centroid_lon.to_numpy())
    xyz=np.array([np.cos(phi)*np.cos(lam),np.cos(phi)*np.sin(lam),np.sin(phi)]).mean(axis=1)
    xyz/=np.linalg.norm(xyz)
    return {'centroid_lat':float(np.rad2deg(np.arcsin(xyz[2]))),
            'centroid_lon':float(np.rad2deg(np.arctan2(xyz[1],xyz[0])))}

def draw_map(ax,source,pv,edges,theta,panel,color,*,surface_unit='K',fill_core=True,
             coordinate_fontsize=8.5,coordinate_color='.25'):
    la,lo=source['lat'],source['lon']; w=shared.base.cell_weights(la,len(lo))
    th=np.linspace(0,2*np.pi,361)
    ax.set_boundary(MplPath(np.c_[np.sin(th),np.cos(th)]*.5+.5),transform=ax.transAxes)
    ax.set_extent([-180,180,20,90],crs=PC)
    r=abs(ax.projection.transform_point(0,20,PC)[1]);ax.set_xlim(-r,r);ax.set_ylim(-r,r)
    ax.set_xticks([]);ax.set_yticks([])
    ax.coastlines(resolution='110m',color='.35',lw=.55,zorder=5)
    ax.gridlines(crs=PC,xlocs=np.arange(-180,181,60),ylocs=[40,60,80],color='.3',linewidth=.5,alpha=.6,linestyle='--')
    cent=[]; lines=[]; plotted=[]; xc=np.r_[lo,180.]
    for selection,col,style in [(PRE,'.4','--'),(EVENT,color,'-')]:
        edge=edges[selection].mean(axis=0)
        projected=ax.projection.transform_points(PC,xc,np.r_[edge,edge[0]])[:,:2]
        if selection is EVENT and fill_core:
            ax.add_patch(Polygon(projected,closed=True,facecolor='#ede2f2' if theta==1200 else '#dcebf1',edgecolor='none',zorder=0))
        line,=ax.plot(projected[:,0],projected[:,1],color=col,linestyle=style,linewidth=1.65,zorder=7)
        lines.append(line);plotted.append(edge)
        g=mean_centroid(pv['geometry'],selection);cent.append(g)
        ax.plot(g['centroid_lon'],g['centroid_lat'],'o',ms=5,markeredgecolor=col,
                markerfacecolor='white' if selection is PRE else col,transform=PC,zorder=9)
    xy=[ax.projection.transform_point(g['centroid_lon'],g['centroid_lat'],PC) for g in cent]
    ax.annotate('',xy=xy[1],xytext=xy[0],arrowprops=dict(arrowstyle='->',color=color,lw=1.2),zorder=8)
    ax.plot(0,90,'+',color='.25',ms=5,transform=PC,zorder=9)
    for mer in [-130,-60]: ax.plot([mer,mer],[20,90],color=GREEN,ls=':',lw=1,transform=PC,zorder=6)
    # Tangential labels just outside both rims, matching the temperature maps.
    for longitude in [-120,-60,0,60,120,180]:
        angle=np.deg2rad(longitude)
        x=.5+.548*np.sin(angle);y=.5-.548*np.cos(angle)
        label='0°' if longitude==0 else ('180°' if longitude==180 else f'{abs(longitude)}°'+('W' if longitude<0 else 'E'))
        rotation=(longitude+90)%180-90
        ax.text(x,y,label,transform=ax.transAxes,ha='center',va='center',rotation=rotation,
                rotation_mode='anchor',fontsize=coordinate_fontsize,color=coordinate_color,clip_on=False)
    ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer()
    paths=[line.get_transform().transform_path(line.get_path()) for line in lines]
    label_boxes=[]
    for latitude in [40,60]:
        label=ax.text(0,latitude,f'{latitude}°N',transform=PC,ha='center',va='center',fontsize=coordinate_fontsize,color=coordinate_color,zorder=10,
                      bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.2))
        found=False
        for display_lat in ([latitude] if latitude==40 else [60,55,50,47.5,45,42.5,40]):
            for meridian in [30,60,90,120,150,180,-150,-120,-90,-60,-30,0]+list(range(-175,180,5)):
                label.set_position((meridian,display_lat));box=label.get_window_extent(renderer).padded(1)
                corners=np.array([[box.x0,box.y0],[box.x0,box.y1],[box.x1,box.y0],[box.x1,box.y1]])
                mid=np.array([(ax.bbox.x0+ax.bbox.x1)/2,(ax.bbox.y0+ax.bbox.y1)/2])
                if np.max(np.linalg.norm((corners-mid)/[ax.bbox.width,ax.bbox.height],axis=1))>=.485: continue
                if any(box.overlaps(other) for other in label_boxes): continue
                if any(path.intersects_bbox(box,filled=False) for path in paths): continue
                px,py=np.meshgrid(np.linspace(box.x0,box.x1,4),np.linspace(box.y0,box.y1,4))
                xy=ax.transData.inverted().transform(np.c_[px.ravel(),py.ravel()]);ll=PC.transform_points(ax.projection,xy[:,0],xy[:,1])
                lons=shared.wrap(ll[:,0]); lats=ll[:,1]
                if np.isfinite(lons).all() and np.isfinite(lats).all() and all(
                    np.all(lats<np.interp(lons,xc,np.r_[edge,edge[0]])) for edge in plotted):
                    found=True;break
            if found: break
        if not found: label.set_visible(False)
        else:
            label_boxes.append(box)
            if display_lat!=latitude:
                ax.annotate('',xy=ax.projection.transform_point(meridian,latitude,PC),
                            xytext=ax.projection.transform_point(meridian,display_lat,PC),
                            arrowprops=dict(arrowstyle='-',linestyle=':',color='.35',lw=.55,shrinkA=4,shrinkB=0),zorder=6)
    ax.set_title(f'({panel}) {theta} {surface_unit} (~{pv["height"]:.0f} km)',loc='left',pad=20,fontsize=11.5)
    ax.legend(handles=[Line2D([],[],color='.4',ls='--',lw=1.5,label='Dec 28–30'),
                       Line2D([],[],color=color,lw=1.7,label='Jan 7–9')],
              loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,fontsize=8.5,frameon=False,
              columnspacing=.6,handlelength=1.15,handletextpad=.35)
    return cent

def sixhour_budget(threshold, cached, lat, lon):
    """Fixed-PV contour tendency, Eulerian, and wind-harmonic horizontal terms.

    Terms at identical contour points use 6-hour u,v and full q gradients,
    before daily averaging. Residual is not classified as a physical process.
    """
    b=shared.base; times,p,la,lo,raw,_=b.load_6hourly()
    assert np.array_equal(la,lat) and np.array_equal(lo,lon)
    theta,q3=b.ertel_pv(p,la,raw)
    iso=b.interpolate_theta(theta,p,{'q':q3,'u':raw['u'],'v':raw['v']},[1200])
    q,u,v=[iso[k][:,0] for k in ['q','u','v']]
    reference=np.nanmean(q.reshape(15,4,*q.shape[1:]),axis=1)
    cached_q=cached['pv'][:,np.flatnonzero(cached['theta']==1200)[0]]
    difference=float(np.nanmax(np.abs(reference-cached_q)))
    if difference>1e-4: raise ValueError(f'6h reconstruction disagrees with daily PV cache: {difference}')
    q=np.stack([b.smooth_map(x) for x in q]);u=np.stack([b.smooth_map(x) for x in u]);v=np.stack([b.smooth_map(x) for x in v])
    lam=np.deg2rad(lo);phi=np.deg2rad(la);R=b.RADIUS
    qx=(np.roll(q,-1,axis=2)-np.roll(q,1,axis=2))/(2*np.deg2rad(2.5)*R*np.maximum(np.cos(phi)[None,:,None],1e-6))
    qy=np.gradient(q,phi,axis=1,edge_order=2)/R
    qt=np.gradient(q,.25,axis=0,edge_order=2)  # PVU per day; centered except endpoints
    pieces={}
    for k in [1,2]:
        e=np.exp(-1j*k*lam)[None,None,:]; ee=np.exp(1j*k*lam)[None,None,:]
        pieces[k]=tuple(2*np.real(np.mean(x*e,axis=2,keepdims=True)*ee) for x in [u,v])
    vals={k:np.full(len(times),np.nan) for k in ['observed','total','w1','w2','w0','other']}
    count=np.zeros(len(times),int);weights=b.cell_weights(la,len(lo));sector=np.flatnonzero((lo>=-130)&(lo<=-60))
    for t in range(len(times)):
        core,_=shared.region.main_component(q[t]>=threshold,weights)
        samples={k:[] for k in vals}
        for j in sector:
            ids=np.flatnonzero(core[:,j]);
            if not len(ids) or ids[0]==0: continue
            n=ids[0]; a,z=q[t,n-1,j],q[t,n,j]
            if not np.isfinite([a,z]).all() or z<=a: continue
            f=(threshold-a)/(z-a)
            if not 0<=f<=1: continue
            def interp(x): return (1-f)*x[n-1,j]+f*x[n,j]
            gx,gy=interp(qx[t]),interp(qy[t]); ob=interp(qt[t])
            if not np.isfinite([gx,gy,ob]).all() or np.hypot(gx,gy)<1e-7: continue
            U,V=interp(u[t]),interp(v[t]); terms=[]
            for k in [1,2]:
                U1,V1=(interp(x[t]) for x in pieces[k]);terms.append(-86400*(U1*gx+V1*gy))
            u0=np.nanmean(u[t],axis=1);v0=np.nanmean(v[t],axis=1)
            um=(1-f)*u0[n-1]+f*u0[n];vm=(1-f)*v0[n-1]+f*v0[n]
            tot=-86400*(U*gx+V*gy); zero=-86400*(um*gx+vm*gy)
            row=[ob,tot,*terms,zero,tot-zero-sum(terms)]
            if not np.isfinite(row).all(): continue
            for k,value in zip(vals,row): samples[k].append(value)
        count[t]=len(samples['observed'])
        if count[t]>=10:
            for k in vals: vals[k][t]=np.mean(samples[k])
    df=pd.DataFrame({'time_UTC':pd.to_datetime(times),'count':count,**vals})
    daily=df.groupby(df.time_UTC.dt.date).agg({**{k:'mean' for k in vals},'count':'min'})
    daily.index=pd.to_datetime(daily.index); daily=daily.reindex(DAYS)
    # Never imply budget closure: total horizontal term excludes diabatic / friction / analysis effects.
    daily['residual']=daily['observed']-daily['total']
    assert np.allclose(daily.total,daily.w0+daily.w1+daily.w2+daily.other,atol=1e-9,equal_nan=True)
    df.to_csv(OUT/'PV_1200K_boundary_transport_6hourly.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(OUT/'PV_1200K_boundary_transport_daily.csv',index_label='date',encoding='utf-8-sig')
    shared.QA['PV_transport']={'coordinate':'1200 K; fixed pre-period PV threshold',
        'units':'PVU day-1; sector mean evaluated at same moving contour; 130-60 W',
        'method':'6-hourly u and v zonal Fourier components acting on full PV gradient; then daily mean',
        'excluded':'cross-isentropic transport, friction, analysis increments and numerical effects remain in residual',
        'gradient_min_PVU_per_m':1e-7,'min_sector_meridians':10,'daily_min_valid_meridians':int(daily['count'].min()),
        'cache_PVU_max_error':difference,'harmonics_sum_closed':True,'budget_physically_closed':False}
    return daily

def plot(waves,source,pv,budget,edges):
    fig_height_cm=12.5
    fig=plt.figure(figsize=(20/2.54,fig_height_cm/2.54))
    a=fig.add_axes([.085,.68,.40,.26]); b=fig.add_axes([.56,.68,.425,.26])
    map_width=.19
    map_height=(20*map_width)/fig_height_cm
    c=fig.add_axes([.045,.16,map_width,map_height],projection=ccrs.NorthPolarStereo(central_longitude=0))
    d=fig.add_axes([.30,.16,map_width,map_height],projection=ccrs.NorthPolarStereo(central_longitude=0))
    e=fig.add_axes([.585,.16,.40,map_height])
    for (name,p),color,ls in zip(orientation(waves),[PURPLE,BLUE],['-','--']):
        shown=p.copy();shown[np.r_[False,np.abs(np.diff(p))>180]]=np.nan
        a.plot(DAYS,shown,color=color,ls=ls,lw=1.8,label=name)
        a.scatter(DAYS,p,color=color,s=10,zorder=4)
    a.axhspan(-130,-60,color=GREEN,alpha=.12,zorder=-4)
    a.set_ylim(-180,180);a.set_yticks([-180,-90,0,90,180]);a.set_yticklabels(['180°','90°W','0°','90°E','180°'])
    a.set_ylabel('Longitude',labelpad=3);a.set_title('(a) W1 trough position',loc='left',pad=7)
    time_axis(a);a.legend(loc='upper left',fontsize=9,frameon=False,handlelength=1.3)
    for th,col,label in [(1200,PURPLE,'1200 K (~36 km)'),(450,BLUE,'450 K (~19 km)')]:
        b.plot(DAYS,pv[th]['edge_lat'],color=col,lw=1.8,label=label)
        b.axhline(pv[th]['edge_lat'][PRE].mean(),color=col,ls=':',lw=.9)
    b.set_ylim(40,85);b.set_yticks([40,50,60,70,80]);b.set_ylabel('Latitude (°N)',labelpad=3)
    b.set_title('(b) North American PV edge',loc='left',pad=7);time_axis(b)
    b.legend(loc='lower left',fontsize=9,frameon=False,handlelength=1.4)
    centers={str(th):draw_map(ax,source,pv[th],edges[th],th,letter,col)
             for th,ax,letter,col in [(1200,c,'c',PURPLE),(450,d,'d',BLUE)]}
    for key,col,ls,label in [('observed','.15','-','PV rate'),('total',PURPLE,'--','Advection'),
                             ('w1',BLUE,'-','W1 wind'),('w2',ORANGE,'-.','W2 wind')]:
        e.plot(DAYS,budget[key]/1000,color=col,ls=ls,lw=1.55,label=label)
    e.axhline(0,color='.4',lw=.7);e.set_ylabel('PV tendency\n'+r'($10^3$ PVU day$^{-1}$)',labelpad=3)
    e.set_title('(e) PV tendency',loc='left',pad=7);time_axis(e)
    e.text(.99,1.045,'1200 K',transform=e.transAxes,ha='right',va='bottom',fontsize=9,clip_on=False)
    e.legend(loc='upper right',bbox_to_anchor=(.99,.99),ncol=2,fontsize=8.7,frameon=False,
             columnspacing=.6,handlelength=1.15,handletextpad=.35,borderaxespad=.25)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in [a,b,c,d,e]:
        bounds=ax._left_title.get_window_extent(renderer)
        if bounds.x0<0 or bounds.y0<0 or bounds.x1>fig.bbox.width or bounds.y1>fig.bbox.height:
            raise RuntimeError(f'Panel title clipped: {ax._left_title.get_text()}')
    shared.save(fig,'Figure4_wave_PV_boundary.png')
    return centers

def main():
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT;shared.QA={};shared.style()
    plt.rcParams.update({'axes.titlesize':11.5,'axes.labelsize':10.5,
                         'xtick.labelsize':10,'ytick.labelsize':10})
    waves=shared.load_waves(); source,pv=shared.load_pv()
    edges={th:daily_outer_boundaries(source,pv[th],th) for th in [1200,450]}
    shared.QA['PV_method']['map']='Equal-day mean of interpolated daily outer boundary by longitude; centroid is equal-day mean of daily spherical centroid vectors.'
    budget=sixhour_budget(pv[1200]['threshold'],source,source['lat'],source['lon'])
    centers=plot(waves,source,pv,budget,edges)
    shared.QA['map_centers']=centers
    shared.QA['windows']={'baseline':'2024-12-28/30','main_highlight':'2025-01-06/10','map':'2025-01-07/09'}
    (OUT/'method_checks.json').write_text(json.dumps(shared.QA,ensure_ascii=False,indent=2),encoding='utf-8')
    write_notes(pv,budget)
    print('Saved:',OUT,flush=True)

def write_notes(pv,budget):
    event=budget.loc['2025-01-07':'2025-01-09']
    text=f'''# 图注与结果说明

**图4. 2024年12月28日至2025年1月11日上层行星波槽位、平流层极涡边界及北美扇区PV水平输送。**（a）60°–80°N纬度带位势高度W1槽位的经度：MLS约50 km（紫色实线）和ERA5 3 hPa（蓝色虚线，事件参考高度约36.6 km）。绿色带为130°–60°W；弱振幅及未通过MLS局地时/经度采样敏感性筛查的相位不显示。（b）130°–60°W扇区逐日主要高PV区外边界的平均纬度，紫色为1200 K（约36 km），蓝色为450 K（约19 km）；水平点线为各自12月28–30日每日指标的均值。（c、d）ERA5 1200 K和450 K等熵面的固定阈值高PV主体平均外边界与平均质心：灰色虚线/空心点为12月28–30日，彩色实线/实心点为1月7–9日；绿色点线为130°W和60°W。对每日高PV主体，在各经线提取最南外边界，再在每条经线上对所选时段等权平均；点为每日面积质心单位向量的等权球面平均。等熵面实际高度和气压随时空变化。（e）1200 K、同一固定PV等值线上的北美扇区平均欧拉PV局地倾向（黑色）、总水平平流倾向−u·∇q（紫色虚线），以及W1和W2水平风分别作用于完整PV梯度的分项（蓝、橙线）。每6小时先计算，再对当天有效样本求均值。红色竖虚线为12月31日16:18 UTC，浅黄色区为1月6–10日主要反射研究窗口。

图（a）展示波槽方向，不重新绘制图1已有的W1振幅和垂直传播通量。MLS与ERA5来源及高度不同，不把两层视为同一观测层。图（b–d）统一使用每日PV外边界：图（c、d）对每条经线的逐日边界纬度求时段平均，图（b）对每日扇区边界纬度求平均；两个时段在全部经线上均有有效边界，因此两高度地图的北美扇区时段平均与图（b）严格一致。图（c、d）填色仅用于显示平均外边界包络，其曲线不必是某一天或平均PV场的实际等值线；平均质心来自每日主体质心，不是对平均包络重新求质心。

## 重要计算与判读边界

- 固定PV阈值为参考期每日诊断阈值的均值，格点为2.5°，空间平滑为0.55格；外边界通过相邻格点插值，精度仍受原网格限制。仅保留最大周期连通的高PV区域；这是一种客观几何诊断，敏感性应结合其他边界定义检查。
- 图（e）的W1/W2是将每时次u、v分解后作用于**完整PV梯度**的水平平流分项，并非波数间的能量转移，也不是单独关闭W1/W2的对照试验。均流及W3及以上项没有画出，但保留在逐日CSV中；其与W1、W2之和数值上等于总水平项。
- 黑线为固定地理位置上的欧拉偏导数，随后在随时间变化的同一PV等值线上取样；不是质点沿边界的全导数，也不是等值线的法向速度。6小时中心差分端点使用单边差分，每日需至少10条有效经线（130°–60°W共29条），同时过滤过弱梯度。
- 水平项并未构成闭合PV收支。实际倾向与水平项的差包括跨等熵输送、非绝热/摩擦、分析增量及数值误差等，不能单一归因。1月7–9日平均，实际局地倾向约{event['observed'].mean():.0f} PVU/day，总水平项约{event['total'].mean():.0f} PVU/day，差约{event['residual'].mean():.0f} PVU/day；W1和W2风场分项分别约{event['w1'].mean():.0f}和{event['w2'].mean():.0f} PVU/day，存在其他分项的相互抵消。
- 时间/空间对应及同一资料场的分解不是独立因果试验。图中不能单独证明反射波驱动极涡，也不能凭净通量论断50 km以下没有下行分量；需进一步比较垂直连接、PV完整收支和有/无上层扰动的对照试验。

## 输出数据

逐日槽位：`W1_vertical_amplitude_phase.csv`（文件保留全部已有计算列，图中只展示相位）。逐日PV边界：`PV_geometry_daily.csv`；各经线逐日外边界及两个时段的平均：`PV_1200K_daily_outer_boundary_by_longitude.csv`、`PV_1200K_period_mean_outer_boundary.csv`及对应450 K文件。6小时和逐日PV输送：`PV_1200K_boundary_transport_6hourly.csv`、`PV_1200K_boundary_transport_daily.csv`。方法和质量检查：`method_checks.json`。
'''
    (OUT/'图注与结果说明.md').write_text(text,encoding='utf-8')

if __name__=='__main__': main()
