"""Event screen of upper-vortex changes and subsequent North American weather.

No statistical or dynamical counterfactual is inferred from one event.
All regions and windows below are fixed before inspecting the extended data.
"""
from pathlib import Path
import sys, json, hashlib
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'plot_packages'),str(ROOT/'python_packages')]
import numpy as np
import pandas as pd
import h5py

WAVE = ROOT.parents[1]/'元旦磁暴_跨层波数1'
DATA = ROOT/'data/stratosphere_troposphere_followup_20261029'
OUT = ROOT/'results/upper_vortex_followup_20261029'
G, A, OMEGA, KAPPA = 9.80665, 6371000., 7.292115e-5, 287.05/1004.
START, END = np.datetime64('2024-12-28'), np.datetime64('2025-01-31')
BOXES = {
    'NA_sector':(-130,-60,45,75),
    'Alaska_ridge':(-170,-140,55,75),
    'NA_trough':(-105,-65,35,55),
    'main':(-105,-70,35,45), 'south':(-107,-80,25,37),
    'footprint':(-130,-60,25,60),
}
WINDOWS = {
    'pre_Dec28_30':('2024-12-28','2024-12-30'),
    'early_Jan01_03':('2025-01-01','2025-01-03'),
    'main_Jan06_09':('2025-01-06','2025-01-09'),
    'core_Jan07_09':('2025-01-07','2025-01-09'),
    'after_Jan10_12':('2025-01-10','2025-01-12'),
    'after_Jan13_15':('2025-01-13','2025-01-15'),
    'context_Jan16_20':('2025-01-16','2025-01-20'),
    'context_Jan21_25':('2025-01-21','2025-01-25'),
    'context_Jan26_31':('2025-01-26','2025-01-31'),
}
INPUTS = []
QA = {}


def window(days, limits):
    return (days >= np.datetime64(limits[0])) & (days <= np.datetime64(limits[1]))


def read_h5(path, names):
    INPUTS.append(path)
    with h5py.File(path) as f:
        tt = f['valid_time'][:].astype('datetime64[s]')
        p, lat = f['pressure_level'][:], f['latitude'][:]
        lon = (f['longitude'][:]+180)%360-180
        iy, ix, ip = np.argsort(-lat), np.argsort(lon), np.argsort(p)
        fields = {k:f[k][:].astype(np.float32)[:,ip][:,:,iy][:,:,:,ix] for k in names}
    return tt,p[ip],lat[iy],lon[ix],fields


def load():
    early = WAVE/'coupling_priority_tests_20261023/early_6h_cache.npz'
    INPUTS.append(early)
    with np.load(early) as f:
        tt = f['time'].astype('datetime64[s]')
        use = (tt >= START) & (tt < np.datetime64('2025-01-16')) & (tt < END+np.timedelta64(1,'D'))
        first_times, p, lat, lon = tt[use],f['p'],f['lat'],(f['lon']+180)%360-180
        ix = np.argsort(lon); lon=lon[ix]
        fields = {k:f[k][use].astype(np.float32)[...,ix] for k in ['z','t','u','v']}
    times=first_times
    if END >= np.datetime64('2025-01-16'):
        late = WAVE/'coupling_priority_tests_20261023/data/upper_Jan16_31.nc'
        lt,lp,ly,lx,ld = read_h5(late, ['z','t','u','v'])
        np.testing.assert_array_equal(p,lp);np.testing.assert_array_equal(lat,ly);np.testing.assert_array_equal(lon,lx)
        use=lt<END+np.timedelta64(1,'D')
        times = np.r_[first_times,lt[use]]
        fields = {k:np.concatenate([fields[k],ld[k][use]]) for k in fields}
    lower_parts=[]; lower_times=[]
    for name,first in [('lower_Dec28_31','2024-12-28'),('lower_Jan01_08','2025-01-01'),
                       ('lower_Jan09_16','2025-01-09'),('lower_Jan17_24','2025-01-17'),('lower_Jan25_31','2025-01-25')]:
        if np.datetime64(first)>END:continue
        t0,p0,y0,x0,d0 = read_h5(DATA/(name+'.nc'), ['z','t','u','v'])
        np.testing.assert_array_equal(y0,lat);np.testing.assert_array_equal(x0,lon)
        use=t0<END+np.timedelta64(1,'D')
        lower_times.append(t0[use]); lower_parts.append({k:v[use] for k,v in d0.items()})
    np.testing.assert_array_equal(np.concatenate(lower_times),times)
    expected = np.arange(START.astype('datetime64[h]'), (END+np.timedelta64(1,'D')).astype('datetime64[h]'),np.timedelta64(6,'h'))
    np.testing.assert_array_equal(times,expected)
    fullp=np.r_[p,p0];order=np.argsort(fullp)
    fields={k:np.concatenate([fields[k],np.concatenate([v[k] for v in lower_parts])],axis=1)[:,order]
            for k in fields}
    p=fullp[order]
    assert len(np.unique(p)) == len(p)
    days=times[::4].astype('datetime64[D]')
    QA.update(time_UTC=[str(times[0]),str(times[-1])],pressure_hPa=p.tolist(),
              native_grid_deg=2.5,all_days_four_six_hourly_samples=True,
              main_window=['2025-01-06','2025-01-09'],primary_followup_end='2025-01-15',
              later_days='context only; not attributed to the earlier reflection event')
    return times,days,p,lat,lon,fields


def daily(a):
    return a.reshape((-1,4)+a.shape[1:]).mean(axis=1)


def roi(lat,lon,box):
    west,east,south,north=box
    return ((lat[:,None]>=south)&(lat[:,None]<=north)&
            (lon[None,:]>=west)&(lon[None,:]<=east))


def spatial_mean(a,lat,lon,box,extra=None):
    mask=roi(lat,lon,box)
    if extra is not None: mask=mask&extra
    w=np.broadcast_to(np.cos(np.deg2rad(lat))[:,None],a.shape[-2:])*mask
    good=np.isfinite(a)
    den=np.sum(w*good,axis=(-2,-1))
    return np.divide(np.sum(np.where(good,a,0)*w,axis=(-2,-1)),den,
                     out=np.full(den.shape,np.nan),where=den>0)


def weighted_r(a,b,weights):
    good=np.isfinite(a)&np.isfinite(b)&(weights>0)
    w=weights[good];x=a[good];y=b[good]
    if len(x)<10: return np.nan
    x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
    den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    return float(np.sum(w*x*y)/den) if den>1e-8 else np.nan


def read_sp(times,lat,lon):
    arrays=[];stamps=[]
    for path in [WAVE/'aligned_daily_crosslevel_20261024/data/surface_pressure_Dec28_31_2024.nc',
                 WAVE/'coupling_priority_tests_20261023/data/surface_Jan01_31_contents/data_stream-oper_stepType-instant.nc']:
        INPUTS.append(path)
        with h5py.File(path) as f:
            lt=f['valid_time'][:].astype('datetime64[s]')
            keep=np.isin(lt,times)
            ly=f['latitude'][:];lx=(f['longitude'][:]+180)%360-180;ix=np.argsort(lx)
            np.testing.assert_array_equal(ly,lat);np.testing.assert_array_equal(lx[ix],lon)
            stamps.append(lt[keep]);arrays.append(f['sp'][:][keep].astype(float)[...,ix])
    np.testing.assert_array_equal(np.concatenate(stamps),times)
    return np.concatenate(arrays)


def land_mask(lat,lon):
    from cartopy.io import shapereader
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from shapely.geometry import Point
    path=WAVE/'news_storm_map_20261024/natural_earth/ne_50m_admin_1_states_provinces.shp'
    INPUTS.append(path)
    union=prep(unary_union([r.geometry for r in shapereader.Reader(path).records()
                if r.attributes['admin'] in ['United States of America','Canada','Mexico']]))
    return np.array([[union.covers(Point(float(x),float(y))) for x in lon] for y in lat])


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'predefined_design.json').write_text(json.dumps(dict(boxes=BOXES,windows=WINDOWS,
        primary_lag_range_days=[-4,4],negative_advection_thresholds_K_day=[-1,-2,-3],
        primary_scope='event screening, not causal-effect estimation',
        no_centered_temporal_smoothing=True,no_NAM_claim_without_climatology=True),indent=2),encoding='utf-8')
    times,days,p,lat,lon,d=load()
    pre=window(days,WINDOWS['pre_Dec28_30'])
    z=daily(d['z'].astype(float)/G)
    t=daily(d['t'].astype(float))
    u=daily(d['u'].astype(float))
    eddy=z-z.mean(-1,keepdims=True)
    zchange=z-z[pre].mean(0)
    asym=spatial_mean(eddy,lat,lon,BOXES['NA_sector'])
    ridge=spatial_mean(z,lat,lon,BOXES['Alaska_ridge'])
    trough=spatial_mean(z,lat,lon,BOXES['NA_trough'])
    dipole=ridge-trough
    dipole_change=dipole-dipole[pre].mean(0)
    eddy_dipole=spatial_mean(eddy,lat,lon,BOXES['Alaska_ridge'])-spatial_mean(eddy,lat,lon,BOXES['NA_trough'])
    eddy_dipole_change=eddy_dipole-eddy_dipole[pre].mean(0)
    polar=spatial_mean(z,lat,lon,(-180,180,60,90))
    polar_change=polar-polar[pre].mean(0)
    i60=int(np.flatnonzero(lat==60)[0])
    records=[];patterns=[]
    p500=int(np.flatnonzero(p==500)[0])
    rweights=np.broadcast_to(np.cos(np.deg2rad(lat))[:,None],eddy.shape[-2:])*roi(lat,lon,BOXES['NA_sector'])
    extra_weights={name:np.broadcast_to(np.cos(np.deg2rad(lat))[:,None],eddy.shape[-2:])*
        roi(lat,lon,(-130,-60,south,north)) for name,south,north in [('40_60N',40,60),('60_80N',60,80)]}
    band=(lat>=60)&(lat<=80)
    band_fields=np.average(eddy[:,:,band],weights=np.cos(np.deg2rad(lat[band])),axis=2)
    for i,day in enumerate(days):
        for j,pp in enumerate(p):
            records.append(dict(date=str(day),pressure_hPa=float(pp),NA_eddy_Z_m=float(asym[i,j]),
                NA_Z_change_m=float(spatial_mean(zchange[i,j],lat,lon,BOXES['NA_sector'])),
                ridge_Z_m=float(ridge[i,j]),trough_Z_m=float(trough[i,j]),
                ridge_trough_change_m=float(dipole_change[i,j]),polar_Z_change_m=float(polar_change[i,j]),
                ridge_trough_eddy_change_m=float(eddy_dipole_change[i,j]),
                U60_m_s=float(u[i,j,i60].mean())))
            if pp<=300:
                patterns.append(dict(date=str(day),pressure_hPa=float(pp),
                    r_raw_eddy_vs_500=weighted_r(eddy[i,j],eddy[i,p500],rweights),
                    r_change_vs_500=weighted_r(zchange[i,j],zchange[i,p500],rweights),
                    r_NA_40_60N=weighted_r(eddy[i,j],eddy[i,p500],extra_weights['40_60N']),
                    r_NA_60_80N=weighted_r(eddy[i,j],eddy[i,p500],extra_weights['60_80N']),
                    r_global_band60_80N_longitude=float(np.corrcoef(band_fields[i,j],band_fields[i,p500])[0,1])))
    records=pd.DataFrame(records);patterns=pd.DataFrame(patterns)
    records.to_csv(OUT/'cross_layer_daily.csv',index=False,encoding='utf-8-sig')
    patterns.to_csv(OUT/'pattern_correlations_daily.csv',index=False,encoding='utf-8-sig')
    sp=read_sp(times,lat,lon)
    j850=int(np.flatnonzero(p==850)[0])
    good=sp>=85000.
    tt,uu,vv=[np.where(good,d[k][:,j850].astype(float),np.nan) for k in ['t','u','v']]
    phi=np.deg2rad(lat);cp=np.cos(phi)[None,:,None];cp=np.where(abs(cp)<1e-5,np.nan,cp)
    tx=(np.roll(tt,-1,-1)-np.roll(tt,1,-1))/(2*np.deg2rad(2.5)*A*cp)
    ty=np.gradient(tt,phi,axis=1,edge_order=2)/A
    zonal=-uu*tx*86400;mer=-vv*ty*86400;adv=zonal+mer
    mer=np.where(np.isfinite(adv),mer,np.nan);zonal=np.where(np.isfinite(adv),zonal,np.nan)
    fixed=np.isfinite(adv).all(axis=0)
    advd,merd,zond=daily(adv),daily(mer),daily(zonal)
    ud,vd,td=daily(uu),daily(vv),daily(tt)
    land=land_mask(lat,lon)
    footprint=roi(lat,lon,BOXES['footprint'])&fixed&land
    weights=np.broadcast_to(np.cos(phi)[:,None],footprint.shape)*footprint
    yy,xx=np.meshgrid(lat,lon,indexing='ij')
    weather=[]
    for i,day in enumerate(days):
        row=dict(date=str(day))
        for name in ['main','south']:
            for key,arr in [('advection',advd),('meridional',merd),('zonal',zond),('v850',vd),('T850',td)]:
                row[name+'_'+key]=float(spatial_mean(arr[i],lat,lon,BOXES[name],fixed))
        cooling=np.maximum(-merd[i],0)*weights
        row['cooling_centroid_lon_E']=float(np.nansum(cooling*xx)/np.nansum(cooling))
        row['cooling_centroid_lat_N']=float(np.nansum(cooling*yy)/np.nansum(cooling))
        for cutoff in [1,2,3]:
            row[f'land_fraction_meridional_below_minus{cutoff}']=float(np.sum(weights*(merd[i]<-cutoff))/weights.sum())
        weather.append(row)
    weather=pd.DataFrame(weather)
    j100=int(np.flatnonzero(p==100)[0])
    weather['NA_T100_K']=spatial_mean(t[:,j100],lat,lon,BOXES['NA_sector'])
    weather['polar_T100_K']=spatial_mean(t[:,j100],lat,lon,(-180,180,60,90))
    surface=WAVE/'troposphere_coupling_20261023/surface_temperature_anomaly_to_Jan31.npz'
    INPUTS.append(surface)
    with np.load(surface) as f:
        sdays=f['time'].astype('datetime64[D]');slat=f['lat'];slon=(f['lon']+180)%360-180
        si=np.array([int(np.flatnonzero(sdays==day)[0]) for day in days])
        for name in ['main','south']:
            weather[name+'_T2m_anomaly_K']=spatial_mean(f['anomaly_K'][si],slat,slon,BOXES[name])
    weather.to_csv(OUT/'cold_transport_daily.csv',index=False,encoding='utf-8-sig')
    # Compare independently obtained, previously validated instantaneous-advection daily cache.
    old=ROOT/'results/weather_key_checks_20261028/advection850_daily_fields.npz'
    INPUTS.append(old)
    with np.load(old) as f:
        idx=[int(np.flatnonzero(days==day)[0]) for day in f['time'].astype('datetime64[D]')]
        np.testing.assert_array_equal(lat,f['lat']);np.testing.assert_array_equal(lon,f['lon'])
        diff=advd[idx]-f['advection850']
        QA['advection_overlap_max_abs_difference_K_day']=float(np.nanmax(abs(diff)))
        QA['advection_overlap_RMS_difference_K_day']=float(np.sqrt(np.nanmean(diff**2)))
    QA['advection_formula']='Instantaneous -(u*dT/dx+v*dT/dy)*86400, then mean of four samples'
    QA['advection_masks']='sp>=850 hPa before derivatives; both terms use full-advection support; fixed valid cells across the analyzed window for regional comparisons'
    QA['fixed_support_fraction']={name:float(np.sum(np.cos(phi)[:,None]*roi(lat,lon,BOXES[name])*fixed)/
        np.sum(np.cos(phi)[:,None]*roi(lat,lon,BOXES[name]))) for name in ['main','south']}
    # A dry Ertel-PV crossing is a dynamical-tropopause proxy, not a thermal tropopause.
    pv,pp=pv_field(p,lat,lon,d,sp)
    trop,coverage=tp_crossings(pv,pp,lat,lon,days)
    trop.to_csv(OUT/'dynamical_tropopause_daily.csv',index=False,encoding='utf-8-sig')
    QA['tropopause_coverage']=coverage
    period_rows=[]
    for name,limits in WINDOWS.items():
        mask=window(days,limits)
        if not mask.any():continue
        row=dict(period=name,first=limits[0],last=limits[1],days=int(mask.sum()))
        for col in weather.columns[1:]: row[col]=float(weather.loc[mask,col].mean())
        period_rows.append(row)
    pd.DataFrame(period_rows).to_csv(OUT/'weather_period_summary.csv',index=False,encoding='utf-8-sig')
    layer_rows=[]
    for name,limits in WINDOWS.items():
        for pp0,group in records[records.date.between(*limits)].groupby('pressure_hPa'):
            row=dict(period=name,pressure_hPa=pp0)
            for col in ['NA_eddy_Z_m','NA_Z_change_m','ridge_trough_change_m','ridge_trough_eddy_change_m','polar_Z_change_m','U60_m_s']:
                row[col]=float(group[col].mean())
            layer_rows.append(row)
    pd.DataFrame(layer_rows).to_csv(OUT/'cross_layer_period_summary.csv',index=False,encoding='utf-8-sig')
    episodes=[]
    for name in ['main','south']:
        for component in ['advection','meridional']:
            for cutoff in [1,2,3]:
                flag=weather[name+'_'+component].to_numpy() < -cutoff
                starts=np.flatnonzero(flag&~np.r_[False,flag[:-1]])
                ends=np.flatnonzero(flag&~np.r_[flag[1:],False])
                for s,e in zip(starts,ends):
                    episodes.append(dict(region=name,component=component,threshold_K_day=-cutoff,
                        start=str(days[s]),end=str(days[e]),duration_days=int(e-s+1),
                        mean_K_day=float(weather[name+'_'+component].iloc[s:e+1].mean()),
                        minimum_K_day=float(weather[name+'_'+component].iloc[s:e+1].min())))
    pd.DataFrame(episodes).to_csv(OUT/'cold_advection_episodes.csv',index=False,encoding='utf-8-sig')
    lag_screen(days,p,asym,dipole_change,eddy_dipole_change,weather)
    np.savez_compressed(OUT/'daily_fields.npz',time=days,pressure=p,lat=lat,lon=lon,
        z=z.astype(np.float32),z_eddy=eddy.astype(np.float32),z_change=zchange.astype(np.float32),
        NA_eddy_Z_m=asym,ridge_trough_change_m=dipole_change,polar_Z_change_m=polar_change,
        ridge_trough_eddy_change_m=eddy_dipole_change,
        advection850=advd.astype(np.float32),meridional_advection850=merd.astype(np.float32),
        u850=ud.astype(np.float32),v850=vd.astype(np.float32),t850=td.astype(np.float32),
        fixed_advection_support=fixed,land_mask=land)
    QA.update(boxes=BOXES,windows=WINDOWS,baseline=['2024-12-28','2024-12-30'],
        interpretation='Exploratory temporal/spatial diagnosis; not an estimate of the upper forcing causal effect',
        no_NAM_or_climatological_ERA5_anomaly_claim=True,
        no_return_to_testing_existing_upper_reflection=True,
        surface_T2m_reference='NCEP R1 1991–2020 daily climatology',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUTS},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'method_checks.json').write_text(json.dumps(QA,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(period_rows)[['period','main_advection','main_meridional','south_meridional',
        'cooling_centroid_lon_E','main_T2m_anomaly_K','south_T2m_anomaly_K']].round(2).to_string(index=False),flush=True)
    print('COMPLETED',OUT,flush=True)


def pv_field(p,lat,lon,d,sp):
    sel=(p>=50)&(p<=700);pp=p[sel]*100
    u,v,t=[np.where(pp[None,:,None,None]<=sp[:,None],d[k][:,sel].astype(float),np.nan)
           for k in ['u','v','t']]
    phi=np.deg2rad(lat);cos=np.cos(phi)[None,None,:,None]
    safe=np.where(abs(cos)<1e-5,np.nan,cos)
    theta=t*(100000/pp[None,:,None,None])**KAPPA
    dl=np.deg2rad(2.5)
    dx=lambda x:(np.roll(x,-1,-1)-np.roll(x,1,-1))/(2*dl*A*safe)
    theta_p=np.gradient(theta,pp,axis=1,edge_order=2)
    theta_x=dx(theta);theta_y=np.gradient(theta,phi,axis=2,edge_order=2)/A
    vort=dx(v)-np.gradient(u*cos,phi,axis=2,edge_order=2)/(A*safe)
    up=np.gradient(u,pp,axis=1,edge_order=2);vp=np.gradient(v,pp,axis=1,edge_order=2)
    pv=-G*((vort+2*OMEGA*np.sin(phi)[None,None,:,None])*theta_p-vp*theta_x+up*theta_y)*1e6
    rotation=1e-6;testu=rotation*A*np.cos(phi)
    computed=-np.gradient(testu*np.cos(phi),phi,edge_order=2)/(A*np.where(abs(np.cos(phi))<1e-5,np.nan,np.cos(phi)))
    exact=2*rotation*np.sin(phi);use=(lat>=35)&(lat<=85)
    error=float(np.max(abs(computed[use]-exact[use]))/np.max(abs(exact[use])))
    assert error<.02
    QA['PV_method']=dict(formula='-g*((zeta+f)*theta_p-v_p*theta_x+u_p*theta_y)',
        units='PVU',pressure_levels_hPa=(pp/100).tolist(),temporal='six-hour PV first',
        analytic_solid_rotation_relative_error=error,terrain_mask_before_derivatives=True,
        limitations='2.5-degree grid and sparse pressure levels; exploratory proxy only')
    return pv,pp/100


def tp_crossings(pv,p,lat,lon,days):
    rows=[];coverage={}
    use=(p>=100)&(p<=700);p=p[use];v=pv[:,use]
    for threshold in [1.5,2.,3.]:
        brackets=(v[:,:-1]>=threshold)&(v[:,1:]<threshold)
        count=brackets.sum(1);idx=brackets.argmax(1)
        at=lambda x,delta=0:np.take_along_axis(x,(idx+delta)[:,None],axis=1)[:,0]
        v0,v1=at(v),at(v,1)
        frac=np.divide(threshold-v0,v1-v0,out=np.full(v0.shape,np.nan),where=abs(v1-v0)>1e-8)
        frac=np.where(count==1,frac,np.nan)
        pressure=np.exp(np.log(p[idx])*(1-frac)+np.log(p[idx+1])*frac)
        unique=np.where(count==1,pressure,np.nan)
        mean6=spatial_mean(unique,lat,lon,BOXES['NA_sector'])
        cover6=spatial_mean((count==1).astype(float),lat,lon,BOXES['NA_sector'])
        mean6=np.where(cover6>=.8,mean6,np.nan)
        meand=daily(mean6);coverd=daily(cover6)
        coverage[str(threshold)]=dict(minimum_sixhour_valid_area_fraction=float(cover6.min()),
            valid_daily_count=int(np.isfinite(meand).sum()))
        for i,day in enumerate(days):
            rows.append(dict(date=str(day),threshold_PVU=threshold,
                unique_crossing_pressure_hPa=float(meand[i]),coverage_fraction=float(coverd[i])))
    QA['tropopause_method']='Unique downward PV crossing in 100–700 hPa, log-pressure interpolation; require >=80% valid area at every six-hour sample. No extrapolation; not the thermal tropopause.'
    return pd.DataFrame(rows),coverage


def lag_screen(days,p,asym,dipole,eddy_dipole,weather):
    index=lambda pp:int(np.flatnonzero(p==pp)[0])
    pairs={
        'Zeddy3_to_Zeddy100':(asym[:,index(3)],asym[:,index(100)]),
        'Zeddy100_to_Zeddy300':(asym[:,index(100)],asym[:,index(300)]),
        'Zeddy300_to_Zeddy500':(asym[:,index(300)],asym[:,index(500)]),
        'D100_to_D500':(dipole[:,index(100)],dipole[:,index(500)]),
        'D300_to_D500':(dipole[:,index(300)],dipole[:,index(500)]),
        'EddyD100_to_EddyD500':(eddy_dipole[:,index(100)],eddy_dipole[:,index(500)]),
        'EddyD300_to_EddyD500':(eddy_dipole[:,index(300)],eddy_dipole[:,index(500)]),
        'D500_to_main_cooling':(dipole[:,index(500)],-weather.main_meridional.to_numpy()),
    }
    rows=[]
    for scope,limits in [('primary',('2024-12-28','2025-01-15')),
                         ('extended_context',('2024-12-28','2025-01-31'))]:
        mask=window(days,limits)
        for name,(x,y) in pairs.items():
            for difference in [False,True]:
                xx,yy=x[mask],y[mask]
                if difference: xx,yy=np.diff(xx),np.diff(yy)
                for lag in range(-4,5):
                    a,b=(xx[:-lag],yy[lag:]) if lag>0 else ((xx[-lag:],yy[:lag]) if lag<0 else (xx,yy))
                    good=np.isfinite(a)&np.isfinite(b);a,b=a[good],b[good]
                    r=float(np.corrcoef(a,b)[0,1]) if len(a)>=10 and min(np.std(a),np.std(b))>1e-8 else np.nan
                    rows.append(dict(scope=scope,pair=name,first_difference=difference,
                        lag_days=lag,n_pairs=len(a),r=r,
                        interpretation='positive lag means first series leads; descriptive, autocorrelated, no causal or significance claim'))
    pd.DataFrame(rows).to_csv(OUT/'descriptive_lag_screen.csv',index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    if '--primary' in sys.argv:
        END=np.datetime64('2025-01-15')
        OUT=OUT/'primary_to_Jan15'
    run()
