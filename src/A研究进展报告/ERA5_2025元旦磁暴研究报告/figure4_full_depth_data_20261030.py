"""Extend Figure 4 with available ERA5 levels; no display-validity gates."""
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_upper_vortex_followup_20261029 as follow


def extend_source(original, days):
    follow.END=np.datetime64('2025-01-11')
    _, dates,p,lat,lon,fields=follow.load()
    np.testing.assert_array_equal(dates,days.to_numpy(dtype='datetime64[D]'))
    iy=np.array([np.flatnonzero(lat==v)[0] for v in original['lat']])
    np.testing.assert_array_equal(lon,original['lon'])
    new=follow.daily(fields['z'].astype(float)/follow.G)[:,:,iy,:]
    # Preserve exact existing values at every already-plotted level.
    for j,pressure in enumerate(p):
        old=np.flatnonzero(original['pressure']==pressure)
        if len(old):new[:,j]=original['z'][:,old[0]]
    return dict(time=original['time'],pressure=p.astype(float),lat=original['lat'],lon=lon,z=new)


def all_phase(waves, days, shared, out):
    waves['longitude']=shared.wrap(180-np.angle(waves['coefficients'],deg=True))
    waves['valid']=np.isfinite(waves['coefficients'])
    records=[]
    for j in range(len(waves['p'])):
        for i,day in enumerate(days):
            coeff=waves['coefficients'][i,j]
            records.append(dict(date=str(day.date()),source=waves['sources'][j],pressure_hPa=float(waves['p'][j]),
                reference_height_km=float(waves['height'][j]),latitude_band='60-80N',
                C_real=float(coeff.real),C_imag=float(coeff.imag),amplitude_m=float(abs(coeff)),
                phase_valid=bool(np.isfinite(coeff)),W1_trough_longitude_degE=float(waves['longitude'][i,j])))
    pd.DataFrame(records).to_csv(out/'W1_trough_multilevel_daily.csv',index=False,encoding='utf-8-sig')
    shared.QA['multilevel_wave'].update(display_filter='none; every finite coefficient is shown',
        amplitude_screen_m=None,MLS_display_sector_screen=False,MLS_display_daynight_screen=False,
        original_MLS_observation_quality_control_retained=True,
        ERA5_pressure_hPa=waves['era_p'].tolist(),weak_amplitude_phase_is_not_removed=True)


def envelopes(source,waves,days,pre,event,shared,out):
    lat,lon,p=source['lat'],source['lon'],source['pressure']
    weights=shared.base.cell_weights(lat,len(lon));sector=(lon>=-130)&(lon<=-60)
    values=np.full((len(days),len(p)),np.nan)
    records=[];summary=[];all_edges=[]
    for j,pressure in enumerate(p):
        reference=50 if pressure>=300 else 60
        threshold=float(source['z'][pre,j,np.flatnonzero(lat==reference)[0]].mean())
        edges=np.full((len(days),len(lon)),np.nan)
        for i,field in enumerate(source['z'][:,j]):
            smooth=shared.base.smooth_map(field,.4)
            core,fraction=shared.region.main_component(smooth<=threshold,weights)
            touches=0
            for x in range(len(lon)):
                inside=np.flatnonzero(core[:,x])
                if not len(inside):continue
                k=inside[0]
                if k==0:
                    touches+=1
                    continue # True outer crossing is outside the data domain, not at 20 N.
                lo,hi=smooth[k-1,x],smooth[k,x]
                if not (np.isfinite(lo) and np.isfinite(hi)) or lo==hi:continue
                if not (lo>=threshold and hi<=threshold):raise ValueError('Invalid boundary bracket')
                edges[i,x]=lat[k-1]+(threshold-lo)/(hi-lo)*(lat[k]-lat[k-1])
            finite=np.isfinite(edges[i,sector]);n=int(finite.sum())
            if n:values[i,j]=float(edges[i,sector][finite].mean())
            records.append(dict(date=str(days[i].date()),pressure_hPa=float(pressure),
                reference_height_km=float(waves['era_height'][j]),reference_latitude_N=reference,
                fixed_Z_threshold_m=threshold,NA_mean_edge_latitude_degN=values[i,j],
                valid_sector_meridians=n,total_sector_meridians=int(sector.sum()),
                southern_limit_meridians=touches,retained_component_area_fraction=float(fraction)))
        all_edges.append(edges)
        a=values[pre,j];b=values[event,j]
        av=float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        bv=float(np.nanmean(b)) if np.isfinite(b).any() else np.nan
        summary.append(dict(pressure_hPa=float(pressure),reference_height_km=float(waves['era_height'][j]),
            pre_mean_latitude_degN=av,Jan07_09_latitude_degN=bv,southward_shift_deg=av-bv))
    pd.DataFrame(records).to_csv(out/'Z_envelope_multilevel_daily.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(out/'Z_envelope_multilevel_periods.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'Z_envelope_daily_by_longitude.npz',time=days.to_numpy(dtype='datetime64[D]'),
        pressure_hPa=p,latitude_edge_degN=np.stack(all_edges,axis=1),longitude=lon)
    shared.QA['full_depth_envelope']=dict(pressure_hPa=p.tolist(),minimum_native_pressure_level_hPa=float(p.max()),
        display_filter=False,all_days_valid_gate=False,meridian_coverage_gate=False,
        statistic='mean of available actual crossings, no invented crossing where outside domain',
        smoothing='original 0.4 grid-cell Gaussian retained as boundary definition',
        threshold='original Dec28-30 zonal mean at 60N for p<300, at 50N for p>=300',
        below_surface_pressure_level_values='finite archived ERA5 values retained as requested; may be extrapolated, not physical vortex boundary')
    return values,pd.DataFrame(summary)
