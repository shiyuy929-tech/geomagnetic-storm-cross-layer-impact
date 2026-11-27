"""Daily nighttime MLS O3 means, without conditioning on T/GPH retrievals."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def calculate():
    records = []
    for day in pd.date_range('2024-12-28', '2025-01-11'):
        path = ROOT/'event_evidence_20261002/data/O3'/f'MLS-Aura_L2GP-O3_v06-01-c01_{day:%Yd%j}.he5'
        with h5py.File(path) as f:
            s=f['HDFEOS/SWATHS/O3']; d=s['Data Fields']; g=s['Geolocation Fields']
            pressure=g['Pressure'][:]; latitude=g['Latitude'][:]
            quality=((d['Status'][:] % 2 == 0) & (d['Quality'][:] > 1)
                     & (d['Convergence'][:] < 1.03) & (g['SolarZenithAngle'][:] > 95))
            for j,p in enumerate(pressure):
                if not .001 <= p <= 1.: continue
                values=d['L2gpValue'][:,j].astype(float)*1e6
                good=quality & (d['L2gpPrecision'][:,j] > 0) & np.isfinite(values) & (np.abs(values)<100)
                for lo,hi in [(40,60),(60,80)]:
                    m=good & (latitude>=lo) & (latitude<hi)
                    records.append(dict(date=day.strftime('%Y-%m-%d'),band=f'{lo}-{hi}',pressure_hPa=float(p),
                                        ozone_ppmv=float(np.mean(values[m])) if m.any() else np.nan,n=int(m.sum())))
    a=pd.DataFrame(records)
    a['pressure_key']=a.pressure_hPa.round(9)
    base=a[a.date.between('2024-12-28','2024-12-30')].groupby(['band','pressure_key']).agg(
        baseline=('ozone_ppmv','mean'),n_days=('ozone_ppmv','count')).reset_index()
    a=a.merge(base,on=['band','pressure_key'],validate='many_to_one')
    a['dO3_daily_mean']=np.where((a.n_days==3)&(a.baseline>.05),100*(a.ozone_ppmv/a.baseline-1),np.nan)
    return a
