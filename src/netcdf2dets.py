from netCDF4 import Dataset
import xarray as xr
import numpy as np
from sys import argv
from typing import Dict, List

from prominence import prominence
from detections import Detection


def readnetcdf(ncfile: str) -> Dict[str, xr.Dataset]:
    """
    Read pulse compressed ping data from a NetCDF file, and generate a set of detections.
    Returns an xarray with (pulse-compressed for bb) backscatter and angles.
    """
    nc_data = Dataset(ncfile)
    freqs = list(nc_data.variables['frequency'][:])
    res = {}
    for g in nc_data.groups.keys():
        if g == 'Environment': continue
        # Each g corresponds to one frequency
        freq = freqs.pop(0)
        d = xr.open_dataset(ncfile, engine='netcdf4', group=g)
        wbtlabel = d.attrs['channel_id']
        if 'pulse_compressed_re' in d:
            # FM data - calc mean magnitude across sectors (quadrants)
            backscatter = abs(d['pulse_compressed_re'] + d['pulse_compressed_im'] * 1j).mean(dim='sector')
            atype = 'FM'
        elif 'sv' in d:
            # CW data (no pulse comression, keep sv (should be TS, no?))
            backscatter = d['sv']
            atype = 'CV'
        else:
            raise RuntimeError(f'Neither FM nor CW data in {ncfile}/{wbtlabel}?')

        theta = d['angle_alongship']
        phi = d['angle_athwartship']

        res[g] = xr.Dataset({'backscatter': backscatter, 'theta': theta, 'phi': phi}).assign_attrs(wbtlabel=wbtlabel, pulsetype=atype, frequency=freq)

    return res

def calc_prom_arrays(channels: Dict[str, xr.Dataset]) -> None:
    """Calculate the prominence array from backscatter"""
    for g in channels.keys():
        print('Processing:', g)
        channels[g]['prominence'] = xr.DataArray(
            np.apply_along_axis(prominence, axis=1, arr=np.log(channels[g]['backscatter'])),
            dims=["ping_time", "range"],
            coords=[channels[g]['ping_time'], channels[g]['range']])

def detections(pch: Dict[str, xr.Dataset], p: int, minprom: float = 0.0, maxrng: float = 999999.0, minrng: float = 0.0) -> Dict[str, List[Detection]]:
    """Calculate detections using prominence threshold"""
    res = {}
    for g, mych in pch.items():
        res[g] = []
        prom = mych['prominence'][p]
        rng = mych['range']
        time = mych['ping_time'][p]
        theta = mych['theta'][p]
        phi = mych['phi'][p]
        idx = (prom > minprom) & (rng > minrng) & (rng < maxrng)
        for pr, r, th, ph in zip(prom[idx], rng[idx], theta[idx], phi[idx]):
            res[g].append(Detection(p, time.item(), int(pch[g].frequency), r.item(), th.item(), ph.item(), pr.item()))
    return res

def load(infile: str) -> Dict[str, xr.Dataset]:
    '''Read the NetCDF file and calculate prominence'''
    ch = readnetcdf(infile)
    calc_prom_arrays(ch)
    return ch


# ../data/D20230803-T230004.nc <- salmon plus seabed?
# Generate the NetCDF by running raw2pc.py from CRIMAC-FM-testdatapaper on D20230803-T230004.raw
# ..except that generates four outputs: pc_{1..4}, none of them matching this file.
if __name__ == '__main__':
    # read args[1]
    ch = readnetcdf(argv[1])
    calc_prom_arrays(ch)

    # compute the detections
    prange = 10  # len(ch[next(iter(ch))]['ping_time'])
    dsd = [detections(ch, i, minprom=2.0, minrng=6.0, maxrng=8.0) for i in range(prange)]

    # Print results
    # for dss in dsd:
    dss = dsd[7]
    for g, s in dss.items():
        print(f'{g}: "{ch[g].wbtlabel}" {int(ch[g].frequency)}kHz, type={ch[g].pulsetype} len={len(dss[g])}')
        for x in s: print(x)
