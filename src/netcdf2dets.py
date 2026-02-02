from netCDF4 import Dataset
import xarray as xr
import numpy as np
from sys import argv

from prominence import prominence
from track import Detection


def readnetcdf(ncfile):
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
            raise RunTimeError(f'Neither FM nor CW data in {ncfile}/{wbtlabel}?')  # noqa - flake doesn't know about RTE

        theta = d['angle_alongship']
        phi = d['angle_athwartship']

        res[g] = xr.Dataset({'backscatter': backscatter, 'theta': theta, 'phi': phi}).assign_attrs(wbtlabel=wbtlabel, pulsetype=atype, frequency=freq)

    return res

def calc_prom_arrays(channels):
    """Calculate the prominence array from backscatter"""
    for g in channels.keys():
        print('Processing:', g)
        channels[g]['prominence'] = xr.DataArray(
            np.apply_along_axis(prominence, axis=1, arr=np.log(channels[g]['backscatter'])),
            dims=["ping_time", "range"],
            coords=[channels[g]['ping_time'], channels[g]['range']])

def dets2(pch, p, minprom=0, maxrng=0, minrng=99999):
    res = {}
    for g, mych in pch.items():
        prom = mych['prominence'][p]
        rng = mych['range']
        time = mych['ping_time'][p]
        theta = mych['theta'][p]
        phi = mych['phi'][p]

        idx = (prom > minprom) & (rng > minrng) & (rng < maxrng)
        res[g] = (time, prom[idx], rng[idx], theta[idx], phi[idx])
    return res

def ds2dets(ds):
    gres = {}
    for g, dets in ds.items():
        res = []
        t, ps, rngs, ths, phs = dets
        for p, r, th, ph in zip(ps, rngs, ths, phs):
            res.append(Detection(-1, t.item(), 0, r.item(), th.item(), ph.item(), 0))
        gres[g] = res
    return gres


# ../data/D20230803-T230004.nc <- salmon plus seabed?
# Generate the NetCDF by running raw2pc.py from CRIMAC-FM-testdatapaper on D20230803-T230004.raw
# ..except that generates four outputs: pc_{1..4}, none of them matching this file.
if __name__ == '__main__':
    # read args[1]
    ch = readnetcdf(argv[1])
    print(ch)
    # calculate prominence
    calc_prom_arrays(ch)
    print(ch)
    # compute the detections
    ds = dets2(ch, 7, minprom=2.0, maxrng=8.0, minrng=6.0)
    dsd = ds2dets(ds)
    for g, s in dsd.items():
        print(f'{g}: "{ch[g].wbtlabel}" {int(ch[g].frequency)}kHz, type={ch[g].pulsetype} len={len(ds[g])}')
        for x in s: print(x)
