from netCDF4 import Dataset
import xarray as xr
import numpy as np
from sys import argv

from prominence import prominence

def readnetcdf(ncfile):
    """
    Read pulse compressed ping data from a NetCDF file, and generate a set of detections.
    Returns an xarray with (pulse-compressed for bb) backscatter and angles.
    """
    nc_data = Dataset(ncfile)
    res = {}
    for g in nc_data.groups.keys():
        if g == 'Environment': continue
        # Each g corresponds to one frequency
        d = xr.open_dataset(ncfile, engine='netcdf4', group=g)
        wbtlabel = d.attrs['channel_id']
        if 'pulse_compressed_re' in d:
            # FM data - calc mean magnitude across sectors (quadrants)
            backscatter = abs(d['pulse_compressed_re'] + d['pulse_compressed_im'] * 1j).mean(dim='sector')
        elif 'sv' in d:
            # CW data (no pulse comression, keep sv (should be TS, no?))
            backscatter = d['sv']
        else:
            raise RunTimeError(f'Neither FM nor CW data in {ncfile}/{wbtlabel}?')  # noqa - flake doesn't know about RTE

        theta = d['angle_alongship']
        phi = d['angle_athwartship']

        res[g] = xr.Dataset({'backscatter': backscatter, 'theta': theta, 'phi': phi}).assign_attrs(wbtlabel=wbtlabel)

    return res

def calc_prom_arrays(channels):
    """Calculate the prominence array from backscatter"""
    for g in channels.keys():
        print('Processing:', g)
        # This is slow - use JAX or PyTorch?
        channels[g]['prominence'] = xr.DataArray(
            np.apply_along_axis(prominence, axis=1, arr=np.log(channels[g]['backscatter'])),
            dims=["ping_time", "range"],
            coords=[channels[g]['ping_time'], channels[g]['range']])

def detections(pchannels):
    """Calculate detections from prominence and channel data"""
    nonzero_indices_per_col = {}
    for g in pchannels.keys():
        gchan = pchannels[g]['prominence']
        nonzero_indices_per_col[g] = [gchan[v, :].values.nonzero()[0] for v in range(gchan.shape[0])]
    return nonzero_indices_per_col


if __name__ == '__main__':
    # read args[1]
    ch = readnetcdf(argv[1])
    print(ch)
    # calculate prominence
    calc_prom_arrays(ch)
    print(ch)
    # compute the detections
    for g, ds in detections(ch).items():
        print(g)
        print(ds[75])
