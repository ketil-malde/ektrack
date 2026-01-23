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
        # This is slow - use JAX or PyTorch?
        channels[g]['prominence'] = xr.DataArray(
            np.apply_along_axis(prominence, axis=1, arr=np.log(channels[g]['backscatter'])),
            dims=["ping_time", "range"],
            coords=[channels[g]['ping_time'], channels[g]['range']])

def detections(pchannels):
    """Calculate detections from prominence and channel data"""
    ret = {}
    for g in pchannels.keys():
        dss = []
        gchan = pchannels[g]
        nz_idx_per_col = [gchan['prominence'][v, :].values.nonzero()[0] for v in range(gchan['prominence'].shape[0])]
        for ping in range(0, 10):  # range(len(nz_idx_per_col)):
            ds = []
            for i in nz_idx_per_col[ping]:
                val = gchan['prominence'][ping][i].item()
                if val > 2.0:
                    rng = gchan['range'][i].item()
                    if 6.0 < rng < 8.0:
                        ptm = gchan['ping_time'][ping].item()
                        theta = gchan['theta'][ping][i].item()
                        phi = gchan['phi'][ping][i].item()
                        ds.append(Detection(pingno=ping, time=ptm, freq=0, range=rng, theta=theta, phi=phi, rank=0))
                        # ds.append(f"time={ptm}\tR={rng:.2f}\ttheta={theta:.2f}\tphi={phi:.2f}\tval={val:.2f}")
            dss.append(ds)
        ret[g] = dss
    return ret


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
    for g, ds in detections(ch).items():
        print(f'{g}: "{ch[g].wbtlabel}" {int(ch[g].frequency)}kHz, type={ch[g].pulsetype} len={len(ds[7])}')
        for s in ds[7]: print(s)
