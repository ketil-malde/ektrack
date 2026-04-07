# Test detections with various parameters.
# Count singleton detections vs number of detections
# Many detections, few singletons = good

from netcdf2dets import load, detections
from netcdf2dets import load, get_detections
from detections import cluster_det
import sys

if __name__ == '__main__':
    print('Reading netCDF file: ' + sys.argv[1])
    ds = load(sys.argv[1])
    # Run tests/benchmark across parameters (randomsearch?)

    # todo: vary angle_sigma and range_sigma
    for prom in [1.0, 1.5, 2.0, 2.5]:
        for f, dets in dsi.items():
            print(prom, f, len(dets))
        print()
        dsi = get_detections(ds, p=100, minrng=5, maxrng=50, minprom=prom)

        dcl = cluster_det(dsi)
        # count number of detections and number of singletons
        print('prom:', prom, 'clusters:', len(dcl), end='')
        for ln in range(1, 6): print(f'  size{ln}: {len([d for d in dcl if len(d) == ln])}', end='')
        print()
