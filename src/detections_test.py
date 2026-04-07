# Test detections with various parameters.
# Count singleton detections vs number of detections
# Many detections, few singletons = good

from netcdf2dets import load, get_detections
import detections
from detections import cluster_det
import sys
import random

if __name__ == '__main__':
    print('Reading netCDF file: ' + sys.argv[1])
    ds = load(sys.argv[1])
    # Run tests/benchmark across parameters (randomsearch?)

    # todo: vary angle_sigma and range_sigma
    for _ in range(50):
        # parameters - sample randomly
        prom = random.uniform(3, 10) / 3
        detections.range_sigma = 0.15 # random.uniform(1, 10) / 50
        detections.angle_sigma = 5  # random.uniform(3, 18) / 3

        dsi = get_detections(ds, p=100, minrng=5, maxrng=50, minprom=prom)
        # for f, dets in dsi.items():
        #    print(prom, f, len(dets))
        # print()

        dcl = cluster_det(dsi)

        print(f'prom: {prom:.2f} sdrng: {detections.range_sigma:.2f} sdang: {detections.angle_sigma:.2f} clusters: {len(dcl)}', end='')
        for ln in range(1, 6): print(f'  size={ln}: {len([d for d in dcl if len(d) == ln])}', end='')
        print()
