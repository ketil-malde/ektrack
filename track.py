from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from math import exp, sin, sqrt
import numpy as np

# 5 cm discrepancy reduces score with 1/e
range_sigma = 0.05
# angle of 1 degree give same penalty as above
angle_sigma = 1
rank_scale = 10  # ranks count 0.5, 0.45, 0.42, 0.38...

PI = 3.1415926


@dataclass
class Detection:
    pingno: int
    freq: str
    range: float  # meters
    theta: float  # degrees
    phi: float
    rank: int     # zero is strongest detection, higher is worse

    def __str__(self):
        x0, y0, z0 = self.location()
        return f'D {self.pingno} {self.freq //1000}kHz  Rng {self.range:5.2f} Th {self.theta:5.2f} Phi {self.phi:5.2f}\t[{x0:5.2f}, {y0:5.2f}, {z0:6.2f}]'

    def location(self):
        '''Convert to 3D coordinates'''
        x = self.range * sin(PI * self.theta / 180)
        y = self.range * sin(PI * self.phi / 180)
        z = sqrt(self.range * self.range - x * x - y * y)
        return x, y, z

    def from_location(self, x, y, z):
        '''Create from 3D coordinates'''
        pass


def delta_pings(det1, det2, uncertainty=1):
    '''Distance between detections in pings with given time difference'''
    d_theta = det1.theta - det2.theta
    d_phi = det1.phi - det2.phi
    d_range = det1.range - det2.range
    d_rank = rank_scale/(rank_scale*2 + det1.rank) + rank_scale/(rank_scale*2 + det2.rank)
    # one if perfect match (all zero), towards zero if deviation is too large
    # uncertainty less than one means the distribution is wider but lower
    return sqrt(uncertainty) * d_rank * exp(- (d_range**2/range_sigma**2 + d_theta**2/angle_sigma**2 + d_phi**2/angle_sigma**2) * uncertainty)


def mkdet(fields):
    '''Parse a line of CSV input into a detection.'''
    return Detection(pingno=int(fields[0]), freq=int(fields[3]), range=float(fields[4]),
                     theta=float(fields[5]), phi=float(fields[6]), rank=0)


# ########### Tracking across frequencies ######################

def link_det(dets1, dets2):
    '''Find optimal pairing'''
    # calc matrix and return list of multi-dets
    pass


def genmdet(dets):
    '''Group across frequencies from one ping.'''
    # iterate link_det, by number of detections
    pass


# ############### Tracking across time ########################

@dataclass
class Track:
    # also track adjustments?
    detections: [Detection]

    def predict(self, curtime):
        '''predict next detection from a track'''
        if len(self.detections) == 1:
            '''Predict avg of other tracks? with high uncertainty'''
            pass
        else:  # at least two observations
            '''Linear extrapolation, less uncertainty'''
            pass

    def summarize(self):
        '''Generate a finished track output'''
        pass


@dataclass
class Tracks:
    track: [Track]
    mru: []  # roll, pitch, heave?
    offsets: {}  # frequency -> location


def predict1(track, frequency):
    '''Predict next detection based on track history'''
    detection = None
    uncertainty = None
    return detection, uncertainty


def delta_chans(tr1, tr2):
    '''Distance between tracks in different frequencies'''
    pass


def track1(tracks, detections):
    # link tracks to detections (high confidence)
    ntracks, ndets = len(tracks), len(detections)
    mx = np.empty((ntracks,ndets))
    for t in range(ntracks):
        t_pred = predict(tracks[t])
        for d in range(ndets):
            mx[t, d] = delta_pings(t_pred, d)
    tind, dind = linear_sum_assignment(mx, maximize=True)
    
    # adjust for average movement

    # predict locations (include older tracks) and recalculate

    # link across frequencies


def tracks(detections):
    '''Process a sequence of lists of detections, per ping'''
    # iterator?  yielding tracks as they are closed
    pass



import csv
from itertools import groupby

if __name__ == '__main__':
    with open('DetectedSingleTargets(in).csv', 'r') as f:
        r = csv.reader(f, delimiter='\t')
        next(r, None)  # skip header
        for k, ds in groupby(map(mkdet, r), key=lambda x: x.pingno):
            print('Ping:', k)
            for k2, ds2 in groupby(ds, key=lambda x: x.freq):
                print('Ping:', k, 'Freq:', k2)
                for det in ds2:
                    if det.range < 12 and det.range > 11: print(det)
                print()
            print()
