import KoronaScript as ks
import KoronaScript.Modules as ksm
from sys import argv

def raw2tracks(indir, outdir):
        # Copied from CRIMAC-FM-testdatapaper
        ksi = ks.KoronaScript(TransducerRanges="korona/TransducerRanges.xml")
        ksi.add(ksm.EmptyPingRemoval())
        ksi.add(ksm.Comment(LineBreak='false', Label='comment'))
        # Remove channels not to be processed
        # ksi.add(ksm.ChannelRemoval(Channels=channels[channel]['channels'], KeepSpecified='true'))
        # tracking parameters are read form JSON file
        # for each transducer frequency?
        for tr_freq in ["38", "70", "120", "200", "333"]:
            ksi.add(ksm.Tracking(Active='true',
                                 TrackerType='Peak',
                                 kHz=tr_freq,
                                 PlatformMotionType='Floating',
                                 MinTS='-50',
                                 PulseLengthDeterminationLevel='50',
                                 MinEchoLength='0',
                                 MaxEchoLength='1',
                                 MaxGainCompensation='18',
                                 DoPhaseDeviationCheck='false',
                                 MaxPhaseDevSteps='10',
                                 MaxTS='0',
                                 MaxDepth='22',
                                 # Must be determined per dataset
                                 MaxAlongshipAngle='10',
                                 MaxAthwartshipAngle='10',
                                 InitiationGateFunction={"Alpha": 2.8, "Beta": 2.8, "Range": 0.1, "TS": 20},
                                 InitiationMinLength='1',
                                 GateFunction={"Alpha": 2.8, "Beta": 2.8, "Range": 0.1, "TS": 20},
                                 AlphaBetaEstimator={"Alpha": 0.9, "Beta": 0.1},
                                 MaxMissingPings='4',
                                 MaxMissingSamples='24',
                                 MaxMissingPingsFraction='0.7',
                                 MinTrackLength='8',
                                 MinSampleToLengthFraction='0.5'))

        ksi.run(src=indir, dst=outdir)


if __name__ == '__main__':
    raw2tracks(argv[1], argv[2])
