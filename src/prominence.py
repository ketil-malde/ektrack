import numpy as np

testvalues = [1, 2, 3, 2, 4, 1]

def prominence(values):
    vpeaks = np.logical_and(values >= np.append(values[1:], [-9999]), values > np.append([-9999], values[:-1]))
    vtroughs = np.logical_and(values <= np.append(values[1:], [9999]), values <= np.append([9999], values[:-1]))

    # These two are very expensive - replace with numpy method?
    def is_peak(i):
        after = True if i == len(values) - 1 else values[i + 1] <= values[i]  # note: asymmetry if equal
        before = True if i == 0 else values[i - 1] < values[i]
        # if after and before: print(i, 'is a peak')
        assert (after and before) == vpeaks[i], f'i={i}, after={after}, before={before}, vpeaks[i]={vpeaks[i]}, {values[i-1:i+1]}'
        return after and before

    def is_trough(i):
        after = True if i == len(values) - 1 else values[i + 1] >= values[i]
        before = True if i == 0 else values[i - 1] >= values[i]
        # if after and before: print(i, 'is a trough')
        assert (after and before) == vtroughs[i], f'i={i}, after={after}, before={before}, vtroughs[i]={vtroughs[i]}, {values[i-1:i+1]}'
        return after and before

    # find the max depth going back to at most j
    def find_max_depth(j, troughs, direction):
        # print('fmd:', j, direction)
        c = len(troughs) - 1
        maxdepth = float(troughs[c])
        if direction == 'left':
            while c >= 0 and troughs[c] > j:
                if float(values[troughs[c]]) < maxdepth:
                    maxdepth = float(values[troughs[c]])
                c -= 1
        elif direction == 'right':
            while c >= 0 and troughs[c] < j:
                if float(values[troughs[c]]) < maxdepth:
                    maxdepth = float(values[troughs[c]])
                c -= 1
        else:
            print('direction must be left or right')
            exit(-1)

        # print('maxdepth before', j, 'was', maxdepth)
        return float(maxdepth)  # necessary to not produce list of 1-d vectors?

    def find_proms(direction):
        myrange = range(len(values)) if direction == 'left' else reversed(range(len(values)))
        peaks = []
        troughs = []
        proms = np.zeros(len(values))

        for i in myrange:
            # print('iteration:', i, values[i])
            # print('  peaks:', peaks)
            # print('  troughs:', troughs)

            if is_peak(i):
                while peaks:
                    j = peaks.pop()
                    if values[j] > values[i]:
                        # set prominence to max depth
                        proms[i] = values[i] - find_max_depth(j, troughs, direction)
                        # if proms[i] < 0: print(direction, i, j, peaks, troughs)
                        peaks.append(j)
                        peaks.append(i)
                        break
                if not peaks:  # no higher peak between us and the end of the vector
                    peaks.append(i)
                    if direction == 'left':
                        maxdepth = values[troughs[0]] if troughs else 0
                        proms[i] = (values[i] - maxdepth) if i != 0 else 0
                        # if proms[i] < 0: print('*left', i, peaks, troughs)
                    elif direction == 'right':
                        maxdepth = values[troughs[0]] if troughs else 0
                        proms[i] = max(values[i] - maxdepth, 0)
                        # if proms[i] < 0: print('*right', i, peaks, troughs)
                    else:
                        print('Error: direction can\'t be', direction)
                        exit(-1)
            if is_trough(i):
                while troughs:
                    j = troughs.pop()
                    if values[j] < values[i]:
                        troughs.append(j)
                        troughs.append(i)
                        break
                if not troughs:
                    troughs.append(i)
        return proms

    p_fwd = find_proms('left')
    p_rev = find_proms('right')
    # print('forward: (', type(p_fwd), ')', p_fwd)
    # print('reverse: (', type(p_rev), ')', p_rev)
    # return [min(a, b) for (a, b) in zip(p_fwd, p_rev)]
    return np.minimum(p_fwd, p_rev)
