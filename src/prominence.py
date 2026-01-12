import numpy as np

values = [1, 2, 3, 2, 4, 1]

def calculate_prominence_factors1(values):
    prominence_factors = [0] * len(values)
    for i in range(len(values)):
        min_descent = float('inf')
        for j in range(i - 1, -1, -1):
            if values[j] > values[i]:
                min_descent = min(min_descent, values[i] - values[j])
            if min_descent != float('inf'):
                break
        prominence_factors[i] = min_descent
    return prominence_factors


def calculate_prominence_factors2(values):
    prominence_factors = [0] * len(values)
    stack = []
    for i in range(len(values)):
        while stack and values[stack[-1]] < values[i]:
            idx = stack.pop()
            prominence_factors[idx] = values[i] - values[idx]
        if not stack or values[i] > values[stack[-1]]:
            stack.append(i)
    return prominence_factors


def myprom(values):
    history = []  # Stack of H, D

    cur_depth = values[0]
    cur_height = values[0]
    for v in values[1:]:
        if v < cur_depth:  # going down
            cur_depth = v
        if v > cur_height:  # going up
            cur_height = v
            if history[-1] < cur_height:
                history.pop()
            # when what to push?


def myprom2(values):
    prominence = [0] * len(values)
    for i, v in enumerate(values):
        maxdepth = v
        left_prom = 0
        for j in range(i, 0, -1):
            if values[j] > v:
                left_prom = v - maxdepth
                break
            else:
                maxdepth = min(values[j], maxdepth)
        else:
            left_prom = v

        maxdepth = v
        right_prom = 0
        for j in range(i, len(values)):
            if values[j] > v:
                right_prom = v - maxdepth
                break
            else:
                maxdepth = min(values[j], maxdepth)
        else:
            right_prom = v

        prominence[i] = min(right_prom, left_prom)
    return prominence

# This one works, I think?
# Terrible scaling
def myprom3(values):
    prominence = [0] * len(values)

    def find_prom(rng):
        # print(v)  # <- inherit from call site
        maxdepth = v  # why the fuck?
        prom = 0
        for j in rng:
            if values[j] > v:
                prom = v - maxdepth
                break
            else:
                maxdepth = min(values[j], maxdepth)
        else:
            prom = v
        return prom

    for i, v in enumerate(values):
        left_prom = find_prom(range(i, -1, -1))
        right_prom = find_prom(range(i, len(values)))
        prominence[i] = min(right_prom, left_prom)
    return prominence


def myprom4(values):
    # How to deal with the ends.  Peaks? Troughs?  Both/neither?

    def is_peak(i):
        after = True if i == len(values) - 1 else values[i + 1] <= values[i]  # note: asymmetry if equal
        before = True if i == 0 else values[i - 1] < values[i]
        # if after and before: print(i, 'is a peak')
        return after and before

    def is_trough(i):
        after = True if i == len(values) - 1 else values[i + 1] >= values[i]
        before = True if i == 0 else values[i - 1] >= values[i]
        # if after and before: print(i, 'is a trough')
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
