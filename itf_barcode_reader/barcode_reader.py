#!/usr/bin/env python

import numpy
import pylab
import scipy.ndimage

# Numberical values for narrow/wide lines
chars = {
    'nnWWn': 0,
    'WnnnW': 1,
    'nWnnW': 2,
    'WWnnn': 3,
    'nnWnW': 4,
    'WnWnn': 5,
    'nWWnn': 6,
    'nnnWW': 7,
    'WnnWn': 8,
    'nWnWn': 9,
}


def lookup_char(char):
    return chars.get(char, -1)

# Parse given array for narrow/wide lines
def parse_linescan(vs, lpn=101, length_threshold=5, use_mean=False):
    # filter to find threshold
    if use_mean:
        fvs = numpy.ones(lpn) * vs.mean()
    else:
        fvs = scipy.ndimage.convolve1d(
            vs, numpy.ones(lpn, dtype='f8') / lpn, mode='reflect')
    # binarize
    b = vs > fvs
    maxv = vs.max()
    minv = vs.min()
    # count how long highs and lows are
    start = None
    state = b[0]
    tokens = []
    for (i, bv) in enumerate(b):
        if (bv != state):
            if start is not None:  # not the first one
                if ((i - start) < length_threshold):
                    continue
                tokens.append([state, start, i - start, 'u'])
            start = i
            state = bv
    # find thick and thin spaces/bars
    tt = numpy.mean([t[2] for t in tokens if t[0]])
    tf = numpy.mean([t[2] for t in tokens if not t[0]])
    for i in xrange(len(tokens)):
        if tokens[i][0]:
            threshold = tt
        else:
            threshold = tf
        if tokens[i][2] > threshold:
            tokens[i][3] = 'W'
        else:
            tokens[i][3] = 'n'
        pylab.text(
            tokens[i][1] + tokens[i][2] / 2, fvs[tokens[i][1]], tokens[i][3])
    # return a list of narrow/wides as [0, 1]
    return tokens

# Take narrow/wide lines from parse_linescan and return barcode value
def parse_tokens(ls):
    vs = ''.join([t[3] for t in ls])
    if vs[:4] != 'nnnn':
        print vs0
        return -1  # no start code
    if vs[-3:] != 'Wnn':
        print vs
        return -2  # no end code
    vs = vs[4:-3]
    c, r = divmod(len(vs), 5)
    if r != 0:
        print vs, len(vs), c, r
        return -3  # invalid number of bars
    if divmod(c, 2)[1] != 0:
        print vs, len(vs), c, r
        return -4  # invalid number of [5-bar sequences] characters
    nc = c / 2
    v = []
    # parse list of narrow/wides into value
    for i in xrange(nc):
        s = i * 10
        e = s + 10
        # lookup chars
        v.append(lookup_char(''.join(vs[s:e:2])))
        v.append(lookup_char(''.join(vs[s+1:e:2])))
    return v


def read_barcode(bcd, lpn=101, length_threshold=5, use_mean=False):
    tokens = parse_linescan(bcd, lpn, length_threshold, use_mean)
    bcd_val = parse_tokens(tokens)
    return bcd_val
