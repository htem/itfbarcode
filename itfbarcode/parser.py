#!/usr/bin/env python

import numpy
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

rchars = {chars[k]: k for k in chars}

errors = {
    -1: 'missing start code',
    -2: 'missing end code',
    -3: 'invalid number of bars',
    -4: 'invalid number of characters',
    -5: 'invalid input',
}


default_bar_info = {
    'bar': {
        'n': 1,
        'w': 1.8,
    },
    'space': {
        'n': 0.1,
        'w': 1.25,
    }
}


def lookup_char(char):
    return chars.get(char, -1)


def parse_line(vs, ndigits=6, bar_info=None):
    if bar_info is None:
        bar_info = default_bar_info
    nvs = len(vs)
    pass


def parse_linescan(
        vs, lpn=101, length_threshold=5, use_mean=False, full=False):
    """Parse given array for narrow/wide lines

    lpn = width of smoothing filter
    length_threshold = minimum bar/space width
    """
    # filter to find threshold
    if use_mean:
        fvs = numpy.ones_like(vs) * vs.mean()
    else:
        fvs = scipy.ndimage.convolve1d(
            vs, numpy.ones(lpn, dtype='f8') / lpn, mode='reflect')
    # binarize
    b = vs > fvs
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
    if full:
        return (
            tokens,
            {
                'tokens': tokens,
                'bar_threshold': tt,
                'space_threshold': tf,
            })
    # return a list of narrow/wides as [0, 1]
    return tokens


def parse_tokens(ls):
    """Take narrow/wide lines from parse_linescan and return barcode value"""
    if isinstance(ls, (str, unicode)):
        vs = ls
    else:
        vs = ''.join([t[3] for t in ls])
    if vs[:4] != 'nnnn':
        vs = vs[::-1]  # first try reversing
    if vs[:4] != 'nnnn':
        return -1  # no start code
    if vs[-3:] != 'Wnn':
        return -2  # no end code
    vs = vs[4:-3]
    c, r = divmod(len(vs), 5)
    if r != 0:
        return -3  # invalid number of bars
    if divmod(c, 2)[1] != 0:
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


def read_barcode(bcd, lpn=101, length_threshold=5, use_mean=False, full=False):
    if len(bcd) == 0:
        return -5  # invalid input
    r = parse_linescan(bcd, lpn, length_threshold, use_mean, full=full)
    if full:
        tokens, info = r
    else:
        tokens = r
    bcd_val = parse_tokens(tokens)
    if full:
        return bcd_val, info
    return bcd_val


def is_valid(bc):
    return isinstance(bc, list) and not any((b < 0 for b in bc))


def gen_tokens(v, ndigits=None):
    if isinstance(v, (str, unicode)):
        ndigits = len(v)
        v = int(v)
    if ndigits is None:
        raise ValueError("ndigits must be defined")
    if (ndigits % 2):
        raise ValueError("ndigits must be even: %s" % ndigits)
    sc = 'nnnn'
    ec = 'Wnn'
    s = sc
    vc = str(v).zfill(ndigits)
    for i in xrange(ndigits / 2):
        c0 = rchars[int(vc[2*i])]  # bars
        c1 = rchars[int(vc[2*i+1])]  # spaces
        for (j0, j1) in zip(c0, c1):
            s += j0
            s += j1
    s += ec
    return s


def test():
    ts = 'nnnnnnnnWWWWnnnnnnWWWWnnnnWWnnnWWnWnn'
    v = '000029'
    assert ''.join([str(i) for i in parse_tokens(ts)]) == v
    assert gen_tokens(v) == ts
