#!/usr/bin/env python

import numpy
import scipy.ndimage

from .objects import Barcode


wide_chars = 'SB'
narrow_chars = 'sb'

start_code = 'bsbs'
end_code = 'Bsb'


# Numberical values for narrow/wide lines
chars = {
    'bbBBb': '0',
    'ssSSs': '0',
    'nnWWn': '0',
    'BbbbB': '1',
    'SsssS': '1',
    'WnnnW': '1',
    'bBbbB': '2',
    'sSssS': '2',
    'nWnnW': '2',
    'BBbbb': '3',
    'SSsss': '3',
    'WWnnn': '3',
    'bbBbB': '4',
    'ssSsS': '4',
    'nnWnW': '4',
    'BbBbb': '5',
    'SsSss': '5',
    'WnWnn': '5',
    'bBBbb': '6',
    'sSSss': '6',
    'nWWnn': '6',
    'bbbBB': '7',
    'sssSS': '7',
    'nnnWW': '7',
    'BbbBb': '8',
    'SssSs': '8',
    'WnnWn': '8',
    'bBbBb': '9',
    'sSsSs': '9',
    'nWnWn': '9',
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
    if 'nnnn' not in vs:
        return -1  # no start code
    i = vs.index('nnnn')
    if i > len(vs) / 2:
        vs = vs[::-1]
        i = vs.index('nnnn')
    vs = vs[i:]
    #if vs[:4] != 'nnnn':
    #    vs = vs[::-1]  # first try reversing
    #if vs[:4] != 'nnnn':
    #    return -1  # no start code
    #if vs[-3:] != 'Wnn':
    #    return -2  # no end code
    if 'Wnn' not in vs:
        return -2  # no end code
    i = vs[::-1].index('nnW')  # find last end code
    vs = vs[:len(vs)-i]
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


def find_token_threshold(tokens, state):
    vs = [t.width for t in tokens if t.state == state]
    return numpy.mean(vs)


def tokens_to_string(
        tokens, bar_threshold, space_threshold, max_bar, max_space):
    ts = [space_threshold, bar_threshold]
    ms = [max_space, max_bar]
    s = ''
    for t in tokens:
        if t.width > ts[t.state]:
            if t.width > ms[t.state]:
                s += '?'
            elif t.width > ts[t.state]:
                s += wide_chars[t.state]
        else:
            s += narrow_chars[t.state]
    return s


def find_all_substring(st, substring):
    i = st.find(substring)
    inds = []
    while i != -1:
        inds.append(i)
        ni = st[i+1:].find(substring)
        if ni == -1:
            break
        i += ni + 1
    return inds


def string_to_value(bcs):
    if not all([s.lower() == 'b' for s in bcs[::2]]):
        return -1
    if not all([s.lower() == 's' for s in bcs[1::2]]):
        return -1
    if len(bcs) % 10 != 0:
        return -1
    nc = len(bcs) // 10
    chars = ''
    for i in xrange(nc):
        s = i * 10
        e = s + 10
        bc = lookup_char(bcs[s:e:2])
        sc = lookup_char(bcs[s+1:e:2])
        if bc < 0 or sc < 0:
            return -1
        chars += bc + sc
    if len(chars) == 0:
        return -1
    return int(chars)


def find_all_barcode_bounds(st, ndigits=None):
    if ndigits is None:
        check_digits = lambda nt: True
    else:
        check_digits = lambda nt: ((nt / 10) == (ndigits / 2))
    starts = find_all_substring(st, start_code)
    ends = find_all_substring(st, end_code)
    bcs = []
    for s in starts:
        for e in ends:
            if e < s:
                continue
            si = s + len(start_code)
            ei = e
            nt = ei - si
            if nt % 10 != 0:
                # barcode has invalid # of tokens
                continue
            if not check_digits(nt):
                # barcode has invalid # of digits
                continue
            bcs.append((s, e+len(end_code)))
    return bcs


def tokens_to_barcodes(
        tokens, bar_threshold=None, space_threshold=None,
        max_bar=None, max_space=None, ndigits=None, full=False):
    if bar_threshold is None:
        bar_threshold = find_token_threshold(tokens, 1)
    if max_bar is None:
        max_bar = bar_threshold * 3.
    if space_threshold is None:
        space_threshold = find_token_threshold(tokens, 0)
    if max_space is None:
        max_space = space_threshold * 3.
    # convert tokens to barcode string
    st = tokens_to_string(
        tokens, bar_threshold, space_threshold, max_bar, max_space)
    # TODO find a way to bring out ndigits
    bounds = find_all_barcode_bounds(st, ndigits=ndigits)
    bcs = []
    for bound in bounds:
        s, e = bound
        value = string_to_value(st[s+len(start_code):e-len(end_code)])
        if value > -1:
            bcs.append(Barcode(value, tokens[s:e]))
    if full:
        info = {
            'bar_threshold': bar_threshold, 'space_threshold': space_threshold,
            'max_bar': max_bar, 'max_space': max_space, 'ndigits': ndigits}
        return bcs, info
    return bcs
