#!/usr/bin/env python
"""
Linescan to tokens
"""

import copy
import inspect

import numpy
import scipy.ndimage


from . import parser
from .objects import Token, Barcode


def binarize(vs, ral=None):
    if ral is None:
        t = vs.mean()
    else:
        t = scipy.ndimage.convolve1d(
            vs, numpy.ones(ral, dtype='f8') / ral, mode='reflect')
    return vs > t


def to_tokens(bvs, min_length=None):
    dbvs = numpy.diff(bvs.astype('int'))
    einds = numpy.where(dbvs != 0)[0]
    if len(einds) < 2:
        return []
    start = einds[0]
    if dbvs[start] > 0:
        state = 1
    else:
        state = 0
    start += 1
    tokens = []
    if min_length is None:
        min_length = 0
    for ei in einds[1:]:
        if ei < start:
            continue
        if (ei - start) < min_length:
            # this bar/space is too small, make it wider
            nei = start + min_length + 1
            if nei >= len(bvs):
                break
            if bvs[nei] and state == 0:  # rising edge
                tokens.append(Token(state, start, nei))
                state = 0
                start = nei
                continue
            if not bvs[nei] and state == 1:  # falling edge
                tokens.append(Token(state, start, nei))
                state = 0
                start = nei
                continue
            continue
        if dbvs[ei] > 0:  # rising edge
            if state == 1:
                continue
            tokens.append(Token(state, start, ei + 1))
            state = 1
            start = ei + 1
        else:
            if state == 0:
                continue
            tokens.append(Token(state, start, ei + 1))
            state = 0
            start = ei + 1
    return tokens


def old_to_tokens(vs, min_length=None):
    """ral = running average length"""
    if min_length is None:
        min_length = numpy.inf
    tokens = []
    start = None
    state = int(b[0])
    for (i, bv) in enumerate(b):
        if (bv != state):
            if start is not None:
                if ((i - start) < min_length):
                    continue
                tokens.append(Token(state, start, i))
                #if isinstance(t, numpy.ndarray) and t.size != 1:
                #    weight = numpy.max(
                #         numpy.abs(vs[start:i+1] - t[start:i+1]))
                #else:
                #    weight = numpy.max(numpy.abs(vs[start:i+1] - t))
                #if numpy.abs(weight) > 0.1:
                #    tokens.append(Token(state, start, i))
                #    tokens[-1].weight = weight
            start = i
            state = int(bv)
    return tokens


def to_barcodes(
        vs, ral=None, min_length=None,
        bar_threshold=None, space_threshold=None,
        max_bar=None, max_space=None,
        ndigits=None, full=False):
    bvs = binarize(vs, ral=ral)
    tokens = to_tokens(bvs, min_length=min_length)
    r = parser.tokens_to_barcodes(
        tokens, bar_threshold=bar_threshold, space_threshold=space_threshold,
        max_bar=max_bar, max_space=max_space,
        ndigits=ndigits, full=full)
    if full:
        bcs, pinfo = r
        info = {'ral': ral, 'min_length': min_length, 'tokens': tokens}
        info.update(pinfo)
        # TODO measure error for each barcodes tokens
        return bcs, info
    return r


def measure_fit(bc, bci):
    tws = numpy.array([b.width for b in bc.bars])
    hm = tws > bci['bar_threshold']
    ot = (numpy.mean(tws[hm]) + numpy.mean(tws[~hm])) / 2.
    r = {'bar': {
        'optimal_threshold': ot,
        'spread': tws.ptp(),
    }}
    tws = numpy.array([b.width for b in bc.bars])
    hm = tws > bci['space_threshold']
    ot = (numpy.mean(tws[hm]) + numpy.mean(tws[~hm])) / 2.
    r['space'] = {
        'optimal_threshold': ot,
        'spread': tws.ptp(),
    }
    return r


def _middle_value(vs, index=True):
    svs = sorted(vs)
    mv = svs[len(svs) // 2]
    if index:
        return mv, vs.index(mv)
    return mv


def _combine_kwargs(kws):
    # look at best.others, take middle value of ral, min_length
    rals = [kw['ral'] for kw in kws]
    middle_ral, ri = _middle_value(rals, index=True)
    min_lengths = [kw['min_length'] for kw in kws]
    middle_min_length, mi = _middle_value(min_lengths, index=True)
    if ri == mi:
        return kws[ri]
    # only select kws with the middle ral
    kws = [kw for kw in kws if kw['ral'] == middle_ral]
    min_lengths = [kw['min_length'] for kw in kws]
    middle_min_length, mi = _middle_value(min_lengths, index=True)
    return kws[mi]


def _best_fit_to_kwargs(best):
    if best is None:
        return None
    if len(best.get('others', [])) > 0:
        kws = [_best_fit_to_kwargs(o) for o in best['others']]
        b = copy.deepcopy(best)
        del b['others']
        kws.append(_best_fit_to_kwargs(b))
        return _combine_kwargs(kws)
    valid_args = inspect.getargspec(to_barcodes).args
    if 'vs' in valid_args:
        valid_args.remove('vs')
    kw = {}
    for k in valid_args:
        if k in best:
            kw[k] = best[k]
    # pull out 'optimal' thresholds
    if 'space' in best and 'optimal_threshold' in best['space']:
        best['space_threshold'] = best['space']['optimal_threshold']
    if 'bar' in best and 'optimal_threshold' in best['bar']:
        best['bar_threshold'] = best['bar']['optimal_threshold']
    return kw


def search_for_fit(vbc, vs, rals, min_lengths, **kwargs):
    """ vbc: function to check if barcode is valid"""
    r = []
    kwargs['full'] = True
    best = None
    sa = numpy.empty((len(rals), len(min_lengths)))
    for (ral_i, ral) in enumerate(rals):
        sr = []
        bvs = binarize(vs, ral)
        for (min_length_i, min_length) in enumerate(min_lengths):
            tokens = to_tokens(bvs, min_length)
            bcs, bci = parser.tokens_to_barcodes(
                tokens, bar_threshold=kwargs.get('bar_threshold', None),
                space_threshold=kwargs.get('space_threshold', None),
                max_bar=kwargs.get('max_bar', None),
                max_space=kwargs.get('max_space', None),
                ndigits=kwargs.get('ndigits', None),
                full=True)
            kwargs['ral'] = ral
            kwargs['min_length'] = min_length
            #bcs, bci = to_barcodes(vs, **kwargs)
            # test that barcodes are valid
            bcs = [bc for bc in bcs if vbc(bc)]
            # how to handle evaluating fit for multiple barcodes?
            if len(bcs) > 0:
                fits = [measure_fit(b, bci) for b in bcs]
                spreads = [
                    f['bar']['spread'] * f['space']['spread'] for f in fits]
                max_i = max(list(range(len(spreads))), key=lambda i: spreads[i])
                spread = spreads[max_i]
                bci.update(fits[max_i])
                bci['spreads'] = spreads
                bci['fits'] = fits
                #bci.update(measure_fit(bcs[0], bci))
            else:
                bci.update({
                    'spreads': [],
                    'fits': [],
                    'bar': {'spread': numpy.nan},
                    'space': {'spread': numpy.nan}})
                spread = numpy.nan
            bci['ral_i'] = ral_i
            bci['min_length_i'] = min_length_i
            sa[ral_i, min_length_i] = spread
            bci.update({
                'spread': spread, 'ral': ral, 'min_length': min_length})
            if not numpy.isnan(spread):
                if best is None or spread > best['spread']:
                    best = copy.deepcopy(bci)
                elif spread == best['spread']:
                    best['others'] = (
                        best.get('others', []) + [copy.deepcopy(bci), ])
            sr.append(bci)
        r.append(sr)
    kw = _best_fit_to_kwargs(best)
    return r, sa, best, kw


# TODO bring out scan parameters
def scan(vbc, vs, kwargs, scan_kwargs=None):
    """
    scan_kwargs:
        scan = True or False, if scanning is enabled
        ral_min, ral_max
        ral_scan (scan +- ral_scan in steps of size ral-step)
        ral_none include None in ral scan
        min_length_min, min_length_max
        min_length_scan (same as ral_scan)
        min_length_none (same as ral_none)
    """
    if scan_kwargs is None:
        scan_kwargs = {}
    # remove invalid barcodes
    bcs = [bc for bc in to_barcodes(vs, **kwargs) if vbc(bc)]
    if len(bcs) == 0 and scan_kwargs.get('scan', True):
        # scan around existing value
        if kwargs.get('ral', None) is None:
            l = scan_kwargs.get('ral_min', 10)
            r = scan_kwargs.get('ral_max', 600)
        else:
            l = max(
                scan_kwargs.get('ral_min', 5),
                kwargs['ral'] - scan_kwargs.get('ral_scan', 200))
            r = min(
                scan_kwargs.get('ral_max', 600),
                kwargs['ral'] + scan_kwargs.get('ral_scan', 200))
        if scan_kwargs.get('ral_none', True):
            rals = [None, ] + list(range(l, r, scan_kwargs.get('ral_step', 10)))
        else:
            rals = list(range(l, r, scan_kwargs.get('ral_step', 10)))
        if kwargs.get('min_length', None) is None:
            l = scan_kwargs.get('min_length_min', 1)
            r = scan_kwargs.get('min_length_max', 10)
        else:
            l = max(
                scan_kwargs.get('min_length_min', 1),
                kwargs['min_length'] - scan_kwargs.get('min_length_scan', 5))
            r = min(
                scan_kwargs.get('min_length_max', 10),
                kwargs['min_length'] + scan_kwargs.get('min_length_scan', 5))
        if scan_kwargs.get('min_length_none', True):
            min_lengths = [None, ] + list(range(l, r))
        else:
            min_lengths = list(range(l, r))
        print("Scanning...")
        _, _, b, kw = search_for_fit(vbc, vs, rals, min_lengths, **kwargs)
        if kw is None:
            return [], kwargs
        print(("Found: %s" % kw))
        kwargs = kw
        bcs = [bc for bc in to_barcodes(vs, **kw) if vbc(bc)]
    # TODO retries?
    return bcs, kwargs

def find_barcode_locations(vs_filt, min_inter_bc_width=40):
    # now find our barcodes from the troughs
    start_i = None
    width_count = 0
    troughs = []
    m = vs_filt.mean()
    for (i, v) in enumerate(vs_filt):
        if v < m:
            if start_i is None:
                start_i = i
            width_count += 1
        else:
            if start_i is not None:
                troughs.append((start_i, width_count))
                start_i = None
                width_count = 0
    bc_coords = []
    bc_start = None
    for t in troughs:
        si, w = t
        if w > min_inter_bc_width:
            if bc_start is not None:
                bc_coords.append((bc_start,si))
            bc_start = si + w
    if len(bc_coords) == 0:
        return []
    ret_bcs = []
    for bcc in bc_coords:
        s, e = bcc
        tokns = []
        if ( e - s ) < 400 or (e - s ) > 800:
            continue
	ret_bcs.append(s + (e-s)/2)
    return ret_bcs


def _find_wide_spaces(vs):
    ts = sorted(vs)
    ti = 5
    troughs = []
    steps = 0
    nsteps = 0
    while len(troughs) < 6 and ti < len(ts) - 1:
        ti += 1
        t= ts[ti]
        # find 6 wide spaces all > 10 pixels wide
        # find troughs
        si = None
        wc = 0
        troughs = []
        for (i, v) in enumerate(vs):
            if v < t:
                if si is None:
                    si = i
                wc += 1
            else:
                if si is not None:
                    if wc > 10:
                        troughs.append((si,wc))
                    si = None
                    wc = 0
        if len(troughs) > 5:
            steps += 1
        if steps > 0 and len(troughs) > 6:
            steps = nsteps + 10
            ti -= 2
        if steps < nsteps:
            troughs = []
    print((t, steps))
    return troughs


def scan_approximate(vs, vs_filt, kwargs):
    """
    vs is the mean of the image
    vs_filt is a harshly filtered linescan

    kwargs:
        possible_bcs= array of possible barcodes (None default)
        ratio = ratio of wide token to small token (2.5 default)
        slop = amount of extra cut on either end of the barcode (1.0 default)
    """
    ratio = kwargs.get("ratio",2.5)
    slop = kwargs.get("slop",1.0)
    min_inter_barcode_width = kwargs.get("min_inter_barcode_width", 75)

    denom = 13 * ratio + 24.0 + slop
    token_spacing = {
      'b': 1/denom,
      's': 1/denom,
      'B': ratio/denom,
      'S': ratio/denom
    }

    # first generate what we would expect to find
    pbcs = kwargs.get("possible_bcs",None)
    if pbcs is None:
         print("Cannot do approximate barcodes if there are no possible bcs")
         return []
    poss_tokens = [parser.gen_tokens(v,ndigits=6) for v in pbcs]

    # calculate wide space offsets for the possible barcodes
    ws_offsets = []
    for tok in poss_tokens:
        wso = []
        l = token_spacing['b'] / 2.0
        for c in tok:
            l += token_spacing[c] / 2.0
            if c == 'S':
                wso.append(l)
            l += token_spacing[c] / 2.0
        ws_offsets.append(wso)
    ws_offsets = numpy.array(ws_offsets)

     # now find our barcodes from the troughs
    start_i = None
    width_count = 0
    troughs = []
    m = vs_filt.mean()
    for (i, v) in enumerate(vs_filt):
        if v < m:
            if start_i is None:
                start_i = i
            width_count += 1
        else:
            if start_i is not None:
                troughs.append((start_i, width_count))
                start_i = None
                width_count = 0
    bc_coords = []
    bc_start = None
    for t in troughs:
        si, w = t
        if w > min_inter_barcode_width:
            if bc_start is not None:
                bc_coords.append((bc_start,si))
            bc_start = si + w
    if len(bc_coords) == 0:
        return []
    bcs = []
    ret_bcs = []
    for bcc in bc_coords:
        s, e = bcc
        tokns = []
        if ( e - s ) < 400 or (e - s ) > 800:
            continue
        bc = { # TODO need to figure out getting this into a barcode object
             'start': s, 'end':e}
        bc_vs = vs_filt[s:e]
        bs_ws = _find_wide_spaces(bc_vs)
        if len(bs_ws) == 0:
            continue
        bc['wide_spaces'] = bs_ws
        w = float(e - s)
        print(bs_ws[0])
        tokns.append(Token(1,s,bs_ws[0][0]))
        for v in bs_ws:
            tokns.append(Token(0,v[0],v[0]))
        tokns.append(Token(1,bs_ws[-1][0],e))
        ret_bcs.append(Barcode(-1,tokns))
        bc['normed_ws'] = [v[0] / w for v in bs_ws]
        print(bc)
        bcs.append(bc)

    nws = numpy.array([bc['normed_ws'] for bc in bcs])
    if nws.size == 0:
        return ret_bcs
    if nws.ndim < 2:
        return ret_bcs
    nbcs = len(bcs)
    match_errors = []
    for i in range(len(ws_offsets) - nbcs + 1 ):
        e = ws_offsets[i:i+nbcs] - nws
        match_errors.append((i,numpy.sum(numpy.abs(e)), pbcs[i:i+nbcs]))

    match = min(match_errors, key=lambda i: i[1])
    for i in range(len(match[2])):
        ret_bcs[i].value = match[2][i]
    return ret_bcs
