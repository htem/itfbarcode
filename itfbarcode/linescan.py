#!/usr/bin/env python
"""
Linescan to tokens
"""

import copy
import inspect

import numpy
import scipy.ndimage


from . import parser
from .objects import Token


def to_tokens(vs, ral=None, min_length=None):
    """ral = running average length"""
    if min_length is None:
        min_length = numpy.inf
    if ral is None:
        t = vs.mean()
    else:
        t = scipy.ndimage.convolve1d(
            vs, numpy.ones(ral, dtype='f8') / ral, mode='reflect')
    b = vs > t
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
    tokens = to_tokens(vs, ral=ral, min_length=min_length)
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
        for (min_length_i, min_length) in enumerate(min_lengths):
            kwargs['ral'] = ral
            kwargs['min_length'] = min_length
            bcs, bci = to_barcodes(vs, **kwargs)
            if 'tokens' in bci:
                del bci['tokens']
            # test that barcodes are valid
            bcs = [bc for bc in bcs if vbc(bc)]
            # how to handle evaluating fit for multiple barcodes?
            if len(bcs) > 0:
                fits = [measure_fit(b, bci) for b in bcs]
                spreads = [
                    f['bar']['spread'] * f['space']['spread'] for f in fits]
                max_i = max(range(len(spreads)), key=lambda i: spreads[i])
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
