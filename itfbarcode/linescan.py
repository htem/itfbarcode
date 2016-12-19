#!/usr/bin/env python
"""
Linescan to tokens
"""

import numpy
import scipy.ndimage


from . import parser
from .objects import Token


def to_tokens(vs, ral=None, min_length=None):
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
            start = i
            state = int(bv)
    return tokens


def to_barcodes(
        vs, ral=None, min_length=None,
        bar_threshold=None, space_threshold=None,
        max_bar=None, max_space=None,
        ndigits=None):
    tokens = to_tokens(vs, ral=ral, min_length=min_length)
    return parser.tokens_to_barcodes(
        tokens, bar_threshold=bar_threshold, space_threshold=space_threshold,
        max_bar=max_bar, max_space=max_space,
        ndigits=ndigits)
