#!/usr/bin/env python
"""
Barcode:
    n digits
    as string
    as number
    rectangle
    center
    tokens
    scanline

Reparse for different bar/space thresholds
Reparse for different binerazation threshold
"""

import numpy
import scipy.ndimage


BAR = 1
SPACE = 0

state_strings = ['space', 'bar']


class Linescan(object):
    def __init__(self, vs, ral=None):
        self.vs = vs
        self.binarize(ral)

    def binarize(self, ral=None):
        if ral is None:
            t = self.vs.mean()
        else:
            t = scipy.ndimage.convolve1d(
                self.vs, numpy.ones(ral, dtype='f8') / ral, mode='reflect')
        self.bvs = self.vs > t

    def to_tokens(self, min_length=None):
        dbvs = numpy.diff(self.bvs.astype('int'))
        einds = numpy.where(dbvs != 0)[0]
        if len(einds) < 2:
            return []
        start = einds[0]
        if dbvs[start] > 0:
            state = 1
        else:
            state = 0
        start += 1
        self.tokens = []
        for ei in einds[1:]:
            if ei < start:
                continue
            if (ei - start) < min_length:
                # this bar/space is too small, make it wider
                nei = start + min_length
                if self.bvs[nei+1] and state == 0:  # rising edge
                    self.tokens.append(Token(state, start, nei + 1))
                    state = 0
                    start = nei + 1
                    continue
                if not self.bvs[nei+1] and state == 1:  # falling edge
                    self.tokens.append(Token(state, start, nei + 1))
                    state = 0
                    start = nei + 1
                    continue
                continue
            if dbvs[ei] > 0:  # rising edge
                if state == 1:
                    continue
                self.tokens.append(Token(state, start, ei + 1))
                state = 1
                start = ei + 1
            else:
                if state == 0:
                    continue
                self.tokens.append(Token(state, start, ei + 1))
                state = 0
                start = ei + 1
        return self.tokens


class Token(object):
    def __init__(self, state, start, end):
        self.state = state  # 1 = bar, 0 = space
        self.start = start
        self.end = end
        self.width = self.end - self.start

    def __repr__(self):
        return "Token(%s, %s[%s, %s])" % (
            state_strings[self.state], self.width, self.start, self.end)


class Barcode(object):
    def __init__(self, value, tokens):
        self.tokens = tokens
        self.value = value

    def __repr__(self):
        return "Barcode(%s, %s)" % (self.value, self.center)

    @property
    def start(self):
        return self.tokens[0].start

    @property
    def end(self):
        return self.tokens[-1].end

    @property
    def width(self):
        return self.end - self.start

    @property
    def center(self):
        return (self.start + self.end) / 2.

    @property
    def bars(self):
        return [t for t in self.tokens if t.state == BAR]

    @property
    def spaces(self):
        return [t for t in self.tokens if t.state == SPACE]
