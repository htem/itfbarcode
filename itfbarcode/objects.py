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


BAR = 1
SPACE = 0

state_strings = ['space', 'bar']


class Token(object):
    def __init__(self, state, start, end):
        self.state = state  # 1 = bar, 0 = space
        self.start = start
        self.end = end

    @property
    def width(self):
        return self.end - self.start

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
