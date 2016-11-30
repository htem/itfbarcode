#!/usr/bin/env python

import pylab


def draw_info(info):
    pylab.axhline(info['y'])
    for t in info['tokens']:
        pylab.axvline(t[1])
        pylab.axvline(t[1]+t[2])
