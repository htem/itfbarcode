#!/usr/bin/env python

import numpy
import pylab


def plot_tokens(tokens, bar_color='b', space_color='r', alpha=0.3):
    colors = {1: bar_color, 0: space_color}
    for t in tokens:
        pylab.axvspan(t.start, t.end, color=colors[t.state], alpha=alpha)


def plot_barcode(im, bc, bci):
    pylab.subplot(221)
    # show image
    pylab.imshow(im)
    # show tokens (colored by state and size)
    bar_colors = ['b', 'g']
    space_colors = ['r', 'm']
    for b in bc.bars:
        c = bar_colors[int(b > bci['bar_threshold'])]
        pylab.axvspan(b.start, b.end, color=c, alpha=0.5)
    for s in bc.spaces:
        c = space_colors[int(s > bci['space_threshold'])]
        pylab.axvspan(s.start, s.end, color=c, alpha=0.5)
    pylab.xlim(bc.start, bc.end)
    # plot tokens with thresholds
    pylab.subplot(223)
    tws = numpy.array([b.width for b in bc.bars])
    hm = tws > bci['bar_threshold']
    ot = (numpy.mean(tws[hm]) + numpy.mean(tws[~hm])) / 2.
    pylab.title('%0.4f [%0.4f]' % (ot, tws.ptp()))
    pylab.scatter(range(tws.size), tws)
    pylab.axhline(bci['bar_threshold'], color='k')
    pylab.axhline(ot, color='g')
    pylab.axhline(bci['max_bar'], color='r')
    pylab.subplot(224)
    tws = numpy.array([b.width for b in bc.spaces])
    hm = tws > bci['space_threshold']
    ot = (numpy.mean(tws[hm]) + numpy.mean(tws[~hm])) / 2.
    pylab.title('%0.4f [%0.4f]' % (ot, tws.ptp()))
    pylab.axhline(bci['space_threshold'], color='k')
    pylab.axhline(ot, color='g')
    pylab.axhline(bci['max_space'], color='r')
    pylab.scatter(range(tws.size), tws)


def draw_info(info):
    pylab.axhline(info['y'])
    for t in info['tokens']:
        pylab.axvline(t[1])
        pylab.axvline(t[1]+t[2])
