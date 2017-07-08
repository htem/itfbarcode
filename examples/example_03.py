#!/usr/bin/env python

import time

import pylab

import itfbarcode.linescan

sy = 500
ey = 620


def tf(vs):
    return itfbarcode.linescan.scan(
        lambda bc: bc.value < 5000, vs, {'ndigits': 6}, {})

if __name__ == '__main__':
    im = pylab.imread('example_02_image.png')[sy:ey, :, 0]
    vs = im.mean(axis=0)
    t0 = time.time()
    bcs, kw = tf(vs)
    t1 = time.time()
    print "Barcodes: %s" % (bcs, )
    print "Time: %s" % (t1 - t0)
