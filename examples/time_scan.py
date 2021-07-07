#!/usr/bin/env python

import sys
import time

import numpy
import pylab

import itfbarcode.linescan

sy = 500
ey = 620
rotate = False
flip = False
ratio = False
fn = 'example_02_image.png'
lf = pylab.imread


def valid_bc(bc):
    return bc.value < 5000


def tf(vs):
    return itfbarcode.linescan.scan(
        valid_bc, vs, {'ndigits': 6}, {})
        #lambda bc: bc.value < 5000, vs, {'ndigits': 6}, {})

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        if fn.split('.')[-1].lower() == 'npy':
            sy = 833
            ey = 1021
            rotate = True
            flip = True
            ratio = True
            lf = numpy.load
    im = lf(fn)
    if rotate:
        im = numpy.swapaxes(im, 0, 1)
    if flip:
        im = im[:, ::-1, :]
    im = im[sy:ey, :, :]
    if ratio:
        vs = (im[:, :, 0] / (im[:, :, 2].astype('f8') + 20.)).mean(axis=0)
    else:
        vs = im[:, :, 0].mean(axis=0)
    t0 = time.time()
    bcs, kw = tf(vs)
    t1 = time.time()
    print("Barcodes: %s" % (bcs, ))
    print("Time: %s" % (t1 - t0))
