#!/usr/bin/env python

import pylab

import itfbarcode 


if __name__ == '__main__':
    im = pylab.imread('example_02_image.png')[:, :, 0]
    barcode = itfbarcode.scanner.scan_image_y(im, 820, 1190, 570, 50)
    print "Barcode is: {}".format(barcode)
