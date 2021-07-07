#!/usr/bin/env python
"""
Load images
try finding barcodes
- crop?
- scan parameters
- optimize scan parameters
"""

import copy
import glob
import json
import os
import pickle
import sys

import numpy
import pylab

import itfbarcode.linescan
import itfbarcode.vis


default_meta = {
    'crop': [0, -1, 1507, 1748],
    'swap_axes': True,
    'denom': 20,
}

default_kwargs = {
    'ndigits': 6,
    'ral': None,
    'min_length': 3,
}

# datasets
ds = sorted([i for i in glob.glob('*_*') if os.path.isdir(i)])
if len(sys.argv) > 1:
    nds = []
    for s in sys.argv[1:]:
        if s.strip('/') in ds:
            nds.append(s)
    ds = nds

rals = [None, ] + range(10, 600, 10)
min_lengths = range(0, 10)


def is_image_fn(fn):
    _, e = os.path.splitext(fn)
    e = e.lower().replace(".", "")
    if e in ('png', 'jpg'):
        return True
    return False


def load_image(fn):
    im = pylab.imread(fn)
    if im.ndim == 4:
        im = im[:, :, :3]
    if im.dtype in ('f4', 'f8'):
        im = (im * 255).astype('u1')
    if im.ndim == 1:
        im = numpy.dstack((im, im, im))
    assert im.dtype == 'u1'
    assert im.ndim == 3
    return im


def crop_image(im, meta):
    if 'crop' in meta:
        c = meta['crop']
        im = im[c[0]:c[1], c[2]:c[3], :]
    if meta.get('swap_axes', False):
        im = numpy.swapaxes(im, 0, 1)
    return im


def image_to_values(im, meta):
    im = crop_image(im, meta)
    # colorspace
    if 'denom' in meta:
        im = im[:, :, 0] / (im[:, :, 2].astype('f4') + meta['denom'])
    # mean
    vs = numpy.mean(im, axis=0)
    return vs


rdir = 'results'
if not os.path.exists(rdir):
    os.makedirs(rdir)
results = {}
for d in ds:
    print("== dataset %s ==" % d)
    fns = sorted([fn for fn in glob.glob('%s/*' % d) if is_image_fn(fn)])
    mfn = '%s/meta.json' % d
    if os.path.exists(mfn):
        with open(mfn, 'r') as mf:
            meta = json.load(mf)
    else:
        meta = copy.deepcopy(default_meta)
    print meta
    rd = '%s/%s' % (rdir, d)
    if not os.path.exists(rd):
        os.makedirs(rd)
    # load images
    ims = [load_image(fn) for fn in fns]
    # convert to values
    vs = [image_to_values(im, meta) for im in ims]
    # find fits
    f, a, b, kw = itfbarcode.linescan.search_for_fit(
        lambda bc: bc.value < 5000, vs[0], rals, min_lengths, **default_kwargs)
    if kw is None:
        print("Failed to find any matches... using default kwargs")
        kw = default_kwargs
    pylab.figure(1)
    pylab.clf()
    pylab.imshow(a, interpolation='nearest')
    if b is not None:
        pylab.scatter(b['min_length_i'], b['ral_i'], color='b')
    if kw is not None:
        ri = rals.index(kw['ral'])
        mi = min_lengths.index(kw['min_length'])
        pylab.scatter(mi, ri, color='g')
        pylab.title("%r" % kw)
    pylab.savefig('%s/fit.jpg' % rd)
    # find best fit
    with open('%s/fit.json' % rd, 'w') as of:
        pickle.dump({
            'f': f, 'a': a, 'b': b, 'kw': kw, 'meta': meta,
        }, of)
    # plot fit
    # evaluate across all
    bcs = []
    for v in vs:
        bc = itfbarcode.linescan.to_barcodes(v, **kw)
        if bc is None or len(bc) == 0:
            print("rescanning...")
            nf, na, nb, nkw = itfbarcode.linescan.search_for_fit(
                lambda bc: bc.value < 5000, v, rals, min_lengths,
                **default_kwargs)
            if nkw is None:
                print("Failed to find any matches... using default kwargs")
                nkw = default_kwargs
            bc = itfbarcode.linescan.to_barcodes(v, **nkw)
            # TODO if still no luck, try scanning denom, crop
        bcs.append(bc)
    #bcs = [itfbarcode.linescan.to_barcodes(v, **kw) for v in vs]
    for i in xrange(len(bcs)):
        im = ims[i]
        cim = crop_image(im, meta)
        bc = bcs[i]
        pylab.figure(1)
        pylab.clf()
        pylab.subplot(211)
        pylab.imshow(cim, interpolation='nearest')
        pylab.subplot(212)
        pylab.plot(vs[i])
        if bc is not None and len(bc):
            minv = bc[0].start
            maxv = bc[-1].end
            for sp in (211, 212):
                pylab.subplot(sp)
                for b in bc:
                    itfbarcode.vis.plot_tokens(b.tokens)
                    minv = min(minv, b.start)
                    maxv = max(maxv, b.end)
                pylab.xlim(minv, maxv)
            pylab.subplot(212)
            t = " ".join(["%06i" % b.value for b in bc])
            pylab.title(t)
        # save figure
        pylab.suptitle(os.path.basename(fns[i]))
        sfn = '%s/%02i' % (rd, i)
        pylab.savefig(sfn)
    results[d] = bcs
    print("%i images" % len(ims))
    print("%i barcodes" % sum([len(b) for b in bcs]))
    print("%i missing" % len([b for b in bcs if len(b) == 0]))
    print("%r" % kw)
    print("")
