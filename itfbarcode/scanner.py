#!/usr/bin/env python

import numpy

from . import parser


def scan_image_y(
        im, start_x=None, end_x=None, y=None, scan_range=None, **kwargs):
    if start_x is None:
        start_x = 0
    if end_x is None:
        end_x = im.shape[1]
    if y is None:
        y = im.shape[0] / 2
    if scan_range is None:
        scan_range = im.shape[0] / 2
    if scan_range < 1:
        raise ValueError("Invalid scan_range: %s" % scan_range)
    full = kwargs.get('full', False)
    dy = 0
    bc = 0
    while not parser.is_valid(bc) and dy < scan_range:
        bc = parser.read_barcode(im[y + dy, start_x:end_x], **kwargs)
        if full:
            bc, info = bc
            info['y'] = y + dy
        dy += 1
    dy = 1
    while not parser.is_valid(bc) and dy < scan_range:
        bc = parser.read_barcode(im[y - dy, start_x:end_x], **kwargs)
        if full:
            bc, info = bc
            info['y'] = y + dy
        dy += 1
    if full:
        return bc, info
    return bc


def scan_image_x(
        im, start_y=None, end_y=None, x=None, scan_range=None, **kwargs):
    if start_y is None:
        start_y = 0
    if end_y is None:
        end_y = im.shape[0]
    if x is None:
        x = im.shape[1] / 2
    if scan_range is None:
        scan_range = im.shape[1] / 2
    if scan_range < 1:
        raise ValueError("Invalid scan_range: %s" % scan_range)
    full = kwargs.get('full', False)
    dx = 0
    bc = 0
    while not parser.is_valid(bc) and dx < scan_range:
        bc = parser.read_barcode(im[start_y:end_y, x + dx], **kwargs)
        if full:
            bc, info = bc
            info['x'] = x + dx
        dx += 1
    dx = 1
    while not parser.is_valid(bc) and dx < scan_range:
        bc = parser.read_barcode(im[start_y:end_y, x - dx], **kwargs)
        if full:
            bc, info = bc
            info['x'] = x + dx
        dx += 1
    if full:
        return bc, info
    return bc
