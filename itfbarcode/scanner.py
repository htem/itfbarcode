#!/usr/bin/env python


from . import parser


def scan_image_y(im, start_x, end_x, y, scan_range=50):
    if scan_range < 1:
        raise ValueError("Invalid scan_range: %s" % scan_range)
    dy = 0
    bc = 0
    while isinstance(bc, int) and dy < scan_range:
        bc = parser.read_barcode(im[y + dy, start_x:end_x])
        dy += 1
    dy = 1
    while isinstance(bc, int) and dy < scan_range:
        bc = parser.read_barcode(im[y - dy, start_x:end_x])
        dy += 1
    return bc


def scan_image_x(im, start_y, end_y, x, scan_range=50):
    if scan_range < 1:
        raise ValueError("Invalid scan_range: %s" % scan_range)
    dx = 0
    bc = 0
    while isinstance(bc, int) and dx < scan_range:
        bc = parser.read_barcode(im[start_y:end_y, x + dx])
        dx += 1
    dx = 1
    while isinstance(bc, int) and dx < scan_range:
        bc = parser.read_barcode(im[start_y:end_y, x - dx])
        dx += 1
    return bc
