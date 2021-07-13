import cv2 as cv
import time
import sys
import numpy
from itfbarcode import linescan
import os
import json

jspath = 'test'
imgpath = 'tapecameraimgs-crop'
configpath = 'linescanconfig'

bcslist = []
failcounter = 0
successcounter = 0
avgtime = 0
failedlist = []
for fn in os.listdir(imgpath):
    filename = os.path.join(imgpath, fn)
    # load image data using openvc
    img = cv.imread(filename)
    # add red filter and convert image data into only one channel
    vs = (img[:,:,2]/(img[:,:,0].astype('f8')+20.)).mean(axis=0)
    t0 = time.time()
    bcs, kw = linescan.scan(lambda a:a.value<20170, vs, {'ndigits':6}, {})
    t1 = time.time()
   # print ("img: %s" % (fn))
   # print ("barcode: %s" % (bcs))
   # print ("time: %s" % (t1 - t0))
    bcslist.append(bcs)
    avgtime = avgtime + t1 -t0
    if bcs == []:
        failcounter += 1
        failedlist.append(fn)
        print('failed img: %s' %(fn))
        print('empty list[] found')
    else:
        realbcs = int(fn.strip('.png').split('_')[1])
    
        bcsresult = bcs[0]
        #if realbcs - 26 or realbcs - 25 or realbcs - 27 == bcsresult.value:
        #if 24 or 25 or 23 == 25
        #if 24 or 25 or False 
        # checking if the barcode read result is correct, the real barcode of the middle one out of 4 that in one image was at the end of the image name and often have a offset of 26, since it might read any of the 4 barcodes so using all 24, 25, 26 and 27 to check.
        if abs(bcsresult.value - realbcs) in (24, 25, 26, 27):
            
            successcounter += 1
        else:
            print('barcode read as: %s' %(bcs,))
            print('real barcode: %s' %(realbcs))
   #  print(len(bcslist))
print(bcslist)
print(failcounter)
print(successcounter)
print('fail ratio %s' %(float(failcounter)/len(bcslist)))
print(avgtime/len(bcslist))
print(failedlist)
    # save the result to json files if needed
    # config = {}
    # config['img'] = fn
    # config['barcode'] = bcs
    # config['time'] = t1 - t0
    # with open(os.path.join(configpath, fn[:-5] +'.json'), 'w') as jfile:
        # json.dump(config, jfile)
