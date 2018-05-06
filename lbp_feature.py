# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:41:37 2018

@author: LENOVO
"""

import numpy as np
import math

def bilinear_interpolation(image, r, c):
    x1, y1 = int(r), int(c)
    x2, y2 = math.ceil(r), math.ceil(c)
    
    r1 = (x2-r)/(x2-x1) * get_pixel(image, x1, y1) + (r-x1)/(x2-x1) * get_pixel(image, x2, y1)
    r2 = (x2-r)/(x2-x1) * get_pixel(image, x1, y2) + (r-x1)/(x2-x1) * get_pixel(image, x2, y2)
    
    return (y2-c) / (y2-y1) * r1 + (c-y1) / (y2-y1) * r2

def thresholding(center, pix):
    out = []
    for p in pix:
        if p >= center:
            out.append(1)
        else:
            out.append(0)
        
    return out

def get_pixel(img, idx, idy):
    if idx < int(len(img)) - 1 and idy < len(img[0]):
        return img[int(idx), int(idy)]
    else:
        return 0

def lbp(image, nPoints, rad):
    transform = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
    
    for x in range(0, len(image)):
        for y in range(0, len(image)):
            center = image[x, y]
            pixels = []
            for point in range(0, nPoints):
                r = x + rad * math.cos(2 * math.pi * point / nPoints)
                c = y - rad * math.sin(2 * math.pi * point / nPoints)
                if r<0 or c<0:
                    pixels.append(0)
                    continue
                if int(r) == r:
                    if int(c) != c:
                        c1 = int(c)
                        c2 = math.ceil(c)
                        w1 = (c2 - c) / (c2 - c1)
                        w2 = (c - c1) / (c2 - c1)
                        pixels.append(int((w1 * get_pixel(image, int(r), int(c)) + w2 * get_pixel(image, int(r), math.ceil(c))) / (w1 + w2)))
                    else:
                        pixels.append(get_pixel(image, int(r), int(c)))
                elif int(c) == c:
                    r1 = int(r)
                    r2 = math.ceil(r)
                    w1 = (r2 - r) / (r2 - r1)
                    w2 = (r - r1) / (r2 - r1)                
                    pixels.append((w1 * get_pixel(image, int(r), int(c)) + w2 * get_pixel(image, math.ceil(r), int(c))) / (w1 + w2))
                else:
                    pixels.append(bilinear_interpolation(image, r, c))
                    
            values = thresholding(center, pixels)
            res = 0
            for a in range(0, len(values)):
                res += values[a] * (2**a)
            
            transform.itemset((x,y), res)
    
    hist, bins = np.histogram(transform.flatten(),256,[0,256]) 
    
    eps = 1e-5
    hist = hist.astype("float")
    norm_hist = hist / np.sqrt(np.sum(hist ** 2) + eps ** 2)
    norm_hist = np.minimum(norm_hist, 0.2)
    norm_hist = norm_hist / np.sqrt(np.sum(norm_hist ** 2) + eps ** 2)
    
    return norm_hist, np.asarray(transform)
