# -*- coding: utf-8 -*-
"""
Created on Tue May 01 15:54:01 2018

@author: LENOVO
"""

import numpy as np

def thresholded(center, pix):
    out = []
    for p in pix:
        if p >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel(img, idx, idy, default=0):
    try:
        return img[int(idx), int(idy)]
    except IndexError:
        return default
    
def lbp(image, visualize=False):
    transform = np.zeros((image.shape[0], image.shape[1]), dtype=np.double)
    height, width = image.shape[:2]
    
    for x in range(0, height):
        for y in range(0, width):
            center        = image[x,y]
            top_left      = get_pixel(image, x-1, y-1)
            top_up        = get_pixel(image, x, y-1)
            top_right     = get_pixel(image, x+1, y-1)
            right         = get_pixel(image, x+1, y)
            left          = get_pixel(image, x-1, y)
            bottom_left   = get_pixel(image, x-1, y+1)
            bottom_right  = get_pixel(image, x+1, y+1)
            bottom_down   = get_pixel(image, x, y+1)
    
            values = thresholded(center, [top_left, top_up, top_right, right, bottom_right, bottom_down, bottom_left, left])
            
            weights = [1, 2, 4, 8, 16, 32, 64, 128]
            res = 0
            for a in range(0, len(values)):
                res += weights[a] * values[a]
                
            transform.itemset((x,y), res)
            
    hist, bins = np.histogram(transform.ravel(),256,[0,256]) 
    
    eps = 1e-5
    hist = hist.astype("float")
    norm_hist = hist / np.sqrt(np.sum(hist ** 2) + eps ** 2)
    norm_hist = np.minimum(norm_hist, 0.2)
    norm_hist = norm_hist / np.sqrt(np.sum(norm_hist ** 2) + eps ** 2)
    
    if visualize:
        return norm_hist, np.asarray(transform)
    else:
        return norm_hist