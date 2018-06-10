"""
Created on Wed Mar 28 15:54:01 2018
@name: foreground_detect.py
@author: LENOVO
"""

import numpy as np
import cv2 as cv

def init_fg():
    return cv.bgsegm.createBackgroundSubtractorMOG(nmixtures=5)

def fg_detect(image, foregrd):
    kernel = np.ones((10,10), np.uint8)
        
    fg_mask = foregrd.apply(image)
    fg_filter = cv.medianBlur(fg_mask, 5)
    fg_closing = cv.dilate(fg_filter, kernel, iterations=1)
    fg_closing = cv.erode(fg_closing, kernel, iterations=1)

    return fg_mask, fg_filter, fg_closing

