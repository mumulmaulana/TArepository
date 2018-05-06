import numpy as np
import cv2 as cv

def init_fg():
    return cv.bgsegm.createBackgroundSubtractorMOG()

def fg_detect(image, foregrd):
    kernel = np.ones((15,15), np.uint8)
        
    fg_mask = foregrd.apply(image)
    fg_filter = cv.medianBlur(fg_mask, 5)
    fg_filter = cv.dilate(fg_filter, kernel, iterations=1)
#    fg_filter = cv.erode(fg_filter, kernel, iterations=1)

    return fg_filter

