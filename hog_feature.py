# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:13:09 2018

@author: LENOVO
"""

from __future__ import division
from skimage._shared.utils import assert_nD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def gradien(image):
    g_row = np.empty(image.shape, dtype=np.double)
    g_col = np.empty(image.shape, dtype=np.double)
    
    g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    
    g_row[1:-1, :] = image[2:, :] - image[:-2, :]
    g_row[0, :] = 0
    g_row[-1, :] = 0
    
    return g_row, g_col

def visualize_hist(nbins, c_row, c_col, s_row, s_col, n_cellsr, n_cellsc, orientation_hist):
    from skimage import draw
    
    rad = min(c_row, c_col) // 2 - 1
    orient_arr = np.arange(nbins)
    orient_bin_midpts = (np.pi * (orient_arr + .5) / nbins)
    dr_arr = rad * np.sin(orient_bin_midpts)
    dc_arr = rad * np.cos(orient_bin_midpts)
    hog_image = np.zeros((s_row, s_col), dtype=float)
    
    
    for r in range(n_cellsr):
        for c in range(n_cellsc):
            for o, dr, dc in zip(orient_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2, 
                                c * c_col + c_col // 2])
                rr, cc = draw.line(int(centre[0] - dc), 
                                   int(centre[1] + dr), 
                                   int(centre[0] + dc), 
                                   int(centre[1] - dr))

                hog_image[rr, cc] += orientation_hist[r, c, o]
                
    return hog_image

def normalise_hist(block):
    eps = 1e-5
    
    out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    out = np.minimum(out, 0.2)
    out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
            
    return out

def cell_hog(magnitude, orientation, orient_start, orient_end, c_col, c_row, c, r, s_col, s_row, range_startr, range_stopr, range_startc, range_stopc):
    total = 0.
    for c_r in range(int(range_startr), int(range_stopr)):
        c_row_idx = int(r + c_r)
        if c_row_idx<0 or c_row_idx>=s_row:
            continue
        
        for c_c in range(int(range_startc), int(range_stopc)):
            c_col_idx = int(c + c_c)
            if c_col_idx<0 or c_col_idx>=s_col or orientation[c_row_idx, c_col_idx]>=orient_start or orientation[c_row_idx, c_col_idx]<orient_end:
                continue
            total += magnitude[c_row_idx, c_col_idx]
     
    return total / (c_row * c_col)

def hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=True, normalise=True, feature_vector=False):
    image = np.atleast_2d(image)
    assert_nD(image, 2)
    if image.dtype.kind == 'u':
        image = image.astype('float')
    
    #compute first order gradients
    g_row, g_col = gradien(image)
    
    #compute gradient histograms
    s_row, s_col = image.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block
    
    n_cellsr = int(s_row // c_row)
    n_cellsc = int(s_col // c_col)
    
    orientation_hist = np.zeros((n_cellsr, n_cellsc, nbins))

    magnitude = np.hypot(g_col, g_row)
    orientation = np.rad2deg(np.arctan2(g_row, g_col)) % 180
    
    r_0 = c_row / 2
    c_0 = c_col / 2
    cc = c_row * n_cellsr
    cr = c_col * n_cellsc
    range_stopr = c_row / 2
    range_startr = -range_stopr
    range_stopc = c_col / 2
    range_startc = -range_stopc
    for i in range(nbins):
        orient_start = (float(180.0 / nbins)) * (i+1)
        orient_end = (float(180.0 / nbins)) * i
        c_idx = c_0
        r_idx = r_0
        r_i = 0
        c_i = 0
        
        while r_idx < cc:
            c_i = 0
            c_idx = c_0
            
            while c_idx < cr:
                orientation_hist[r_i, c_i, i] = cell_hog(magnitude, orientation, orient_start, orient_end, c_col, c_row, c_idx, r_idx, s_col, s_row, range_startr, range_stopr, range_startc, range_stopc)
                c_i += 1
                c_idx += c_col
                
            r_i += 1
            r_idx += c_row
    
    #visualize if needed
    hog_image = None
    if visualize:
        hog_image = visualize_hist(nbins, c_row, c_col, s_row, s_col, n_cellsr, n_cellsc, orientation_hist)

        
    #normalise histogram per block
    if normalise:    
        n_blockr = (n_cellsr - b_row) + 1
        n_blockc = (n_cellsc - b_col) + 1
        normalised_block = np.zeros((n_blockr, n_blockc, b_row, b_col ,nbins))
        
        for r in range(n_blockr):
            for c in range(n_blockc):
                block = orientation_hist[r:r+b_row, c:c+b_col, :]
                normalised_block[r, c, :] = normalise_hist(block)
                
    #create feature vector
    if feature_vector:
        normalised_block = normalised_block.ravel()
        
    if visualize:
        return normalised_block, hog_image
    else:
        return normalised_block
    
    
if __name__ == '__main__':
    
    image = mpimg.imread('vehicle_0002124.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    features, im = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=True, normalise=True, feature_vector=True)
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(im, cmap=plt.cm.Greys_r)
    plt.title('HOG Visualization')
    plt.show()
    
    image2 = mpimg.imread('vehicle_0002125.jpg')
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    features2, im2 = hog(gray2, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=True, normalise=True, feature_vector=True)
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image2)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(im2, cmap=plt.cm.Greys_r)
    plt.title('HOG Visualization')
    plt.show()