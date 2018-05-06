# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:52:51 2018

@author: LENOVO
"""

import cv2 as cv
import numpy as np
import pickle
import time
from foreground_detect import init_fg, fg_detect
from hog_feature import hog
from lbp_3x3_feature import lbp
from sklearn import svm
from sklearn.preprocessing import StandardScaler
#from skimage.feature import hog, local_binary_pattern

foregrd = init_fg()

cap = cv.VideoCapture('vid_dataset/M6Motorway360p.mp4')
#cap = cv.VideoCapture('vid_dataset/videoplayback.mp4')
#out = cv.VideoWriter('output1.avi', cv.VideoWriter_fourcc(*'MJPG'), 25.0, (640, 360))
classifier = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
with open('svm_classifiers/scaled_hog_lbp_3x3_model_1200.p', 'rb') as f:
    classifier = pickle.load(f)
scaler = StandardScaler()
with open('svm_classifiers/hog_lbp_3x3_scaler_1200.p', 'rb') as f:
    scaler = pickle.load(f)
#cap_spacer = 1

while(cap.isOpened()):
    ret, image = cap.read()
    
    if ret == True:
        
        roi_x = 340
        roi_y = 150
        roi_x_w = 640
        roi_y_h = 360
        cv.rectangle(image, (roi_x, roi_y), (roi_x_w, roi_y_h), (0,0,255), 2)
        roi = image[roi_y:roi_y_h, roi_x:roi_x_w]
 
        foreground = fg_detect(roi, foregrd)
        
#        if cap_spacer%5 == 0:
        _,contours,hierarchy = cv.findContours(foreground, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE, offset=(roi_x, roi_y))
            
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
                
            if w >= 64 or h >= 64:
                if w >= h:
                    h = w
                    if x+w > roi_x_w:
                        temp_w = (x+w) - roi_x_w
                        x = x - temp_w
                    if y+h > roi_y_h:
                        temp_h = (y+h) - roi_y_h
                        y = y - temp_h
                    obj = image[y:y+h, x:x+w]
                    obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                    if w > 64:
                        obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_AREA)
#                    else:
#                        obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_LINEAR)
                else:
                    w = h
                    if x+w > roi_x_w:
                        temp_w = (x+w) - roi_x_w
                        x = x - temp_w
                    if y+h > roi_y_h:
                        temp_h = (y+h) - roi_y_h
                        y = y - temp_h
                    obj = image[y:y+h, x:x+w]
                    obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                    if h > 64:
                        obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_AREA)
#                    else:
#                        obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_LINEAR)

                t=time.time()
                hog_hist = hog(obj, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=False, normalise=True, feature_vector=True)
#                hog_hist = hog(obj, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualise=False, block_norm='L2-Hys', feature_vector=True)
                t2=time.time()
                print "hog detect : "+str(round(t2-t, 2))+"s"
                
                t=time.time()
                lbp_hist = lbp(obj, visualize=False)
#                lbp_hist = local_binary_pattern(obj, 8, 1, method='default')
#                hist, bins = np.histogram(lbp_hist.flatten(),256,[0,256]) 
#                eps = 1e-5
#                hist = hist.astype("float")
#                norm_hist = hist / np.sqrt(np.sum(hist ** 2) + eps ** 2)
#                norm_hist = np.minimum(norm_hist, 0.2)
#                norm_hist = norm_hist / np.sqrt(np.sum(norm_hist ** 2) + eps ** 2)
#                t2=time.time()
                print "lbp detect : "+str(round(t2-t, 2))+"s"
                
                hog_lbp_hist = np.hstack([hog_hist, lbp_hist])
                    
                scaled_hist = scaler.transform(np.array(hog_lbp_hist).reshape(1, -1))
                predict = classifier.predict(scaled_hist)
                    
                bbox = cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
                cv.putText(bbox, predict[0], (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                
            else:
#                obj = image[y:y+h, x:x+w]
#                obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                continue
                
#            cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
#        out.write(image)
        
        cv.namedWindow('mask', cv.WINDOW_NORMAL)
        cv.resizeWindow('mask', roi_x_w-roi_x, roi_y_h-roi_y)
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 640, 360)
        
        cv.imshow('mask', foreground)
        cv.imshow('frame', image)
        
#        cap_spacer += 1
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
    else:
        break
		
cap.release()
#out.release()
cv.destroyAllWindows()
