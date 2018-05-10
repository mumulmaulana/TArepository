# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:52:51 2018

@author: LENOVO
"""

import cv2 as cv
import numpy as np
import pickle
import time
import os
from foreground_detect import init_fg, fg_detect
from hog_feature import hog
from lbp_3x3_feature import lbp
from sklearn import svm
from sklearn.preprocessing import StandardScaler
#from skimage.feature import hog, local_binary_pattern
path_vidplayback = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\videoplayback'
path_m6motorway = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\M6Motorway'

foregrd = init_fg()

#cap = cv.VideoCapture('vid_dataset/M6Motorway360p.mp4')
cap = cv.VideoCapture('vid_dataset/videoplayback.mp4')
out = cv.VideoWriter('output2.avi', cv.VideoWriter_fourcc(*'MJPG'), 25.0, (640, 360))
classifier_64 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
classifier_32 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
with open('svm_classifiers/scaled_hog_lbp_3x3_model_64.p', 'rb') as f:
    classifier_64 = pickle.load(f)
with open('svm_classifiers/scaled_hog_lbp_3x3_model_32.p', 'rb') as f:
    classifier_32 = pickle.load(f)
scaler_64 = StandardScaler()
scaler_32 = StandardScaler()
with open('svm_classifiers/hog_lbp_3x3_scaler_64.p', 'rb') as f:
    scaler_64 = pickle.load(f)
with open('svm_classifiers/hog_lbp_3x3_scaler_32.p', 'rb') as f:
    scaler_32 = pickle.load(f)
cap_spacer = 1
d = 1

while(cap.isOpened()):
    ret, image = cap.read()
    
    if ret == True:
        print "cap "+str(cap_spacer)
        
        roi_x = 0
        roi_y = 160
        roi_x_w = 540
        roi_y_h = 360
        roi = image[roi_y:roi_y_h, roi_x:roi_x_w]
 
        foreground = fg_detect(roi, foregrd)
        
        obj_coord = []
        obj_predict = []
        _,contours,hierarchy = cv.findContours(foreground, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE, offset=(roi_x, roi_y))
            
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
                
            if (w >= 32 or h >= 32) and (w <= 96 or h <= 96):
                if w >= h:
                    if w*0.4 <= h:
                        y = y - (w-h)
                        h = w
                        if x+w > roi_x_w:
                            temp_w = (x+w) - roi_x_w
                            x = x - temp_w
            
                        obj = image[y:y+h, x:x+w]
                        obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                        if w > 64:
                            obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_AREA)
                        else:
                            obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_LINEAR)
                    else:
                        continue
                else:
                    if h%2 != 0:
                        h = h+1
                    if h*0.7 >= w:
                        if h/2 >= w:
                            if h*0.35 > w:
                                continue
                            else:
                                w = h/2
                        else:
                            h_new = w*2
                            y = y - (h_new-h)
                            h = h_new
                        if x+w > roi_x_w:
                            temp_w = (x+w) - roi_x_w
                            x = x - temp_w
                        
                        obj = image[y:y+h, x:x+w]
                        obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                        if h > 64:
                            obj = cv.resize(obj, (32, 64), interpolation = cv.INTER_AREA)
                        else:
                            obj = cv.resize(obj, (32, 64), interpolation = cv.INTER_LINEAR)
                    else:
                        w = h
                        if x+w > roi_x_w:
                            temp_w = (x+w) - roi_x_w
                            x = x - temp_w
            
                        obj = image[y:y+h, x:x+w]
                        obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                        if h > 64:
                            obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_AREA)
                        else:
                            obj = cv.resize(obj, (64, 64), interpolation = cv.INTER_LINEAR)
                            
                
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
                t2=time.time()
                print "lbp detect : "+str(round(t2-t, 2))+"s"
                    
                hog_lbp_hist = np.hstack([hog_hist, lbp_hist])
    
                if h == w:
                    scaled_hist = scaler_64.transform(np.array(hog_lbp_hist).reshape(1, -1))
                    predict = classifier_64.predict(scaled_hist)
                    
#                    savepath = os.path.join(path_vidplayback, predict[0])
#                    filename = "file_%d.jpg"%d
#                    cv.imwrite(os.path.join(savepath, filename), obj)
                    
                    obj_coord.append([x, y, w, h])
                    obj_predict.append(predict[0])
                    
                    d += 1
                else:
                    scaled_hist = scaler_32.transform(np.array(hog_lbp_hist).reshape(1, -1))
                    predict = classifier_32.predict(scaled_hist)
                    
#                    savepath = os.path.join(path_vidplayback, "Motorcycle")
#                    filename = "file_%d.jpg"%d
#                    cv.imwrite(os.path.join(savepath, filename), obj)
                    
                    obj_coord.append([x, y, w, h])
                    obj_predict.append(predict[0])
                    
                    d += 1
                    
            else:
#                obj = image[y:y+h, x:x+w]
#                obj = cv.cvtColor(obj, cv.COLOR_RGB2GRAY)
                continue
                
#            cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv.rectangle(image, (roi_x, roi_y), (roi_x_w, roi_y_h), (0,0,255), 2)
        for i,obj in enumerate(obj_coord):
            obj_x, obj_y, obj_w, obj_h = obj
            if obj_predict[i] == 'SUV_Minivan':
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(0,255,0),3)
                cv.putText(bbox, obj_predict[i], (obj_x,obj_y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            elif obj_predict[i] == 'Sedan':    
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(255,0,0),3)
                cv.putText(bbox, obj_predict[i], (obj_x,obj_y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            elif obj_predict[i] == 'Bus_Truck':
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(0,255,255),3)
                cv.putText(bbox, obj_predict[i], (obj_x,obj_y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
            elif obj_predict[i] == 'Motorcycle':
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(204,188,100),3)
                cv.putText(bbox, obj_predict[i], (obj_x,obj_y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (204,188,100), 1, cv.LINE_AA)
        
        out.write(image)
        
        cv.namedWindow('mask', cv.WINDOW_NORMAL)
        cv.resizeWindow('mask', roi_x_w-roi_x, roi_y_h-roi_y)
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 640, 360)
        
        cv.imshow('mask', foreground)
        cv.imshow('frame', image)
        
        cap_spacer += 1
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
    else:
        break
		
cap.release()
out.release()
cv.destroyAllWindows()
