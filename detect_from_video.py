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

def slide_window(img, img_width, img_height, wh_window=(64, 64), wh_overlap=(0.5, 0.5)):
      
    wspan = img_width
    hspan = img_height
    nw_pix_per_step = np.int(wh_window[0]*(1 - wh_overlap[0]))
    nh_pix_per_step = np.int(wh_window[1]*(1 - wh_overlap[1]))
    nw_windows = np.int(wspan/nw_pix_per_step) - 1
    nh_windows = np.int(hspan/nh_pix_per_step) - 1
    
    window_list = []
    for hs in range(nh_windows):
        for ws in range(nw_windows):
            startw = ws*nw_pix_per_step
            endw = startw + wh_window[0]
            starth = hs*nh_pix_per_step
            endh = starth + wh_window[1]

            window_list.append([startw, starth, endw-startw, endh-starth])
            
    return window_list

path_vidplayback = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\output_frames\\videoplayback_single'
path_alibisurveillance = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\output_frames\\alibisvln_single'

runtime_start = time.time()

foregrd = init_fg()

cap = cv.VideoCapture('vid_dataset/videoplayback.mp4')
#cap = cv.VideoCapture('vid_dataset/AlibiSurveillance360p.mp4')
#out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), 25.0, (640, 360))

classifier_64 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
classifier_32 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
with open('svm_classifiers_new/scaled_hog_lbp_3x3_model_64.p', 'rb') as f:
    classifier_64 = pickle.load(f)
with open('svm_classifiers_new/scaled_hog_lbp_3x3_model_32.p', 'rb') as f:
    classifier_32 = pickle.load(f)
scaler_64 = StandardScaler()
scaler_32 = StandardScaler()
with open('svm_classifiers_new/hog_lbp_3x3_scaler_64.p', 'rb') as f:
    scaler_64 = pickle.load(f)
with open('svm_classifiers_new/hog_lbp_3x3_scaler_32.p', 'rb') as f:
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
 
        fg_mask, fg_filter, foreground = fg_detect(roi, foregrd)
        
        obj_coord = []
        obj_predict = []
        _,contours,hierarchy = cv.findContours(foreground, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE, offset=(roi_x, roi_y))
            
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
                
            if (w >= 32 and w <= 96) or (h >= 32 and h <= 96):
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
                print "hog detect : "+str(round(t2-t, 4))+"s"
                    
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
                print "lbp detect : "+str(round(t2-t, 4))+"s"
                    
                hog_lbp_hist = np.hstack([hog_hist, lbp_hist])
    
                if h == w:
                    scaled_hist = scaler_64.transform(np.array(hog_lbp_hist).reshape(1, -1))
                    predict = classifier_64.predict(scaled_hist)
                    
#                    savepath = os.path.join(path_alibisurveillance, predict[0])
#                    filename = "file_%d.jpg"%d
#                    cv.imwrite(os.path.join(savepath, filename), obj)
                    
#                    if predict[0] == 'Other_n':
#                        obj_width = obj.shape[1]
#                        obj_height = obj.shape[0]
#                        vert_windows = slide_window(obj, obj_width, obj_height, wh_window=(32, 64), wh_overlap=(0.5, 0.5))
#                        horz_windows = slide_window(obj, obj_width, obj_height, wh_window=(32, 32), wh_overlap=(0.5, 0.5))
#                        
#                        motor_detect = False
#                        for verts in vert_windows:
#                            vx, vy, vw, vh = verts
#                            obj_vert = obj[vy:vy+vh, vx:vx+vw]
#                            obj_vert = cv.resize(obj_vert, (32, 64), interpolation = cv.INTER_LINEAR)
#                            
#                            t=time.time()
#                            hog_hist = hog(obj_vert, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=False, normalise=True, feature_vector=True)
#                            t2=time.time()
#                            print "hog detect : "+str(round(t2-t, 4))+"s"
#                            t=time.time()
#                            lbp_hist = lbp(obj_vert, visualize=False)
#                            t2=time.time()
#                            print "lbp detect : "+str(round(t2-t, 4))+"s"
#                            
#                            hog_lbp_hist = np.hstack([hog_hist, lbp_hist])
#                            scaled_hist = scaler_32.transform(np.array(hog_lbp_hist).reshape(1, -1))
#                            predict = classifier_32.predict(scaled_hist)
#                            
#                            if predict[0] == 'Motorcycle':
#                                scale = float(h)/64
#                                obj_coord.append([x+int(vx*scale), y+int(vy*scale), int(vw*scale), int(vh*scale)])
#                                obj_predict.append(predict[0])
#                                motor_detect = True
#                    
#                        if motor_detect == False:
#                            for horzs in horz_windows:
#                                hx, hy, hw, hh = horzs
#                                obj_horz = obj[hy:hy+hh, hx:hx+hw]
#                                obj_horz = cv.resize(obj_horz, (64, 64), interpolation = cv.INTER_LINEAR)
#                                
#                                t=time.time()
#                                hog_hist = hog(obj_horz, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=False, normalise=True, feature_vector=True)
#                                t2=time.time()
#                                print "hog detect : "+str(round(t2-t, 4))+"s"
#                                t=time.time()
#                                lbp_hist = lbp(obj_horz, visualize=False)
#                                t2=time.time()
#                                print "lbp detect : "+str(round(t2-t, 4))+"s"
#                                
#                                hog_lbp_hist = np.hstack([hog_hist, lbp_hist])
#                                scaled_hist = scaler_64.transform(np.array(hog_lbp_hist).reshape(1, -1))
#                                predict = classifier_64.predict(scaled_hist)
#                                
#                                if predict[0] == 'Car' or predict[0] == 'Minivan' or predict[0] == 'Bus_Truck':
#                                    scale = float(w)/64
#                                    obj_coord.append([x+int(hx*scale), y+int(hy*scale), int(hw*scale), int(hh*scale)])
#                                    obj_predict.append(predict[0])
#                    else:
#                        obj_coord.append([x, y, w, h])
#                        obj_predict.append(predict[0])
                    obj_coord.append([x, y, w, h])
                    obj_predict.append(predict[0])
                    
                else:
                    scaled_hist = scaler_32.transform(np.array(hog_lbp_hist).reshape(1, -1))
                    predict = classifier_32.predict(scaled_hist)
                    
#                    savepath = os.path.join(path_alibisurveillance, predict[0])
#                    filename = "file_%d.jpg"%d
#                    cv.imwrite(os.path.join(savepath, filename), obj)
                    
#                    if predict[0] == 'Other_m':
#                        obj_width = obj.shape[1]
#                        obj_height = obj.shape[0]
#                        horz_windows = slide_window(obj, obj_width, obj_height, wh_window=(32, 32), wh_overlap=(0.5, 0.5))
#                        for horzs in horz_windows:
#                            hx, hy, hw, hh = horzs
#                            obj_horz = obj[hy:hy+hh, hx:hx+hw]
#                            obj_horz = cv.resize(obj_horz, (64, 64), interpolation = cv.INTER_LINEAR)
#                            
#                            t=time.time()
#                            hog_hist = hog(obj_horz, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=False, normalise=True, feature_vector=True)
#                            t2=time.time()
#                            print "hog detect : "+str(round(t2-t, 4))+"s"
#                            t=time.time()
#                            lbp_hist = lbp(obj_horz, visualize=False)
#                            t2=time.time()
#                            print "lbp detect : "+str(round(t2-t, 4))+"s"
#                                
#                            hog_lbp_hist = np.hstack([hog_hist, lbp_hist])
#                            scaled_hist = scaler_64.transform(np.array(hog_lbp_hist).reshape(1, -1))
#                            predict = classifier_64.predict(scaled_hist)
#                            
#                            if predict[0] == 'Car' or predict[0] == 'Minivan' or predict[0] == 'Bus_Truck':
#                                scale = float(h)/64
#                                obj_coord.append([x+int(hx*scale), y+int(hy*scale), int(hw*scale), int(hh*scale)])
#                                obj_predict.append(predict[0])
#                        
#                    else:
#                        obj_coord.append([x, y, w, h])
#                        obj_predict.append(predict[0])
                    
                    obj_coord.append([x, y, w, h])
                    obj_predict.append(predict[0])
            else:
                continue
                
#            cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv.rectangle(image, (roi_x, roi_y), (roi_x_w, roi_y_h), (0,0,255), 2)
        cv.putText(image, 'Green: Minivan', (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, 'Blue: Car', (130, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(image, 'Yellow: Bus_Truck', (230, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, 'Light Blue: Motorcycle', (400, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (204, 188, 100), 2, cv.LINE_AA)
        for i,obj in enumerate(obj_coord):
            obj_x, obj_y, obj_w, obj_h = obj
            if obj_predict[i] == 'Minivan':
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(0,255,0),3)
            elif obj_predict[i] == 'Car':    
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(255,0,0),3)
            elif obj_predict[i] == 'Bus_Truck':
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(0,255,255),3)
            elif obj_predict[i] == 'Motorcycle':
                bbox = cv.rectangle(image,(obj_x,obj_y),(obj_x+obj_w,obj_y+obj_h),(204,188,100),3)
        
        savepath = path_vidplayback
        filename = "file_%d.jpg"%d
        cv.imwrite(os.path.join(savepath, filename), image)
        d += 1
        
#        out.write(image)
        
        cv.namedWindow('mask', cv.WINDOW_NORMAL)
        cv.resizeWindow('mask', roi_x_w-roi_x, roi_y_h-roi_y)
        cv.namedWindow('filter', cv.WINDOW_NORMAL)
        cv.resizeWindow('filter', roi_x_w-roi_x, roi_y_h-roi_y)
        cv.namedWindow('closing', cv.WINDOW_NORMAL)
        cv.resizeWindow('closing', roi_x_w-roi_x, roi_y_h-roi_y)
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 640, 360)
        
        cv.imshow('mask', fg_mask)
        cv.imshow('filter', fg_filter)
        cv.imshow('closing', foreground)
        cv.imshow('frame', image)
        
        cap_spacer += 1
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        
    else:
        break
		
cap.release()
#out.release()
cv.destroyAllWindows()

runtime_stop = time.time()
print "runtime : "+str(round(runtime_stop-runtime_start, 4))+"s"
