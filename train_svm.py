# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:43:40 2018

@author: LENOVO
"""

from hog_feature import hog
from lbp_3x3_feature import lbp
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import shutil
import random
import matplotlib.image as mpimg
import cv2
import pickle

path = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\source'
path_hog = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\hog_vidplayback'
path_lbp = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\lbp_vidplayback'
path_select = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\selected_vidplayback'
path_vidplayback = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\videoplayback'
hog_feat_64 = []
lbp_feat_64 = []
label_64 = []
hog_feat_32 = []
lbp_feat_32 = []
label_32 = []

run_path = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(path_vidplayback, topdown=False):
    for name in dirs:
        print "now in folder "+name
        img_dir = os.path.join(root, name)
        img_hog = os.path.join(path_hog, name)
        img_lbp = os.path.join(path_lbp, name)
        img_select = os.path.join(path_select, name)
        
        os.chdir(img_dir)
        if name == 'Motorcycle' or name == 'Other_m':
            for file in os.listdir(img_dir):
                if file.endswith(".jpg"):
                    print "extracting feature of "+file+"..."
                    gray = mpimg.imread(file)
                    if len(gray.shape) > 2:
                        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
                    
                    hog_hist, hog_img = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=True, normalise=True, feature_vector=True)
                    hog_feat_32.append(hog_hist)
                    cv2.imwrite(os.path.join(img_hog, file), hog_img)
                    
                    lbp_hist, lbp_img = lbp(gray, visualize=True)
                    lbp_feat_32.append(lbp_hist)
                    cv2.imwrite(os.path.join(img_lbp, file), lbp_img)
                    
                    label_32.append(name)
                    
                    srcpath = os.path.join(img_dir, file)
                    destpath = os.path.join(img_select, file)
                    shutil.move(srcpath, destpath)                    
        else:
            for x in range(0, 1260):
                file = random.choice(os.listdir(img_dir))
                if file.endswith(".jpg"):
                    print "extracting feature of "+file+"..."
                    gray = mpimg.imread(file)
                    if len(gray.shape) > 2:
                        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
                    
                    hog_hist, hog_img = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=True, normalise=True, feature_vector=True)
                    hog_feat_64.append(hog_hist)
                    cv2.imwrite(os.path.join(img_hog, file), hog_img)
                    
                    lbp_hist, lbp_img = lbp(gray, visualize=True)
                    lbp_feat_64.append(lbp_hist)
                    cv2.imwrite(os.path.join(img_lbp, file), lbp_img)
                    
                    label_64.append(name)
                    
                    srcpath = os.path.join(img_dir, file)
                    destpath = os.path.join(img_select, file)
                    shutil.move(srcpath, destpath)
                
os.chdir(run_path)
hog_lbp_feat_64 = np.hstack([hog_feat_64, lbp_feat_64])
hog_lbp_feat_32 = np.hstack([hog_feat_32, lbp_feat_32])

#print "fitting SVM model for HOG, LBP, and HOG+LBP feature..."
#hog_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
#lbp_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
#hog_lbp_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
#hog_train, hog_test, label_train1, label_test1 = train_test_split(hog_feat, label, test_size=0.05)
#lbp_train, lbp_test, label_train2, label_test2 = train_test_split(lbp_feat, label, test_size=0.05)
#hog_lbp_train, hog_lbp_test, label_train3, label_test3 = train_test_split(hog_lbp_feat, label, test_size=0.05)
#hog_model.fit(hog_train, label_train1)
#lbp_model.fit(lbp_train, label_train2)
#hog_lbp_model.fit(hog_lbp_train, label_train3)
#print('Test Accuracy of HOG = ', round(hog_model.score(hog_test, label_test1), 4))
#print('Test Accuracy of LBP = ', round(lbp_model.score(lbp_test, label_test2), 4))
#print('Test Accuracy of HOG+LBP = ', round(hog_lbp_model.score(hog_lbp_test, label_test3), 4))
#
#print "save SVM models to directory..."
#with open('svm_classifiers/hog_model_1200.p', 'wb') as f:
#	pickle.dump(hog_model, f)
#with open('svm_classifiers/lbp_3x3_model_1200.p', 'wb') as f:
#	pickle.dump(lbp_model, f)
#with open('svm_classifiers/hog_lbp_3x3_model_1200.p', 'wb') as f:
#	pickle.dump(hog_lbp_model, f)

print "create standardized version for features..."
hog_scaler_64 = StandardScaler()
lbp_scaler_64 = StandardScaler()
hog_lbp_scaler_64 = StandardScaler()
hog_scaler_32 = StandardScaler()
lbp_scaler_32 = StandardScaler()
hog_lbp_scaler_32 = StandardScaler()
hog_scaler_64.fit(hog_feat_64)
scaled_hog_64 = hog_scaler_64.transform(hog_feat_64)
lbp_scaler_64.fit(lbp_feat_64)
scaled_lbp_64 = lbp_scaler_64.transform(lbp_feat_64)
hog_lbp_scaler_64.fit(hog_lbp_feat_64)
scaled_hog_lbp_64 = hog_lbp_scaler_64.transform(hog_lbp_feat_64)
hog_scaler_32.fit(hog_feat_32)
scaled_hog_32 = hog_scaler_32.transform(hog_feat_32)
lbp_scaler_32.fit(lbp_feat_32)
scaled_lbp_32 = lbp_scaler_32.transform(lbp_feat_32)
hog_lbp_scaler_32.fit(hog_lbp_feat_32)
scaled_hog_lbp_32 = hog_lbp_scaler_32.transform(hog_lbp_feat_32)

print "fitting SVM model for HOG, LBP, and HOG+LBP standardized feature..."
hog_model_64 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
lbp_model_64 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_lbp_model_64 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_model_32 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
lbp_model_32 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_lbp_model_32 = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_train_64, hog_test_64, label_train1, label_test1 = train_test_split(scaled_hog_64, label_64, test_size=0.05)
lbp_train_64, lbp_test_64, label_train2, label_test2 = train_test_split(scaled_lbp_64, label_64, test_size=0.05)
hog_lbp_train_64, hog_lbp_test_64, label_train3, label_test3 = train_test_split(scaled_hog_lbp_64, label_64, test_size=0.05)
hog_train_32, hog_test_32, label_train4, label_test4 = train_test_split(scaled_hog_32, label_32, test_size=0.05)
lbp_train_32, lbp_test_32, label_train5, label_test5 = train_test_split(scaled_lbp_32, label_32, test_size=0.05)
hog_lbp_train_32, hog_lbp_test_32, label_train6, label_test6 = train_test_split(scaled_hog_lbp_32, label_32, test_size=0.05)
hog_model_64.fit(hog_train_64, label_train1)
lbp_model_64.fit(lbp_train_64, label_train2)
hog_lbp_model_64.fit(hog_lbp_train_64, label_train3)
hog_model_32.fit(hog_train_32, label_train4)
lbp_model_32.fit(lbp_train_32, label_train5)
hog_lbp_model_32.fit(hog_lbp_train_32, label_train6)
print "Test Accuracy of scaled HOG for 64x64 images = " + str(round(hog_model_64.score(hog_test_64, label_test1), 4))
print "Test Accuracy of scaled LBP for 64x64 images = " + str(round(lbp_model_64.score(lbp_test_64, label_test2), 4))
print "Test Accuracy of scaled HOG+LBP for 64x64 images = " + str(round(hog_lbp_model_64.score(hog_lbp_test_64, label_test3), 4))
print "Test Accuracy of scaled HOG for 64x32 images = " + str(round(hog_model_32.score(hog_test_32, label_test4), 4))
print "Test Accuracy of scaled LBP for 64x32 images = " + str(round(lbp_model_32.score(lbp_test_32, label_test5), 4))
print "Test Accuracy of scaled HOG+LBP for 64x32 images = " + str(round(hog_lbp_model_32.score(hog_lbp_test_32, label_test6), 4))

print "save SVM models to directory..."
with open('svm_classifiers/scaled_hog_model_64.p', 'wb') as f:
	pickle.dump(hog_model_64, f)
with open('svm_classifiers/scaled_lbp_3x3_model_64.p', 'wb') as f:
	pickle.dump(lbp_model_64, f)
with open('svm_classifiers/scaled_hog_lbp_3x3_model_64.p', 'wb') as f:
	pickle.dump(hog_lbp_model_64, f)
with open('svm_classifiers/hog_scaler_64.p', 'wb') as f:
	pickle.dump(hog_scaler_64, f)
with open('svm_classifiers/lbp_3x3_scaler_64.p', 'wb') as f:
	pickle.dump(lbp_scaler_64, f)
with open('svm_classifiers/hog_lbp_3x3_scaler_64.p', 'wb') as f:
	pickle.dump(hog_lbp_scaler_64, f)
with open('svm_classifiers/scaled_hog_model_32.p', 'wb') as f:
	pickle.dump(hog_model_32, f)
with open('svm_classifiers/scaled_lbp_3x3_model_32.p', 'wb') as f:
	pickle.dump(lbp_model_32, f)
with open('svm_classifiers/scaled_hog_lbp_3x3_model_32.p', 'wb') as f:
	pickle.dump(hog_lbp_model_32, f)
with open('svm_classifiers/hog_scaler_32.p', 'wb') as f:
	pickle.dump(hog_scaler_32, f)
with open('svm_classifiers/lbp_3x3_scaler_32.p', 'wb') as f:
	pickle.dump(lbp_scaler_32, f)
with open('svm_classifiers/hog_lbp_3x3_scaler_32.p', 'wb') as f:
	pickle.dump(hog_lbp_scaler_32, f)
    
print "done."
