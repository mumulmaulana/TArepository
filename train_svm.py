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
#import shutil
import random
import matplotlib.image as mpimg
import cv2
import pickle

path = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\source'
path_hog = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\hog1200'
path_lbp = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\lbp1200'
path_select = 'E:\\TA\\Vehicle Detection\\vehicle_classification\\img_dataset\\selected1200'
hog_feat = []
lbp_feat = []
label = []

run_path = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(path_select, topdown=False):
    for name in dirs:
        print "now in folder "+name
        img_dir = os.path.join(root, name)
        img_hog = os.path.join(path_hog, name)
        img_lbp = os.path.join(path_lbp, name)
#        img_select = os.path.join(path_select, name)
        
        os.chdir(img_dir)
        for x in range(0, 1260):
            file = random.choice(os.listdir(img_dir))
            if file.endswith(".jpg"):
                print "extracting feature of "+file+"..."
                image = mpimg.imread(file)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                hog_hist, hog_img = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9, visualize=True, normalise=True, feature_vector=True)
                hog_feat.append(hog_hist)
                cv2.imwrite(os.path.join(img_hog, file), hog_img)
                
                lbp_hist, lbp_img = lbp(gray, visualize=True)
                lbp_feat.append(lbp_hist)
                cv2.imwrite(os.path.join(img_lbp, file), lbp_img)
                
                label.append(name)
                
#                srcpath = os.path.join(img_dir, file)
#                destpath = os.path.join(img_select, file)
#                shutil.move(srcpath, destpath)
                
os.chdir(run_path)
hog_lbp_feat = np.hstack([hog_feat, lbp_feat])

print "fitting SVM model for HOG, LBP, and HOG+LBP feature..."
hog_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
lbp_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_lbp_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_train, hog_test, label_train1, label_test1 = train_test_split(hog_feat, label, test_size=0.05)
lbp_train, lbp_test, label_train2, label_test2 = train_test_split(lbp_feat, label, test_size=0.05)
hog_lbp_train, hog_lbp_test, label_train3, label_test3 = train_test_split(hog_lbp_feat, label, test_size=0.05)
hog_model.fit(hog_train, label_train1)
lbp_model.fit(lbp_train, label_train2)
hog_lbp_model.fit(hog_lbp_train, label_train3)
print('Test Accuracy of HOG = ', round(hog_model.score(hog_test, label_test1), 4))
print('Test Accuracy of LBP = ', round(lbp_model.score(lbp_test, label_test2), 4))
print('Test Accuracy of HOG+LBP = ', round(hog_lbp_model.score(hog_lbp_test, label_test3), 4))

print "save SVM models to directory..."
#with open('svm_classifiers/hog_model_1200.p', 'wb') as f:
#	pickle.dump(hog_model, f)
with open('svm_classifiers/lbp_3x3_model_1200.p', 'wb') as f:
	pickle.dump(lbp_model, f)
with open('svm_classifiers/hog_lbp_3x3_model_1200.p', 'wb') as f:
	pickle.dump(hog_lbp_model, f)

print "create standardized version for features..."
hog_scaler = StandardScaler()
lbp_scaler = StandardScaler()
hog_lbp_scaler = StandardScaler()
hog_scaler.fit(hog_feat)
scaled_hog = hog_scaler.transform(hog_feat)
lbp_scaler.fit(lbp_feat)
scaled_lbp = lbp_scaler.transform(lbp_feat)
hog_lbp_scaler.fit(hog_lbp_feat)
scaled_hog_lbp = hog_lbp_scaler.transform(hog_lbp_feat)

print "fitting SVM model for HOG, LBP, and HOG+LBP standardized feature..."
hog_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
lbp_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_lbp_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=1.0)
hog_train, hog_test, label_train1, label_test1 = train_test_split(scaled_hog, label, test_size=0.05)
lbp_train, lbp_test, label_train2, label_test2 = train_test_split(scaled_lbp, label, test_size=0.05)
hog_lbp_train, hog_lbp_test, label_train3, label_test3 = train_test_split(scaled_hog_lbp, label, test_size=0.05)
hog_model.fit(hog_train, label_train1)
lbp_model.fit(lbp_train, label_train2)
hog_lbp_model.fit(hog_lbp_train, label_train3)
print('Test Accuracy of scaled HOG = ', round(hog_model.score(hog_test, label_test1), 4))
print('Test Accuracy of scaled LBP = ', round(lbp_model.score(lbp_test, label_test2), 4))
print('Test Accuracy of scaled HOG+LBP = ', round(hog_lbp_model.score(hog_lbp_test, label_test3), 4))

print "save SVM models to directory..."
#with open('svm_classifiers/scaled_hog_model_1200.p', 'wb') as f:
#	pickle.dump(hog_model, f)
with open('svm_classifiers/scaled_lbp_3x3_model_1200.p', 'wb') as f:
	pickle.dump(lbp_model, f)
with open('svm_classifiers/scaled_hog_lbp_3x3_model_1200.p', 'wb') as f:
	pickle.dump(hog_lbp_model, f)
#with open('svm_classifiers/hog_scaler_1200.p', 'wb') as f:
#	pickle.dump(hog_scaler, f)
with open('svm_classifiers/lbp_3x3_scaler_1200.p', 'wb') as f:
	pickle.dump(lbp_scaler, f)
with open('svm_classifiers/hog_lbp_3x3_scaler_1200.p', 'wb') as f:
	pickle.dump(hog_lbp_scaler, f)
    
print "done."
