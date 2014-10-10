# PROGRAM: read_cifar10.py
# 
# This code is created to read CIFAR10 unzip images to pandas dataframe
# 
# HISTORY: 30 July 2014 Created by Titipat Achakulvisut
#
# NOTATION: use '_' as a variable name
#           use capital letter e.g. getImage() as a function
#
# INPUT : X = [x1;x2;...;xn] where x_i = vectize_row(all color pixels concat together)
# OUTPUT: y = [1;0;0;...;0] if category = 'cat
#         y = [0;1;0;...;0] if category = 'horse' ...
# (This format allows us to train NN with DropConnect)

import numpy as np # numpy
import os, sys # get directory and others
import re # regular expression
import pandas as pd # pandas
import matplotlib.pyplot as plt # matplotlib library
import cv2 # opencv library

##### GET DIRECTORY  #####

curr_path = os.getcwd() # current directory
working_path = os.path.join(curr_path, '/../Kaggle/CIFAR10') # for Titipat
# working_path = curr_path + '...' # for Zaw
# print working_path

test_path = os.path.join(working_path, '/test')
train_path = os.path.join(working_path, '/train')
full_train_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(train_path)) for f in fn]
full_test_path = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(test_path)) for f in fn]

n_train = np.shape(full_train_path)[0] # I need to delete .DS_... which add 1 excess element
n_test = np.shape(full_test_path)[0]

##### READ LABELS CSV file #####

train_labels = pd.read_csv(working_path + '/trainLabels.csv', index_col = False, header = False) # get data frame from csv file
train_labels_unique = train_labels['label'].unique() # get unique categories
m = np.shape(train_labels_unique)[0] # number of unique categories (equal 10 in CIFAR10)
Y = np.eye(m,m).astype(np.int) # create prototype output Y = eye(m) first
print train_labels_unique


##### READ IMAGE AND STORE IN ARRAY #####




##### FUNCTIONS #####



