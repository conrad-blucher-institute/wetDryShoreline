# Importing packages
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Dropout, Concatenate, Flatten, MaxPooling2D, Conv2DTranspose, UpSampling2D
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os.path
import pywt
import pywt.data
from sklearn.linear_model import LogisticRegression
from PIL import Image


def dataloader(root_dir):
    
    OriginalImgs = sorted(glob.glob(os.path.join(root_dir, 'original/*.*')))
    LabeledImgs = sorted(glob.glob(os.path.join(root_dir, 'labeled/*.*'))) 
    
    # Creating arrays for training and testing images
    OriginalImg = []
    LabeledImg = []
    
    # Reading training and testing original images
    for file in OriginalImgs:
        readOriginalImg = cv2.imread(file)
        
        # Applying clahe to the original images
        claheHE = clahe(readOriginalImg)
        OriginalImg.append(np.asarray(claheHE)) 
        
        
    # Reading training and testing labeled images
    for file in LabeledImgs:
        readLabeledImg = cv2.imread(file)
        
        # Applying the binarization function the the labeled images
        readLabeledImg = binarization(readLabeledImg, 125)
        LabeledImg.append(np.asarray(readLabeledImg))   

    # Converting to float32
    arrayOriginalImg    = np.asarray(OriginalImg, dtype=np.float32)  
    arrayLabeledImg     = np.asarray(LabeledImg, dtype=np.float32)

    #Normalizing the RGB codes by dividing it to the max RGB value.
    arrayOriginalImg =arrayOriginalImg / 255.0

    return arrayOriginalImg, arrayLabeledImg


# CLAHE (Contrastive Limited Adaptative Histogram Equalitzation)
# We apply clahe in order to equalize the images from different
# locations while at the same time increasing the image contrast
def clahe(rgb_img):
    image_bw = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit = 0.01)
    img = clahe.apply(image_bw) + 30   
    
    final_img = np.expand_dims(img, axis = -1)    

    return final_img


# Converting the labeledd images to binary
def binarization(input_image, threshold):
    
    src = input_image[:,:, 0]
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] > threshold:
                src[i][j] = 1
            else:
                src[i][j] = 0
    src = np.expand_dims(src, axis = -1)       
    
    return src
