# Importing packages
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, Dropout,Flatten, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers import Concatenate, Activation
import keras.backend as K
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os.path
import pywt
import pywt.data
from sklearn.linear_model import LogisticRegression
from keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers

# The modified hed architecture is defined in this class
class hed_model(): 
    def __init__(self, inputShape):
        self.inputShape = inputShape

    # This function is used to convert the results from the side branches
    # to the original image size
    def side_branch(self, x, factor):
        x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
        kernel_size = (2*factor, 2*factor)
        x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', 
                            use_bias=False, activation=None)(x)
        
        return x
    
    def _to_tensor(self, x, dtype):
        x = tf.convert_to_tensor(x, dtype=dtype)

        return x

    # Defining a custom cross entropy balanced loss function
    def cross_entropy_balanced(self, y_true, y_pred):  
        
        _epsilon = self._to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=_epsilon, clip_value_max=1 - _epsilon)  
        y_pred = tf.math.log(y_pred / (1 - y_pred))  

        y_true = tf.cast(y_true, tf.float32)

        neg = tf.reduce_sum(1 - y_true)  
        pos = tf.reduce_sum(y_true)  

        beta = neg / (neg + pos)
        pos_weight = beta / (1 - beta)
        
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
        loss = tf.reduce_mean(loss * (1 - beta))

        return tf.where(condition=tf.equal(loss, 0.0), x=0.0, y=loss)  

    # Converting the fuse output image to the original image size
    def fuse_pixel_error(self, y_true, y_pred):
        
        pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='prediction')
        error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
        return tf.reduce_mean(error, name='pixel_error')
    
   
    # Defining the model architecture
    def model(self):
        # Input                 
        img_input = Input(shape=self.inputShape, name='input')
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', dilation_rate=1, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block1_conv1')(img_input)   
        x = Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block1_conv2')(x)
        b1= self.side_branch(x, 1)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) 

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', dilation_rate=1, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block2_conv2')(x)
        b2= self.side_branch(x, 2) 
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) 

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', dilation_rate=1, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', dilation_rate=4, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block3_conv3')(x)
        b3= self.side_branch(x, 4)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) 

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', dilation_rate=1, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', dilation_rate=4, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block4_conv3')(x)
        b4= self.side_branch(x, 8) 
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) 

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', dilation_rate=1, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', dilation_rate=4, kernel_regularizer=regularizers.l2(0.0001), padding='same', name='block5_conv3')(x) # 30 30 512
        b5= self.side_branch(x, 16) 

        # Fuse
        fusion = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
        fusion = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fusion) 


        # Outputs
        o_s1 = Activation(activation='sigmoid', name='o_s1')(b1)
        o_s2 = Activation(activation='sigmoid', name='o_s2')(b2)
        o_s3 = Activation(activation='sigmoid', name='o_s3')(b3)
        o_s4 = Activation(activation='sigmoid', name='o_s4')(b4)
        o_s5 = Activation(activation='sigmoid', name='o_s5')(b5)
        o_fuse = Activation(activation='sigmoid', name='o_fuse')(fusion)
        
        # Defining the model
        model = Model(inputs=[img_input], outputs=[o_s1, o_s2, o_s3, o_s4, o_s5, o_fuse])
        
        # Compiling the model
        model.compile(loss={'o_s1': self.cross_entropy_balanced,
                    'o_s2': self.cross_entropy_balanced,
                    'o_s3': self.cross_entropy_balanced,
                    'o_s4': self.cross_entropy_balanced,
                    'o_s5': self.cross_entropy_balanced,
                    'o_fuse': self.cross_entropy_balanced,
                    },
              metrics={'o_fuse': self.fuse_pixel_error},
              optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        
        # Returning the model
        return model 

