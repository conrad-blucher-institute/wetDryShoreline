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
from keras.callbacks import EarlyStopping
import os
import pandas as pd 
import shutil

import datatools as utils 
from modifiedHED import hed_model


def training():

    filesNamePath = 'testingOrtho19/testing/labeled'
    files = sorted(os.listdir(filesNamePath))

    counter = 0
    
    for z in range(len(files)):
        if(".DS_Store" == files[z]):
            counter = 1
            
    if (counter == 1):
        files.remove(".DS_Store")
    

    filesWithoutExtension = []

    for k in range(len(files)):

        temp = files[k][:-4]
        filesWithoutExtension.append(temp)

    
    trainPath = 'testingOrtho19/training/'
    validationPath = 'testingOrtho19/validation/'
    testPath = 'testingOrtho19/testing/'
    
    
    arrayTrainingOriginalImg, arrayTrainingLabeledImg      = utils.dataloader(trainPath)
    arrayValidationOriginalImg, arrayValidationLabeledImg  = utils.dataloader(validationPath)
    arrayTestingOriginalImg, arrayTestingLabeledImg        = utils.dataloader(testPath) 
    
    print(f"Training Input Tensor Size: {arrayTrainingOriginalImg.shape}, Label Size: {arrayTrainingLabeledImg.shape}")
    print(f"Validation Input Tensor Size: {arrayValidationOriginalImg.shape}, Label Size: {arrayValidationLabeledImg.shape}")
    print(f"Test Input Tensor Size: {arrayTestingOriginalImg.shape}, Label Size: {arrayTestingLabeledImg.shape}")
    print("==================================================================================================================")
    
    
    inputShape = arrayTrainingOriginalImg[0].shape
    
    obj = hed_model(inputShape)
    model = obj.model() 
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=25, 
                                       verbose=1,
                                       mode='auto')
    
    
    history = model.fit(arrayTrainingOriginalImg, arrayTrainingLabeledImg, 
                        batch_size = 16, epochs=1000,
                        validation_data=(arrayValidationOriginalImg, arrayValidationLabeledImg), callbacks=[early_stopping])
    

    outputPath = '../results/dilatedRegularizedHED_testingOrtho19_'

    os.mkdir(outputPath)
    os.mkdir(outputPath + '/report')
    os.mkdir(outputPath + '/report/metrics')
    os.mkdir(outputPath + '/report/weights')
    os.mkdir(outputPath + '/report/lossGraph')
    os.mkdir(outputPath + '/imagesOutput')
    os.mkdir(outputPath + '/imagesOutput/labeled')
    os.mkdir(outputPath + '/imagesOutput/predicted')


    for m in range(len(arrayTestingLabeledImg)):

        os.mkdir(outputPath + '/imagesOutput/' + filesWithoutExtension[m])

        os.mkdir(outputPath + '/imagesOutput/' + filesWithoutExtension[m] + '/imageLabeled')
        os.mkdir(outputPath + '/imagesOutput/' + filesWithoutExtension[m] + '/imageOriginal')
        os.mkdir(outputPath + '/imagesOutput/' + filesWithoutExtension[m] + '/imagePredicted')

     
    fig1 = plt.gcf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig1.savefig(outputPath + '/report/lossGraph/lossHed.png')
    plt.show()  
    
    
    
    ### saving the model .h5
    h5Path = outputPath + '/report/weights/modelSaved.h5'
    model.save(h5Path)  
    
    print()
    print("Model saved succesfully")
    print()


    test = model.evaluate(arrayTestingOriginalImg, arrayTestingLabeledImg, batch_size = 16)
    pred = model.predict(arrayTestingOriginalImg) 


    for z in range(len(arrayTestingLabeledImg)):

        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[z] + '/imagePredicted/' + filesWithoutExtension[z] + '_sideOutput1.png', pred[0][z,:,:,:])
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[z] + '/imagePredicted/' + filesWithoutExtension[z] + '_sideOutput2.png', pred[1][z,:,:,:])
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[z] + '/imagePredicted/' + filesWithoutExtension[z] + '_sideOutput3.png', pred[2][z,:,:,:])
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[z] + '/imagePredicted/' + filesWithoutExtension[z] + '_sideOutput4.png', pred[3][z,:,:,:])
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[z] + '/imagePredicted/' + filesWithoutExtension[z] + '_sideOutput5.png', pred[4][z,:,:,:])
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[z] + '/imagePredicted/' + filesWithoutExtension[z] + '_fuseOutput.png', pred[5][z,:,:,:])




    
    # Saving orginal + label + predicted images
    for i in range(len(arrayTestingLabeledImg)):
        fig, axs = plt.subplots(1, 3, squeeze=False, figsize = (15,7))
        
        fig.tight_layout()
                 
        axs[0][0].imshow(arrayTestingOriginalImg[i], cmap='Greys_r')
        axs[0][1].imshow(arrayTestingLabeledImg[i], cmap='Greys_r')
        axs[0][2].imshow(pred[5][i,:,:,:], cmap='Greys_r')
        
        axs[0][0].axis('off')
        axs[0][1].axis('off')
        axs[0][2].axis('off')
             
        
        fileNameAndPath = os.path.join(outputPath + '/imagesOutput/' + filesWithoutExtension[i] + '/imagePredicted/' + filesWithoutExtension[i] + '_together.png')
           
        plt.show()
        fig.savefig(fileNameAndPath)



    for i in range(len(arrayTestingLabeledImg)):
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[i] + '/imageLabeled/' + filesWithoutExtension[i] + '_labeled.png', arrayTestingLabeledImg[i])

    
    for i in range(len(arrayTestingOriginalImg)):
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/' + filesWithoutExtension[i] + '/imageOriginal/' + filesWithoutExtension[i] + '_original.png', arrayTestingOriginalImg[i])


    for i in range(len(arrayTestingLabeledImg)):
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/labeled/' + filesWithoutExtension[i] + '_labeled.png', arrayTestingLabeledImg[i])

    for i in range(len(arrayTestingLabeledImg)):
        tf.keras.preprocessing.image.save_img(outputPath + '/imagesOutput/predicted/' + filesWithoutExtension[i] + '_predicted.png', pred[5][i,:,:,:])

    return outputPath



def main():

    outputPath = training()
    

    label_images = sorted(glob.glob(outputPath + '/imagesOutput/labeled/*.*'))
    pred_images = sorted(glob.glob(outputPath + '/imagesOutput/predicted/*.*'))
    
    files = outputPath + '/imagesOutput/labeled'
    files = sorted(os.listdir(files))
    
    
    counter = 0
    
    for z in range(len(files)):
        if(".DS_Store" == files[z]):
            counter = 1
            
    if (counter == 1):
        files.remove(".DS_Store")


    count = 0

    for z in range(len(label_images)):
        if(".DS_Store" == label_images[z]):
            count = 1
            
    if (count == 1):
        label_images.remove(".DS_Store")


    counte = 0

    for z in range(len(pred_images)):
        if(".DS_Store" == pred_images[z]):
            counte = 1
            
    if (counte == 1):
        pred_images.remove(".DS_Store")
        
 
    def get_metrics_np(pred,label,black_to_white=False):
        
        tp = np.sum(np.logical_and(pred == 0, label == 0))                                  
        tn = np.sum(np.logical_and(pred == 255, label == 255))                             
        fp = np.sum(np.logical_and(pred == 0, label == 255))
        fn = np.sum(np.logical_and(pred == 255, label == 0))
    
        return tp, tn, fp, fn
    
    

    def get_metrics_for_pair_simple(pred, label, thresh=-1, black_to_white= False): 
    
        pred  = cv2.imread(pred,0)
        label = cv2.imread(label,0)
        
        tp, tn, fp, fn = get_metrics_np(pred,label)
    
        try:
            tpr = float(tp)/(float(tp) + float(fn))
            fpr = float(fp)/(float(tp) + float(fn))

        except Exception as e:
            tpr = 1
            fpr = 1
    
        try:
            accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
            precision = float(tp)/(float(tp) + float(fp))
            recall = tpr
            fmeasure = (2 * (precision * recall)) / (precision + recall)
            
        except Exception as e:
            accuracy = float(0);
            precision = float(0);
            recall = float(0);
            fmeasure = float(0);
    
        return accuracy, precision, recall, fmeasure # we want to save our thresholds too
    
    
    
    eval_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'fmeasure'])
    accuracy, precision, recall, fmeasure  = [], [], [], []
    
    for i in range(len(label_images)):
        this_accuracy, this_precision, this_recall, this_fmeasure = get_metrics_for_pair_simple(pred_images[i], label_images[i], thresh=-1, black_to_white= False)
        accuracy.append(this_accuracy)
        precision.append(this_precision)
        recall.append(this_recall)
        fmeasure.append(this_fmeasure)
        
    eval_df['accuracy'] = accuracy  
    eval_df['precision'] = precision
    eval_df['recall'] = recall
    eval_df['fmeasure'] = fmeasure
            
    
    mean_accuracy = round(eval_df["accuracy"].mean(), 3)
    mean_precision = round(eval_df["precision"].mean(), 3)
    mean_recall = round(eval_df["recall"].mean(), 3)
    mean_f1score = round(eval_df["fmeasure"].mean(), 3)
    
    
    print(f"Accuracy_avg:  {mean_accuracy}\nPrecision_avg: {mean_precision}\nRecall_avg:    {mean_recall}\nF1-Score:  {mean_f1score}")
    
    
    j = 0

    with open(outputPath + '/report/metrics/metrics-imgs-ois.txt', 'w') as file:
        for x in range(len(files)):
            file.write('-----------------------------------\n')
            file.write('IMAGE [{}]\n'.format(x))
            file.write('Accuracy for image    = {0:.3}\n'.format(accuracy[x]))
            file.write('Precision for image   = {0:.3}\n'.format(precision[x]))
            #file.write('Recall for image      = {0:.3}\n'.format(recall[x]))
            file.write('F1 measure for image  = {0:.3}\n'.format(fmeasure[x]))
            file.write('\nFilename: ' + files[j] + '\n')
        
            j = j + 1
        file.close()


    with open(outputPath + '/report/metrics/metrics.txt', 'w') as file:

        file.write('-----------------------------------\n')
        file.write('METRICS\n')
        file.write('Accuracy for image    = {0:.3}\n'.format(mean_accuracy))
        file.write('Precision for image   = {0:.3}\n'.format(mean_precision))
        file.write('Recall for image      = {0:.3}\n'.format(mean_recall))
        file.write('F1-Score              = {0:.3}\n'.format(mean_f1score))

        file.close()

    
    
if __name__ == '__main__':
    main()  
        