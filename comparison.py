import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Model
from keras.layers import Convolution2D, Activation, Input
from keras.optimizers import Adam
from keras import optimizers
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from pathlib import Path
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
import cv2


BATCH_SIZE = 16
CLASSES = 2
EPOCHS = 80


def cnn_4():
    model = Sequential()
    model.add(Conv2D(32,kernel_size = (3,3),activation='relu',input_shape=(50,50,3)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(CLASSES,activation='sigmoid'))
    model.summary()
    opt = keras.optimizers.Adam(lr=0.000031)
    model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
    return model


def main():
    model = cnn_4()
    model.load_weights('cnn_weight/new/best_cnn.h5')

    original_predict_benign_images = glob('cnn/srcnn_enhanced_predict/benign/*.png')
    original_predict_malignant_images = glob('cnn/srcnn_enhanced_predict/malignant/*.png')
    true_false_true = []
    true_true_false = []
    for image_path in original_predict_benign_images:
        print(image_path)
        result = ''
        img_name = os.path.basename(image_path)
        orig_img = load_img(image_path ,target_size = (50,50))
        orig_img = img_to_array(orig_img)
        orig_img = expand_dims(orig_img,axis=0)
        orig_img /= 255.0
        predict = model.predict(orig_img)
        if(predict[0][0]>predict[0][1]):
            result = 'Benign'
            lr_img_path = glob('cnn/lr_predict/benign/'+ img_name)[0]
            lr_img = load_img(lr_img_path ,target_size = (50,50))
            lr_img = img_to_array(lr_img)
            lr_img = expand_dims(lr_img,axis=0)
            lr_img /= 255.0
            predict = model.predict(lr_img)
            if(predict[0][0]>predict[0][1]):
                lr_predict_result = 'Benign'
                hr_img_path = glob('cnn/srcnn_enhanced_predict/benign/'+img_name)[0]
                hr_img = load_img(hr_img_path ,target_size = (50,50))
                hr_img = img_to_array(hr_img)
                hr_img = expand_dims(hr_img,axis=0)
                hr_img /= 255.0
                predict = model.predict(hr_img)
                if(predict[0][0]>predict[0][1]):
                    hr_predict = 'Benign'
                else:
                    hr_predict = 'Malingant'
                    true_true_false.append(hr_img_path)
            else:
                lr_predict_result = 'Malignant'
                lr_predict_result = 'Benign'
                hr_img_path = glob('cnn/srcnn_enhanced_predict/benign/'+img_name)[0]
                hr_img = load_img(hr_img_path ,target_size = (50,50))
                hr_img = img_to_array(hr_img)
                hr_img = expand_dims(hr_img,axis=0)
                hr_img /= 255.0
                predict = model.predict(hr_img)
                if(predict[0][0]>predict[0][1]):
                    hr_predict = 'Benign'
                    true_false_true.append(hr_img_path)
                else:
                    hr_predict = 'Malingant'
        else:
            result = 'Maligant'


    for image_path in original_predict_malignant_images:
        print(image_path)
        result = ''
        img_name = os.path.basename(image_path)
        orig_img = load_img(image_path ,target_size = (50,50))
        orig_img = img_to_array(orig_img)
        orig_img = expand_dims(orig_img,axis=0)
        orig_img /= 255.0
        predict = model.predict(orig_img)
        if(predict[0][0]>predict[0][1]):
            result = 'Benign'
        else:
            result = 'Maligant'
            lr_img_path = glob('cnn/lr_predict/malignant/'+img_name)[0]
            lr_img = load_img(lr_img_path ,target_size = (50,50))
            lr_img = img_to_array(lr_img)
            lr_img = expand_dims(lr_img,axis=0)
            lr_img /= 255.0
            predict = model.predict(lr_img)
            if(predict[0][0]>predict[0][1]):
                lr_predict_result = 'Benign'
                hr_img_path = glob('cnn/srcnn_enhanced_predict/malignant/'+img_name)[0]
                hr_img = load_img(hr_img_path ,target_size = (50,50))
                hr_img = img_to_array(hr_img)
                hr_img = expand_dims(hr_img,axis=0)
                hr_img /= 255.0
                predict = model.predict(hr_img)
                if(predict[0][0]>predict[0][1]):
                    hr_predict = 'Benign'
                else:
                    hr_predict = 'Malingant'
                    true_false_true.append(hr_img_path)
            else:
                lr_predict_result = 'Malignant'
                hr_img_path = glob('cnn/srcnn_enhanced_predict/malignant/'+img_name)[0]
                hr_img = load_img(hr_img_path ,target_size = (50,50))
                hr_img = img_to_array(hr_img)
                hr_img = expand_dims(hr_img,axis=0)
                hr_img /= 255.0
                predict = model.predict(hr_img)
                if(predict[0][0]>predict[0][1]):
                    hr_predict = 'Benign'
                    true_true_false.append(hr_img_path)
                else:
                    hr_predict = 'Malingant'

    print("true_false_true------")
    for i in true_false_true:
        print(i)
    print(len(true_false_true))


    print("true_true_false------")
    for i in true_true_false:
        print(i)
    print(len(true_true_false))



if __name__ == '__main__':
    main()
