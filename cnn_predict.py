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
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
import cv2
import random
from keras import backend as k
from PIL import Image
from numpy import zeros, newaxis
from matplotlib.pyplot import imshow
from matplotlib import pyplot





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
    opt = keras.optimizers.Adam(lr=0.00003)
    model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
    return model


def filters():
    model = cnn_4()
    model.load_weights('cnn_weight/new/best_cnn.h5')
    for layer in model.layers:
        if'conv' not in layer.name:
            continue
    filters,biases = model.layers[1].get_weights()
    f_min,f_max = filters.min(),filters.max()
    filters = (filters - f_min)/(f_max - f_min)

    n_filters,ix = 8,1
    for i in range(0,8):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)

    ix = 1
    pyplot.show()
    for i in range(8,16):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()

    ix = 1
    for i in range(16,24):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()

    ix = 1
    for i in range(24,32):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()

    ix = 1
    for i in range(32,40):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()

    ix = 1
    for i in range(40,48):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()

    ix = 1
    for i in range(48,56):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()

    ix = 1
    for i in range(56,64):
        f = filters[:,:,:,i]
        for j in range(3):
            ax = pyplot.subplot(n_filters,3,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(f[:,:,j])
            ix+=1
    print(layer.name,filters.shape)
    pyplot.show()


def main():

    model = cnn_4()
    model.load_weights('cnn_weight/new/best_cnn.h5')

    pred_mal_and_act_mal_tp = 0
    pred_mal_but_act_ben_fp = 0
    pred_ben_but_act_mal_fn = 0
    pred_ben_and_act_ben_tn = 0

    total = 0
    correct = 0

    original_predict_benign_images = glob('cnn/srcnn_enhanced_predict/benign/*.png')
    original_predict_malignant_images = glob('cnn/srcnn_enhanced_predict/malignant/*.png')
    for image_path in original_predict_benign_images:
        result = ''
        print(image_path)
        img = load_img(image_path ,target_size = (50,50))
        img = img_to_array(img)
        img = expand_dims(img,axis=0)
        img /= 255.0
        predict = model.predict(img)
        if(predict[0][0]>predict[0][1]):
            result = 'Benign'
            pred_ben_and_act_ben_tn += 1
            correct += 1
        else:
            result = 'Maligant'
            pred_mal_but_act_ben_fp += 1
        total += 1

    for image_path in original_predict_malignant_images:
        result = ''
        print(image_path)
        img = load_img(image_path ,target_size = (50,50))
        img = img_to_array(img)
        img = expand_dims(img,axis=0)
        img /= 255.0
        predict = model.predict(img)
        if(predict[0][0]>predict[0][1]):
            result = 'Benign'
            pred_ben_but_act_mal_fn += 1
        else:
            result = 'Maligant'
            pred_mal_and_act_mal_tp += 1
            correct += 1
        total += 1

    print("Predicted malignatn and actually malignant (true positive):" + str(pred_mal_and_act_mal_tp))
    print("Predicted malignatn but actually benign (false positive):" + str(pred_mal_but_act_ben_fp))
    print("Predicted benign but actually malignant (false negative):" + str(pred_ben_but_act_mal_fn))
    print("Predicted benign and actually benign (true negative):" + str(pred_ben_and_act_ben_tn))
    print('accuracy: ' + str(correct/total))

def filters_maps():
    model = cnn_4()
    model.load_weights('cnn_weight/new/best_cnn.h5')
    model=Model(inputs=model.inputs, outputs=model.layers[1].output)
    benign_original = Path('cnn_predict_sample/benign/original.png')
    benign_lr = Path('cnn_predict_sample/benign/lr.png')
    benign_srcnn = Path('cnn_predict_sample/benign/srcnn.png')

    malignant_original = Path('cnn_predict_sample/malignant/original.png')
    malignant_lr = Path('cnn_predict_sample/malignant/lr.png')
    malignant_srcnn = Path('cnn_predict_sample/malignant/srcnn.png')

    img = load_img(malignant_srcnn ,target_size = (50,50))
    img = img_to_array(img)
    img = expand_dims(img,axis=0)
    img /= 255.0
    features_maps = model.predict(img)
    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = pyplot.subplot(square,square,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(features_maps[0, :, :, ix-1])
            ix += 1
    pyplot.show()


if __name__ == '__main__':
    # main()
    # filters()
    filters_maps()
