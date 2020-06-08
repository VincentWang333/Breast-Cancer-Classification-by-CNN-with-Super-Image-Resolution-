from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D,Convolution2D, Activation, Input
from keras.optimizers import Adam
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
import cv2
import os
from glob import glob
import numpy as np
from pathlib import Path



BATCH_SIZE = 128
IMAGE_SIZE = 33
LABLE_SIZE = 21
SCALE = 2
STRIDE = 21 #OR 14
PADDING = (IMAGE_SIZE-LABLE_SIZE)//2


def SRCNN(image_shape = (33,33,3)):
  srcnn = Sequential()
  # srcnn.add(Conv2D(64,(9,9), kernel_initializer='he_normal',activation='relu',input_shape=image_shape,padding='valid'))
  # srcnn.add(Conv2D(32,(1,1), kernel_initializer='he_normal',activation='relu',padding='valid'))
  # srcnn.add(Conv2D(3,(5,5), kernel_initializer='he_normal',activation='relu',padding='valid',))


  input_image = Input(shape = image_shape)
  C1 = Convolution2D(64,(9,9), kernel_initializer='he_normal',input_shape=image_shape,padding='valid')(input_image)
  A1 = Activation('relu',name='layer_1')(C1)
  C2 = Convolution2D(32,(1,1), kernel_initializer='he_normal',padding='valid')(A1)
  A2 = Activation('relu',name='layer_2')(C2)
  C3 = Convolution2D(3,(5,5), kernel_initializer='he_normal',padding='valid')(A2)
  A3 = Activation('relu',name='layer_3')(C3)
  srcnn = Model(input_image,A3)


  opt = optimizers.Adam(lr=0.0003)
  srcnn.compile(optimizer=opt,loss='mean_squared_error', metrics = ['accuracy'])
  # categorical_crossentropy
  srcnn.summary()
  return srcnn


def main():
    predict_400X_lr_benign = glob('cnn/lr_predict/benign/*.png')
    predict_400X_lr_malignant = glob('cnn/lr_predict/malignant/*.png')

    predict_model = SRCNN()
    predict_model = load_model("srcnn_weight/srcnn.h5")

    Path('cnn/srcnn_enhanced_predict').mkdir()
    Path('cnn/srcnn_enhanced_predict/benign').mkdir()
    Path('cnn/srcnn_enhanced_predict/malignant').mkdir()


    for input in predict_400X_lr_benign:
        print(input)
        image = cv2.imread(input,cv2.IMREAD_COLOR)
        output = np.zeros(image.shape)
        (h, w) = image.shape[:2]
        for y in range(0, h - IMAGE_SIZE + 1, LABLE_SIZE):
            for x in range(0, w - IMAGE_SIZE + 1, LABLE_SIZE):
                # crop ROI from scaled image
                crop = image[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
                #crop.reshape([IMAGE_SIZE, IMAGE_SIZE, 3])
                # make a prediction on the crop and store it in output image

                P = predict_model.predict(np.expand_dims(crop, axis = 0))
                P = P.reshape((LABLE_SIZE, LABLE_SIZE, 3))
                output[y + PADDING : y + PADDING + LABLE_SIZE,
                       x + PADDING : x + PADDING + LABLE_SIZE] = P


        output = output[PADDING : h - ((h % IMAGE_SIZE) + PADDING),
                        PADDING : w - ((w % IMAGE_SIZE) + 3 * PADDING)]
        output = np.clip(output, 0, 255).astype("uint8")
        new_image_path = 'cnn/srcnn_enhanced_predict/benign/'+ os.path.basename(input)
        cv2.imwrite(new_image_path, output)

    for input in predict_400X_lr_malignant:
        print(input)
        image = cv2.imread(input,cv2.IMREAD_COLOR)
        output = np.zeros(image.shape)
        (h, w) = image.shape[:2]
        for y in range(0, h - IMAGE_SIZE + 1, LABLE_SIZE):
            for x in range(0, w - IMAGE_SIZE + 1, LABLE_SIZE):
                # crop ROI from scaled image
                crop = image[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
                #crop.reshape([IMAGE_SIZE, IMAGE_SIZE, 3])
                # make a prediction on the crop and store it in output image

                P = predict_model.predict(np.expand_dims(crop, axis = 0))
                P = P.reshape((LABLE_SIZE, LABLE_SIZE, 3))
                output[y + PADDING : y + PADDING + LABLE_SIZE,
                       x + PADDING : x + PADDING + LABLE_SIZE] = P


        output = output[PADDING : h - ((h % IMAGE_SIZE) + PADDING),
                        PADDING : w - ((w % IMAGE_SIZE) + 3 * PADDING)]
        output = np.clip(output, 0, 255).astype("uint8")
        new_image_path = 'cnn/srcnn_enhanced_predict/malignant/'+ os.path.basename(input)
        cv2.imwrite(new_image_path, output)

if __name__ == '__main__':
    main()
