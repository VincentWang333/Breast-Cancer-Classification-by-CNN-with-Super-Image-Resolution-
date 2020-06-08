BATCH_SIZE = 128
IMAGE_SIZE = 33
LABLE_SIZE = 21
SCALE = 2
STRIDE = 21 #OR 14
PADDING = (IMAGE_SIZE-LABLE_SIZE)//2

import imageio
import scipy.ndimage
import cv2
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D,Convolution2D, Activation, Input
from keras.optimizers import Adam
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from pathlib import Path
import os
from glob import glob
from skimage.measure import compare_ssim as ssim
import math
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import random
from keras import backend as k
from PIL import Image
from numpy import zeros, newaxis
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


def pre_process(image_path):
  #read the image and convert to YCbCr format
  #image = imageio.imread(image_path, as_gray=True, pilmode='RGB').astype(np.float)
  image = cv2.imread(image_path,cv2.IMREAD_COLOR)

  #modcrop the imagve so that the image can be scaled tightly
  image = modcrop(image,SCALE)
  #generate the label
  label_= image


  shape = image.shape

  #down scale the image and then unscale so that the input is generated
  input_ = cv2.resize(image, ((int)(shape[1] / SCALE), (int)(shape[0] / SCALE)), cv2.INTER_CUBIC)
  input_ = cv2.resize(input_, ((int)(shape[1]), (int)(shape[0])), cv2.INTER_CUBIC)

  return input_, label_


def modcrop(image, scale=3):
  (h, w) = image.shape[:2]
  w -= int(w % scale)
  h -= int(h % scale)
  image = image[0 : h, 0 : w]
  return image


def imsave(image, path):
  '''
  this function show the image directorly
  '''
  return cv2.imwrite(path, image )


def show_image(data,name):
  '''
  this function convert the matrix to the image and show out
  '''
  img = plt.imshow(data, interpolation='nearest')
  img.set_title = name
  plt.show(img)

def sub_image_process(images_set):
  sub_input_sequences = []
  sub_label_sequences = []
  for image in images_set:
    #preprocess the image so get the input_ and label_
    input_, label_ = pre_process(image)

    (h, w) = label_.shape[:2]

    #slide a window from left to right and top to bottom
    for x in range(0, h-IMAGE_SIZE+1,STRIDE):
      for y in range(0,w-IMAGE_SIZE+1, STRIDE):
        sub_input = input_[x:x+IMAGE_SIZE, y:y+IMAGE_SIZE]
        sub_label = label_[x+PADDING:x+PADDING+LABLE_SIZE,y+PADDING:y+PADDING+LABLE_SIZE];

        sub_input = sub_input.reshape([IMAGE_SIZE, IMAGE_SIZE, 3])
        sub_label = sub_label.reshape([LABLE_SIZE, LABLE_SIZE, 3])

        sub_input_sequences.append(sub_input)
        sub_label_sequences.append(sub_label)

  return sub_input_sequences,sub_label_sequences

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



def generate_sub_train_data():
  DATASET = 'srcnn_train/'
  filenames = os.listdir(DATASET)
  images_dir = os.path.join(os.getcwd(), DATASET)
  images_set = glob.glob(os.path.join(images_dir, "*.png"))
  train_inputs, train_labels = sub_image_process(images_set)
  return train_inputs,train_labels

def generate_sub_validation_data():
  VALIDATION_DATASET = 'srcnn_validate/'
  validation_filenames = os.listdir(VALIDATION_DATASET)
  validation_images_dir = os.path.join(os.getcwd(), VALIDATION_DATASET)
  validation_images_set = glob.glob(os.path.join(validation_images_dir, "*.png"))
  validation_inputs, validation_labels = sub_image_process(validation_images_set)
  return validation_inputs, validation_labels


def psnr(target,ref):
  target_data = target.astype(float)
  ref_data = ref.astype(float)
  diff = ref_data - target_data

  rmse = math.sqrt(np.mean(diff ** 2.))
  return 20 * math.log10(255./rmse)

def mse(target,ref):
  err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
  err /= float(target.shape[0] * target.shape[1])

  return err

def compare_images(target,ref):
  scores = []
  scores.append(psnr(target,ref))
  scores.append(mse(target,ref))
  scores.append(ssim(target,ref,multichannel=True))
  return scores

def print_scores(scores):
  print("psnr: " + str(scores[0]))
  print("mes: " + str(scores[1]))
  print("ssim: " + str(scores[2]))

def filter():
    model = SRCNN()
    model = load_model("srcnn_weight/srcnn.h5")
    model = Model(inputs=model.inputs, outputs=model.layers[3].output)
    predict_sample = glob('srcnn_predict_sample/*.png')
    image_path = predict_sample[0]
    img = load_img(image_path)
    features_maps = model.predict(img)
    ix = 1
    for _ in range(4):
        index = 1
        for _ in range(16):
            ax = pyplot.subplot(4,4,index)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(feature_maps[0, :, :,ix-1])
            ix += 1
            index += 1
        pyplot.show()




def main():
    predict_sample = glob('srcnn_predict_sample/*.png')
    Path('srcnn_predict_sample/result').mkdir()

    model = SRCNN()
    model = load_model("srcnn_weight/srcnn.h5")

    count = 0;
    for image_path in predict_sample:
        print(image_path)
        input_, label_ = pre_process(image_path)
        output = np.zeros(label_.shape)
        (h, w) = label_.shape[:2]
        for y in range(0, h - IMAGE_SIZE + 1, LABLE_SIZE):
            for x in range(0, w - IMAGE_SIZE + 1, LABLE_SIZE):
                # crop ROI from scaled image
                crop = input_[y : y + IMAGE_SIZE, x : x + IMAGE_SIZE]
                #crop.reshape([IMAGE_SIZE, IMAGE_SIZE, 3])
                # make a prediction on the crop and store it in output image

                P = model.predict(np.expand_dims(crop, axis = 0))
                P = P.reshape((LABLE_SIZE, LABLE_SIZE, 3))
                output[y + PADDING : y + PADDING + LABLE_SIZE,
                       x + PADDING : x + PADDING + LABLE_SIZE] = P


        output = output[PADDING : h - ((h % IMAGE_SIZE) + PADDING),
                        PADDING : w - ((w % IMAGE_SIZE) + 3*PADDING)]
        output = np.clip(output, 0, 255).astype("uint8")


        print("----------------------" + image_path + " ----------------")
        print("----------------" + str(count) + "----------------")


        label_ = label_[PADDING : h - ((h % IMAGE_SIZE) + PADDING),
                        PADDING : w - ((w % IMAGE_SIZE) + 3*PADDING)]

        input_ = input_[PADDING : h - ((h % IMAGE_SIZE) + PADDING),
                        PADDING : w - ((w % IMAGE_SIZE) + 3*PADDING)]

        label_path = 'srcnn_predict_sample/result/label_'+ str(count)+'.png'
        cv2.imwrite(label_path, label_)
        input_path = 'srcnn_predict_sample/result/input_'+ str(count)+'.png'
        cv2.imwrite(input_path, input_)

        scores_biubic = compare_images(label_,input_)
        print("biubic scores: ")
        print_scores(scores_biubic)

        output_path = 'srcnn_predict_sample/result/output_'+ str(count)+'.png'
        cv2.imwrite(output_path, output)

        scores_srcnn = compare_images(label_,output)
        print("srcnn scores: ")
        print_scores(scores_srcnn)
        count += 1
if __name__ == '__main__':
    filter()
