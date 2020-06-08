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
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import glob

from skimage.measure import compare_ssim as ssim
import math



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
  DATASET = 'srcnn/train/'
  filenames = os.listdir(DATASET)
  images_dir = os.path.join(os.getcwd(), DATASET)
  images_set = glob.glob(os.path.join(images_dir, "*.png"))
  train_inputs, train_labels = sub_image_process(images_set)
  return train_inputs,train_labels

def generate_sub_validation_data():
  VALIDATION_DATASET = 'srcnn/validate/'
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


#build the model
train_model = SRCNN()
#set up the sub train inputs and data
sub_train_inputs, sub_train_labels = generate_sub_train_data()
#set up the sub validation inputs and data
validation_sub_inputs, validation_sub_labels = generate_sub_validation_data()

train_inputs = np.asarray(sub_train_inputs)
train_label = np.asarray(sub_train_labels)
validation_inputs = np.asarray(validation_sub_inputs)
validation_labels = np.asarray(validation_sub_labels)
#start trainning
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model_checkpoint = ModelCheckpoint('srcnn.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = train_model.fit(train_inputs,train_label,batch_size = BATCH_SIZE, epochs = 5, verbose = 1,validation_data=(validation_inputs,validation_labels),callbacks=[early_stopping_monitor,model_checkpoint])
scores = train_model.evaluate(train_inputs,train_label,verbose = 0)
print('Accuracy: %.2f%%' % (scores[1]*100))
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('srcnn_accuracy.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('srcnn_loss.png')
plt.show()
#save trainning data in .h5 file
train_model.save("srcnn.h5")
