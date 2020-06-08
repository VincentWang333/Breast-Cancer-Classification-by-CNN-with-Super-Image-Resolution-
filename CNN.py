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

def main():
    model = cnn_4()
    toal_train = len(glob('cnn/train/*/*.png'))



    train_datagen = ImageDataGenerator(rescale=1/255.0,
                                       rotation_range = 90,
                                       horizontal_flip=True,
                                       vertical_flip=True,)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('cnn/train',
                                                     target_size=(50, 50),
                                                     batch_size=BATCH_SIZE)
    test_set = test_datagen.flow_from_directory('cnn/validate',
                                                 target_size=(50, 50))



    # early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint('best_cnn_400x.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


    label_map = (training_set.class_indices)
    print(label_map)

    trainning = model.fit_generator(training_set,
                                    steps_per_epoch = toal_train//BATCH_SIZE,
                                    epochs=EPOCHS,
                                    validation_data = test_set,
                                    callbacks=[model_checkpoint])
    model.save("cnn_400x.h5")
    # list all data in history
    print(trainning.history.keys())
    # summarize history for accuracy
    plt.plot(trainning.history['accuracy'])
    plt.plot(trainning.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('cnn_400x_acc.png')
    plt.show()
    # summarize history for loss
    plt.plot(trainning.history['loss'])
    plt.plot(trainning.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('cnn_400x_loss.png')


if __name__ == '__main__':
    main()
