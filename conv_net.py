from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import sys
import data_utils

if __name__ == '__main__':
    batch_size = 32
    nb_classes = 20
    nb_epoch = 10

    img_rows, img_cols = 32, 32
    img_channels = 3

    (X_train, y_train,X_test, y_test) = data_utils.load_CIFAR100("dataset/cifar-100-python/")

    X_val = X_train[49000:50000,:,:,:]
    y_val = y_train[49000:50000]
    X_train = X_train[0:49000,:,:,:]
    y_train = y_train[0:49000]
    X_test = X_test[0:10000,:,:,:]
    y_test = y_test[0:10000]


    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_val /=255

    model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_val, Y_val),
                  shuffle=True)

    loss, accuracy = model.evaluate(X_test, Y_test, show_accuracy=True)
    print("loss ",loss)
    print("Accuracy: ",accuracy)
