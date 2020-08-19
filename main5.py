from array import array as pyarray
import struct
import keras
#load mnist dataset
import matplotlib.pyplot as plt
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import math
import os
import tensorflow as tf
from keras.datasets import mnist
from keras.models import load_model
import random


########################
## CNN Neural Network ##
########################
class Network():
    def __init__(self):
        input_shape = None
        X_train = None
        y_train = None
        X_test = None
        y_test = None
        images = None
        digits = None
        num_category = None

    def prepare_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        self.images = X_train        #Copy used for visualization etc.
        self.digits = y_train        #Copy used for visualization etc.

        # input image dimensions
        img_rows, img_cols = 28, 28

        # reshaping data
        if k.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        self.X_train = X_train.astype('float32')
        self.X_test = X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        self.num_category = 10       # Set number of categories/classes

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(y_train, self.num_category)
        self.y_test = keras.utils.to_categorical(y_test, self.num_category)
    
    # 128 is default batch size. Use 10 categories because there are 10 different digits.
    def create_model(self):
        # cnn model
        model = Sequential()
        #convolutional layer with rectified linear unit activation
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=self.input_shape,
                        ))
        #32 convolution filters used each of size 3x3
        #again
        model.add(Conv2D(64, (3, 3), activation='relu'))
        #64 convolution filters used each of size 3x3
        #choose the best features via pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #randomly turn neurons on and off to improve convergence
        model.add(Dropout(0.25))
        #flatten since too many dimensions, we only want a classification output
        model.add(Flatten())
        #fully connected to get all relevant data
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #output a softmax to squash the matrix into output probabilities
        model.add(Dense(self.num_category, activation='softmax'))
        return model


    def compile_model(self, model):
        #Adaptive learning rate (adaDelta a popular form of gradient descent)
        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])


    def train_model(self, model, batch_size=128, num_epoch=60):
        #model training
        model_log = model.fit(self.X_train, self.y_train,
                        batch_size=batch_size,
                        epochs=num_epoch,
                        verbose=1,                           
                        validation_data=(self.X_test, self.y_test))

if __name__ == '__main__':


    network = Network()
    network.prepare_data()
    model = network.create_model()
    network.compile_model(model)
    network.train_model(model)
    
    score = model.evaluate(network.X_test, network.y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) 
    
    #Save model
    print("Saving model")
    model.save('mnist_model.h5')
    print("Model saved")



    model = load_model('mnist_model.h5')
    weights = model.get_weights()
    
    network = Network()
    network.prepare_data()
    
    
    ## Pick 2 random images
    images_count = len(network.images)
    #random image index 1
    rii1 = random.randrange(0, images_count)
    #random image index 2
    rii2 = random.randrange(0, images_count)
    
    
    # Predict first digit #
    
    raw_image1 = network.images[rii1]
    parsed_image1 = np.array([raw_image1])
    parsed_image1 = parsed_image1[..., np.newaxis]
    
    prediction1 = np.round(model.predict(parsed_image1))
    
    maximum = np.max(prediction1)
    index_of_maximum = np.where(prediction1 == maximum)
    wrong_digit = [x[0] for x in index_of_maximum]
    
    prediction1_digit = wrong_digit[1]
    
    
    # Predict second digit #
    
    raw_image2 = network.images[rii2]
    parsed_image2 = np.array([raw_image2])
    parsed_image2 = parsed_image2[..., np.newaxis]
    
    prediction2 = np.round(model.predict(parsed_image2))
    
    maximum = np.max(prediction2)
    index_of_maximum = np.where(prediction2 == maximum)
    wrong_digit = [x[0] for x in index_of_maximum]
    
    prediction2_digit = wrong_digit[1]
    
    
    # Sum #
    
    prediction_sum = prediction1_digit + prediction2_digit
    real_sum = network.digits[rii1] + network.digits[rii2]
    
    
    
    # Plot results #
    
    fig = plt.figure()
    ##for i in range(2):
    plt.subplot(2, 2, 1)
    #plt.tight_layout()
    plt.imshow(network.images[rii1], cmap='gray', interpolation='none')
    plt.title("Predicted digit: {} Correct: {}".format(prediction1_digit, network.digits[rii1]))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(network.images[rii2], cmap='gray', interpolation='none')
    plt.title("Predicted digit: {} Correct: {}".format(prediction2_digit, network.digits[rii2]))
    plt.xticks([])
    plt.yticks([])
    
    plt.text(x=-30, y=40, s="Predicted sum: {} Correct: {}".format(prediction_sum, real_sum))
    
    fig
    plt.show()
    