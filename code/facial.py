import cv2
from keras import optimizers
import numpy as np
from keras.models import *
from keras.layers import *

from keras import regularizers

def my_model():
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(7, kernel_regularizer=regularizers.l2(0.1)))
    model.add(Activation('softmax'))

    return model


def facial(image_paths):
    ################################################################################
    # function that detects human face and recognize the facial expression
    #
    # input: image_paths: list of N elements containing image paths
    # output: the confidence coefficients for bad expression and good expression.
    #           if no face detected, these two coefficients will both be 0.
    ################################################################################
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load the classifier file for face detection
    model = my_model()
    model.load_weights('facial_model.h5')
    sgd = optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    facial = []
    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            xa = x # setting the corner position of human face
            ya = y
            xb = x+w
            yb = y+h
            cv2.rectangle(img, (xa,ya ), (xb, yb), (255, 0, 0), 2)
            crop_img = gray[ya:yb,xa:xb] # crop the face
            crop_img = np.array(crop_img)
            crop_img = cv2.resize(crop_img, (48,48)) # resize the face
            crop_img = crop_img.reshape(48, 48, 1)
            crop_img = np.array([crop_img])/255
            prediction = model.predict(crop_img) # predict the facial expression
            prediction = np.array(prediction)
            a = np.array([[0.5,1.3]]) # tuning the final prediction
            prediction = np.multiply(prediction, a)
        if len(faces) == 0:
            if facial == []:
                facial = [[0,1]]
            else:
                facial = np.concatenate((facial, [[0,1]]), axis=0)
            continue
        if facial == []:
            facial = prediction
        else:
            facial = np.concatenate((facial, prediction), axis = 0)

    return facial


if __name__ == '__main__':
    ###################################################
    #                    Test                         #
    # This is for testing the function and            #
    # demonstrating how to use the function.          #
    ###################################################
    paths = []

    paths.append('test/15.jpg')
    paths.append('test/16.jpg')
    paths.append('test/17.jpg')
    paths.append('test/18.jpg')
    paths.append('test/testSet/set2/good/1.jpg')
    paths.append('test/testSet/set2/bad/2.jpg')
    paths.append('test/testSet/set2/bad/3.jpg')
    paths.append('test/testSet/set2/bad/4.jpg')

    facial = facial(paths)
    print(facial)
    print("complete")
