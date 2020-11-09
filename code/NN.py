from keras.models import *
from keras.layers import *
import numpy as np
import matplotlib.pyplot as plt
from get_dataset import readpath
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

def mymodel(inputsize = 45, outputsize = 2):
    ####################################################
    # this is the model for training our final features
    ####################################################
    model = Sequential()

    # first hidden layer has 32 nodes
    model.add(Dense(32, input_dim=inputsize))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # second hidden layer has 32 nodes
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # output layer has 2 nodes
    model.add(Dense(outputsize))
    model.add(Activation("softmax"))

    return model

def train():
    ######################################################
    # training all the features extracted from dataset
    ###################################################
    data = np.load('data_with_mine.npy') # load dataset
    data = data.astype(float)

    # get labels
    datalabel = []
    for i in range(len(data)):
        if data[i, -1] == 0:
            print('0')
            datalabel.append([1, 0])
        else:
            print('1')
            datalabel.append([0, 1])
    datalabel = np.array(datalabel)
    print(datalabel.shape)

    # split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], datalabel, test_size=0.25, random_state=3)

    # start training
    model = mymodel(inputsize = 45, outputsize = 2)
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, validation_data = (X_test, y_test), batch_size=20)
    # save model weights
    model.save_weights('NN.h5')

    # show the training process
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def test_my_data():
    #####################################
    # Function that test new images
    #####################################
    # load my model
    model = mymodel(inputsize=45, outputsize=2)
    opt = optimizers.Adam(learning_rate=0.0005)
    model.load_weights('NN.h5')
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # set paths
    paths = []
    paths.append('test/15.jpg')
    paths.append('test/16.jpg')
    paths.append('test/17.jpg')
    paths.append('test/18.jpg')

    # predicting
    feature = np.array(readpath(paths))
    predictions = model.predict(feature)

    print(predictions)

def baseline_model(input_shape , output = 2):
    ###############################################################
    # this is the baseline CNN model for training the original images
    ##############################################################
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation("relu"))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.2)))
    model.add(Activation("relu"))
    model.add(Dense(output, kernel_regularizer=regularizers.l2(0.2)))
    model.add(Activation('softmax'))

    return model

def train_baseline():
    ###################################################
    # Function that trains the baseline model
    ###################################################
    img_size = (640,480)
    train_gen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2)
    train_generator = train_gen.flow_from_directory('../data2/train',
                                                   target_size=img_size,
                                                   batch_size=20,
                                                   color_mode='grayscale',
                                                   class_mode="categorical",
                                                   subset='training',
                                                   )
    valid_generator = train_gen.flow_from_directory('../data2/test',
                                                    target_size=img_size,
                                                    batch_size=16,
                                                    color_mode='grayscale',
                                                    class_mode="categorical",
                                                    subset='validation',
                                                    )
    model = baseline_model(input_shape = (640,480,1))
    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  validation_data=valid_generator,
                                  validation_steps=valid_generator.n // valid_generator.batch_size,
                                  epochs=20,
                                  )

    model.save_weights('baseline.h5')

    # show the training process
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__=='__main__':
    ##################################################################
    # this main function can be used to decide which function to run
    ##################################################################
    train()
    # test_my_data()
    # train_baseline()
