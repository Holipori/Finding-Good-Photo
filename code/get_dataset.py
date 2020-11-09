import os
import os.path as osp
from glob import glob
from blur import sharpness
from exposure import exposureness
from object import object
from facial import facial
from orientation import orientation
import numpy as np


def readpath(paths):
    ######################################################################################
    # Function that extract all the features from the input paths
    #
    # input: image_paths: list of N elements containing image paths
    # output: N x M matrix. where M is the dimensionality of the feature representation.
    ######################################################################################
    print('getting sharpness')
    temp = sharpness(paths)
    temp = temp / np.linalg.norm(temp)
    input = np.copy(temp)
    print(input.shape)

    print('getting exposureness')
    temp = exposureness(paths)
    input = np.concatenate((input, temp), axis=1)
    print(input.shape)

    print('getting object position')
    temp = object(paths)
    input = np.concatenate((input, temp), axis=1)
    print(input.shape)

    print('getting orientation')
    temp = orientation(paths)
    input = np.concatenate((input, temp), axis=1)
    print(input.shape)

    print('getting facial expression')
    temp = facial(paths)
    input = np.concatenate((input, temp), axis=1)
    print(input.shape)

    return(input)

if __name__ == '__main__':
    #########################################################
    # this main function include some prepossessing
    # to the dataset before the features extraction.
    #########################################################
    data_path = '..\data3\good'
    pth = osp.join(data_path, '*.{:s}'.format('jpg'))
    good_pth = glob(pth)
    a = np.array(good_pth)
    a = a.reshape(len(good_pth),1)
    b = np.ones((len(good_pth), 1)) # adding label '1' for good photo
    good_set = np.concatenate((a, b), axis = 1)


    data_path = '..\data3\\bad'
    pth = osp.join(data_path, '*.{:s}'.format('jpg'))
    bad_pth = glob(pth)
    a = np.array(bad_pth)
    a = a.reshape(len(bad_pth),1)
    b = np.zeros((len(bad_pth), 1)) # adding label '0' for bad photo
    bad_set = np.concatenate((a, b), axis = 1)

    set = np.concatenate((good_set, bad_set), axis = 0) # concatenate the paths of 2 classes together
    print(set.shape)
    del good_set, bad_set

    X, y = set[:, 0], set[:, 1]


    X = readpath(X)
    y = y.reshape(len(y), 1)
    data = np.concatenate((X, y), axis = 1)
    np.save('data_with_mine.npy',data)

