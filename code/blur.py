###################################################
#                    Blur                         #
# This part measures the sharpness of the images. #
###################################################
import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import gradient

############################################################################
# Function that returns the sharpness of the images,                       # 
# which is the average number of the square gradients of the image.        #
# This will be one of the features of the image for neural network to use. #
#                                                                          #
# input: image_paths: list of N elements containing image paths            #
#                                                                          #
# output: sharpness: a float that represents the average number of the     #
#         square gradients of the image.                                   #
############################################################################
def sharpness(image_paths):
    sharpness = []

    selection_element = disk(5)  # matrix of n pixels with a disk shape

    for path in image_paths:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_sharpness = gradient(img, selection_element)
        img_sharpness_f = np.asarray(img_sharpness).flatten()
        sharpness.append(np.average(img_sharpness_f))
    sharpness = np.transpose([np.array(sharpness)])
    return sharpness

###################################################
#                    Test                         #
# This is for testing the function and            #
# demonstrating how to use the function.          #
###################################################
if __name__ == '__main__':
    paths = []
    path = '../data/bad/a-house-near-the-lake-shore-3218137-blur.jpg'
    paths.append(path)
    paths.append('../data/good/abc.jpg')

    print(paths)
    sharpness = sharpness(paths)
    print(sharpness)
    print("complete")