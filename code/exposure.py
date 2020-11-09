##########################################################
#                    Exposureness                        #
# This function measures the exposureness of the images. #
##########################################################
import cv2
import numpy as np


############################################################################
# This function measures the exposureness of the image and returns the     # 
# exposureness as a feature of the image for further use.                  #
#                                                                          #
# input: image_paths: list of N elements containing image paths            #
#                                                                          #
# output: exposures: a N x D list of histograms of color distribution of   #
#         the images. (N = number of image paths. D = 8)                   #
############################################################################
def exposureness(image_paths):
    exposures = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        img = np.floor(img/16)
        histogram = np.histogram(img, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        histogram = histogram[0] / np.linalg.norm(histogram[0])
        if exposures == []:
            exposures = [histogram]
        else:
            exposures = np.concatenate((exposures, [histogram]), axis = 0)
    return exposures

###################################################
#                    Test                         #
# This is for testing the function and            #
# demonstrating how to use the function.          #
###################################################
if __name__ == '__main__':
    paths = []
    path = '../data/good/a-house-near-the-lake-shore-3218137.jpg'
    paths.append(path)
    paths.append('../data/bad/a-house-near-the-lake-shore-3218137-dark.jpg')
    paths.append('../data/bad/a-house-near-the-lake-shore-3218137-bright.jpg')

    exposures = exposureness(paths)
    print(exposures)
    print ("complete")