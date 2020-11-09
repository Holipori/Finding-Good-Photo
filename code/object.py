import cv2
import cvlib as cv
import numpy as np
from sklearn.neighbors import NearestNeighbors


def object(image_paths):
    ################################################################################
    # function that detects object positions
    #
    # input: image_paths: list of N elements containing image paths
    # output: 1x25 histogram representing the position in the image.
    ################################################################################
    histogram = [] # create the histogram
    for path in image_paths:
        im = cv2.imread(path)
        bbox, label, conf = cv.detect_common_objects(im) # get the objects boundary box
        center = []
        for i in range(len(bbox)): # get the center position of the object
            x = int((bbox[i][0]+bbox[i][2])/2)
            y = int((bbox[i][1]+bbox[i][3])/2)
            if center == []:
                center = [[x,y]]
            else:
                center = np.concatenate((center,[[x,y]]),axis = 0)


        for i in range(5): # create the clustering centers
            for j in range(5):
                if i==0 and j == 0:
                    clu_centr = [[0, 0]]
                else:
                    clu_centr = np.concatenate((clu_centr,[[im.shape[1]/5 * i, im.shape[1]/5 * j]]), axis = 0)

        hist = np.zeros(25)
        for i in range(len(center)): # assign each object position to the clustering centers
            neigh = NearestNeighbors(n_neighbors=1).fit(clu_centr)
            distances, indices = neigh.kneighbors([center[i]])
            hist[indices[0][0]] += 1
        if histogram == []:
            histogram = [hist]
        else:
            histogram = np.concatenate((histogram, [hist]), axis = 0)
    return(histogram)

if __name__ == '__main__':
    ###################################################
    #                    Test                         #
    # This is for testing the function and            #
    # demonstrating how to use the function.          #
    ###################################################
    paths = []
    paths.append('../data/good/dancing-with-the-taj-mahal-in-the-orange-mist.jpg')
    paths.append('../data/good/friendly-smiling-man.jpg')
    object = object(paths)
    print(object)
    print("complete")