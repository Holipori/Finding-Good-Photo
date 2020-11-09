##########################################################
#            Orientation Function                        #
# This returns the feature of orientation of the images. #
##########################################################
from skimage import feature, color, transform, io
import numpy as np
import logging
import matplotlib.pyplot as plt

######################################################################
# function to compute all the edgelets by using hough transformation #
######################################################################
def compute_edgelets(image, sigma=3):
    # hough tranformation
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=3,
                                               line_gap=5)
    locations = []
    directions = []
    strengths = []

    if len(lines)==0:
        return -1
    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) /np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)

#####################################
# concatenate the edgelets to lines #
#####################################
def edgelet_lines(edgelets):
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

###################################################################
# function computes the votes of each candidate vanishing points  #
###################################################################
def compute_votes(edgelets, model, threshold_inlier=5):
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths

###############################################
# ransac function to find the vanishing point #
###############################################
def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_model

###################################################################
# function calculates the distance between input point and origin #
###################################################################
def disToOrigin(x, y):
    dis = np.sqrt(x**2 + y**2)
    return dis

#########################################################################
# main function                                                         #
#                                                                       #
# input: image_paths: list of N elements containing image paths         #
#                                                                       #
# output: orientations: list of N elemenst containing the orientations  #
#########################################################################
def orientation(image_paths):
    orientations = []

    for path in image_paths:

        image = io.imread(path)

        # Compute all edgelets.
        edgelets1 = compute_edgelets(image)
        if edgelets1 == -1:
            orientations.append(-1)
            continue

        num = 0
        for i in range(len(edgelets1[2])):
            if edgelets1[2][i] > 100:
                num += 1
        # print(num)

        if num > 10:
            vp = ransac_vanishing_point(edgelets1, 2000, threshold_inlier=5)
            vp = vp / vp[2]

            # printImgWithVP(image, vp)

            (y, x, _) = np.shape(image)

            x = np.absolute(vp[0] - x / 2)
            y = np.absolute(vp[1] - y / 2)

            # print("x = ", x, " y = ", y)

            if x == 0 or y == 0:
                orientations.append(0)
            else:
                if x > y:
                    cos = y / disToOrigin(x, y)
                    orientations.append(cos)
                else:
                    cos = x / disToOrigin(x, y)
                    orientations.append(cos)
        else:
            orientations.append(-1)
    orientations = np.array(orientations)
    orientations = orientations.reshape(len(orientations),1)
    return orientations

#################################################################
# print out the image with its vanishing point (for debug use)  #
#################################################################
def printImgWithVP(image, vp):
    plt.figure(figsize=(15,10))
    plt.imshow(image, cmap='gray')
    plt.plot(vp[0],vp[1],'+', markersize=120)
    plt.show()

###################################################
#                    Test                         #
# This is for testing the function and            #
# demonstrating how to use the function.          #
###################################################
if __name__ == '__main__':

    paths = []
    paths.append('../data/good/city-buildings-downtown.jpg')
    paths.append('../data/good/woman-wearing-red-sunglasses-2682154.jpg')
    paths.append('../data/bad/autumn-photographer-taking-picture-blur.jpg')
    paths.append('../data/good/person-standing-in-green-grass-terraces-2162133.jpg')
    paths.append('../data/good/people-walking-on-the-street-2506923_1.jpg')

    orientations = orientation(paths)
    print(orientations)
    print("complete")