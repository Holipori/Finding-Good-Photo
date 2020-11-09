import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from utils import get_image_paths
from utils import show_results
import os.path as osp

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################


  for i in range(len(image_paths)):
    img = load_image_gray(image_paths[i])
    frames, descriptors = vlfeat.sift.dsift(img, step = 3, fast = True)
    if i == 0:
      descList = descriptors
    else:
      descList = np.concatenate((descList,descriptors),axis = 0)
  # descList = np.array(descList)
  vocab = vlfeat.kmeans.kmeans(descList.astype(np.float32), vocab_size)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
          (but possibly used for extra credit in get_bags_of_sifts if you're
          making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  # dummy features variable
  feats = np.zeros((len(image_paths), len(vocab)))

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  for i in range(len(image_paths)):
    img = load_image_gray(image_paths[i])
    frames, descriptors = vlfeat.sift.dsift(img, step = 3, fast = True)
    assignments = vlfeat.kmeans.kmeans_quantize(descriptors.astype(np.float32), vocab.astype(np.float32))
    for j in range(len(assignments)):
      feats[i, assignments[j]] += 1
    feats[i] = (feats[i] - np.mean(feats[i]) )/np.std(feats[i], ddof = 1)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats

def svm_classify(train_image_feats, train_labels, test_image_feats):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  # categories
  categories = list(set(train_labels))

  # construct 1 vs all SVMs for each category
  svms = {cat: LinearSVC(dual = False, penalty = 'l1', random_state=0, tol=10, loss='squared_hinge', C=1,
                         fit_intercept = False) for cat in categories}
  # svms = {cat: SVC(random_state=0, tol=1e-0,  C=0.5, kernel='sigmoid') for cat in categories}

  test_labels = []
  scores = []
  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  print ("train labels size: ", len(train_labels), "\n")
  print ("test feats size: ", len(test_image_feats), "\n")

  test_score_list =np.zeros((len(categories),len(test_image_feats)))
  for cat in svms:
    train_labels_cat = []
    for i in range(len(train_labels)): # making train_labels_cat
      if train_labels[i] == cat:
        train_labels_cat.append(1)
      else:
        train_labels_cat.append(0)
    svms.get(cat).fit(train_image_feats, train_labels_cat) # fit
    for i in range(len(categories)):# compute the scores for each 1 vs all SVMs.
      if categories[i] == cat:
        test_score_list[i] = np.array([svms.get(cat).decision_function(test_image_feats)])
        break
  for i in range(test_score_list.shape[1]): # choose the label that has highest score
    print(test_score_list[:, i])
    sort = np.argsort(test_score_list[:, i])
    print(sort)
    test_labels.append(categories[sort[-1]])

  clf = LinearSVC(dual = False, penalty = 'l1', random_state=0, tol=1e-0, loss='squared_hinge', C=1, # compute cross-validation score
                         fit_intercept = False)
  scores = cross_val_score(clf, train_image_feats, train_labels, cv=5)
  stddiv = np.std(scores) # compute standard deviations
  #print(scores)
  #print(stddiv)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return test_labels

def findFace(path):
  # Load the cascade
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  # Read the input image
  img = cv2.imread(path)
  # Convert into grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detect faces
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)

  if faces is None:
    print ("No face in image: ", path, "\n")
    return gray/255
  else:
    for (x, y, w, h) in faces:
      xa = x  # setting the corner position of human face
      ya = y
      xb = x + w
      yb = y + h
      cv2.rectangle(img, (xa, ya), (xb, yb), (255, 0, 0), 2)
      crop_img = gray[ya:yb, xa:xb]  # crop the face
      crop_img = np.array(crop_img)
      crop_img = cv2.resize(crop_img, (48, 48))  # resize the face
      crop_img = np.array(crop_img) / 255
      break
    print(np.shape(crop_img))
    return crop_img


if  __name__ == '__main__':
  # make categories for use
  categories = ['Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Surprise', 'Neutral'];
  abbr_categories = ['Hap', 'Ang', 'Dis', 'Fea', 'Sad', 'Sur', 'Neu']

  # number of images needed from each category
  num_train_per_cat = 100

  #    get train image paths
  data_path = osp.join('..', 'data', 'expressions')
  train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, categories,
                                                                                   num_train_per_cat);
  print("number of train images: ", len(train_image_paths))
  print("number of train labels: ", len(train_labels))
  print("number of test images: ", len(test_image_paths))
  print("number of test labels: ", len(test_labels))

  #    build the vocabulary
  vocab_filename = 'vocab_50.pkl'
  if not osp.isfile(vocab_filename):
    # Construct the vocabulary
    print('No existing visual word vocabulary found. Computing one from training images')
    vocab_size = 50  # Larger values will work better (to a point) but be slower to compute
    vocab = build_vocabulary(train_image_paths, vocab_size)
    with open(vocab_filename, 'wb') as f:
      pickle.dump(vocab, f)
      print('{:s} saved'.format(vocab_filename))

  #    get bags of sifts features
  train_image_feats = get_bags_of_sifts(train_image_paths, vocab_filename)
  test_image_feats = get_bags_of_sifts(test_image_paths, vocab_filename)

  #    return the list of scores of test images, and expressions
  predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

  y = 0
  count = 0
  for x in test_labels:
    if x == predicted_categories[y]:
      count = count + 1
    y = y + 1

  print("accuracy: ", count / len(test_labels))

  # save the confusion map of the svm test results
  show_results(train_image_paths, test_image_paths, train_labels, test_labels,
               categories, abbr_categories, predicted_categories)
