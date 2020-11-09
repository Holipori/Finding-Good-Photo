import cv2
import os
from tqdm import tqdm

##########################################################################
# This file is for data augmentation.
# we only do horizontally flip our dataset here.
# Because our dataset is about good photos,
# we can't implement other augmentation methods like rotation, zoom, etc.
# that will influence the quality of the photo
##########################################################################
path = os.path.abspath('../data2/my')
for root, dirs, files in os.walk(path):
    for file in tqdm(files):
        src_file = os.path.join(root, file)
        img = cv2.imread(src_file) # read dataset
        img = cv2.flip(img, 1)
        dst_file = file[:-4]+ '-flip.jpg'
        dst = os.path.join(root, dst_file)
        cv2.imwrite(dst, img)  # write flipped dataset

