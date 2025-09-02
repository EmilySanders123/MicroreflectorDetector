#!/usr/bin/env python3
import argparse

import cv2
import numpy as np

# get command-line arguments
parser = argparse.ArgumentParser(description="Highlights microreflectors in images of secure text/dendrites")
parser.add_argument("path", help="Path to the image")
args = parser.parse_args()

# find image to be processed
im = cv2.imread(args.path)

# get all points that meet the reflective white threshold
min_white = [150, 150, 150]
Y, X = np.where(np.all(im >= min_white, axis=2))
white_coord_pairs = np.column_stack((X,Y))
print(white_coord_pairs)

# make copy of image to write on
marked_up_image = im.copy()

# color white pixels
for pair in white_coord_pairs:
    marked_up_image[pair[1], pair[0]] = (5, 5, 255)

# display images
cv2.imshow("Original image", im)
cv2.imshow("Marked up image", marked_up_image)
cv2.waitKey(0)
cv2.destroyAllWindows()