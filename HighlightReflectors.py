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

# mask entire image except for brightest points
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh_im = cv2.threshold(gray_im, 125, 255, cv2.THRESH_BINARY)
# TODO: normalize image colors so brightest color is always pure white?

# generate contours from filtered image
contours, hierarchies = cv2.findContours(thresh_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# find set of centerpoints for each contour
centerpoints = set()
for i in contours:
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centerpoints.add((cx, cy))

# draw all centerpoints on a new image
im_height, im_width, _ = im.shape
points_im = np.zeros((im_height, im_width, 3), np.uint8)
points_im[:,:] = [255, 255, 255]
for i in centerpoints:
    cv2.circle(points_im, i, 2, (0, 0, 255), -1)

# display images
cv2.imshow("Original image", im)
cv2.imshow("Processed image", thresh_im)
cv2.imshow("Detected points", points_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: make this its own class