#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
from CenterpointCalculator import CenterpointCalculator


# get command-line arguments
parser = argparse.ArgumentParser(description="Highlights microreflectors in images of secure text/dendrites")
parser.add_argument("path", help="Path to the image")
args = parser.parse_args()

# calculate and store centerpoints for specified image
calc = CenterpointCalculator()
img_centerpoints = calc.get_centerpoints(args.path)

print(img_centerpoints)