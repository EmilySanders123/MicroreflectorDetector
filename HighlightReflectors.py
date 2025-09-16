#!/usr/bin/env python3
import argparse
import json
import cv2
import numpy as np
from CenterpointCalculator import CenterpointCalculator


# get command-line arguments
parser = argparse.ArgumentParser(description="Highlights microreflectors in images of secure text/dendrites")
parser.add_argument("path", help="Path to the image")
parser.add_argument("action", help="Action to perform.  Options: store, match")
args = parser.parse_args()

# calculate and store centerpoints for specified image
calc = CenterpointCalculator()
img_centerpoints = calc.get_centerpoints(args.path, False)

print(img_centerpoints)

# take action indicated by command-line argument
# store point set in database
if args.action.lower() == "store":
    try:
        print("Opening file...")
        with open("StorageJSON.json", "r+") as file:
            print("File found.  Adding new point data...")
            data = json.load(file)
            data["stored_graphs"].append(img_centerpoints)
            file.seek(0)
            json.dump(data, file, indent=4)
        print("Point data added.")
    except FileNotFoundError:
        print("File not found.  Adding point data to newly created file...")
        data = {"stored_graphs": []}
        data["stored_graphs"].append(img_centerpoints)
        with open("StorageJSON.json", "w") as file:
            json.dump(data, file, indent=4)
        print("Point data added.")
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
# try to match point set with set in database
elif args.action.lower() == "match":
    print("Not implemented yet.  I WILL GET TO IT")
else:
    print("Invalid action")