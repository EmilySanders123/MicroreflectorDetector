#!/usr/bin/env python3
import argparse
import json
from operator import itemgetter
import cv2
import numpy as np
from CenterpointCalculator import CenterpointCalculator
from SimilarityCalculator import SimilarityCalculator


# get command-line arguments
parser = argparse.ArgumentParser(description="Highlights microreflectors in images of secure text/dendrites")
parser.add_argument("path", help="Path to the image")
parser.add_argument("action", help="Action to perform.  Options: store, match, test")
parser.add_argument("id", help="Unique ID of the image")
args = parser.parse_args()

# calculate and store centerpoints for specified image
calc = CenterpointCalculator()
if args.action == "test":
    img_centerpoints = calc.get_centerpoints(args.path, True)
else:
    img_centerpoints = calc.get_centerpoints(args.path, False)

x_list = [point[0] for point in img_centerpoints]
y_list = [point[1] for point in img_centerpoints]
x_max = max(x_list)
x_min = min(x_list)
x_range = x_max - x_min
y_max = max(y_list)
y_min = min(y_list)
y_range = y_max - y_min

max_range = max(x_range, y_range)

# normalize x and y points according to smaller range
x_list_normalized = [(val - x_min) / max_range * 480 + 10 for val in x_list]
y_list_normalized = [(val - y_min) / max_range * 480 + 10 for val in y_list]
norm_img_centerpoints = list(zip(x_list_normalized, y_list_normalized))

# take action indicated by command-line argument
# store point set in database
# TODO: store more info
if args.action.lower() == "store":
    try:
        print("Opening file...")
        with open("StorageJSONOldFormat.json", "r+") as file:
            print("File found.  Adding new point data...")
            data = json.load(file)
            id_list = [item.get("id") for item in data["stored_graphs"]]
            if args.id not in id_list:
                new_point_obj = {"id": args.id, "points": norm_img_centerpoints}
                data["stored_graphs"].append(new_point_obj)
                print("Point data added.")
            else:
                print("Point with ID already exists in file.")
            file.seek(0)
            json.dump(data, file, indent=4)

    except FileNotFoundError:
        print("File not found.  Adding point data to newly created file...")
        data = {"stored_graphs": []}
        new_point_obj = {"id": args.id, "points": norm_img_centerpoints}
        data["stored_graphs"].append(new_point_obj)
        with open("StorageJSONOldFormat.json", "w") as file:
            json.dump(data, file, indent=4)
        print("Point data added.")
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")

# try to match point set with set in database
elif args.action.lower() == "match":
    try:
        # read data from file
        with open("StorageJSONOldFormat.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Points file not found.")
        exit(0)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
        exit(0)

    simCalc = SimilarityCalculator(data["stored_graphs"])

    best_match_id = simCalc.find_best_match(norm_img_centerpoints)

    if best_match_id is None:
        print("No matches found.")
    else:
        print("ID with best match: " + str(best_match_id))

elif args.action.lower() == "display":
    try:
        with open("StorageJSONOldFormat.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Points file not found.")
        exit(0)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
        exit(0)

    point_obj_list = data["stored_graphs"]

    for obj in point_obj_list:
        if obj["id"] == args.id:
            # draw all centerpoints on a new image
            display_img = np.zeros((500, 500, 3), np.uint8)
            display_img[:, :] = [255, 255, 255]
            for i in obj["points"]:
                print(i)
                cv2.circle(display_img, (int(i[0]), int(i[1])), 2, (0, 0, 255), -1)

            cv2.imshow("Chosen image", display_img)
            cv2.waitKey(0)
            exit(0)

    print("Entry with id " + args.id + " not found.")

elif args.action.lower() == "test":
    pass
else:
    print("Invalid action")