#!/usr/bin/env python3
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from CenterpointCalculator import CenterpointCalculator


# get command-line arguments
parser = argparse.ArgumentParser(description="Highlights microreflectors in images of secure text/dendrites")
parser.add_argument("path", help="Path to the image")
parser.add_argument("action", help="Action to perform.  Options: store, match, test")
parser.add_argument("id", help="Unique ID of the image")
args = parser.parse_args()

# calculate and store centerpoints for specified image
calc = CenterpointCalculator()
if args.action == "store" or args.action == "match":
    img_centerpoints = calc.get_centerpoints(args.path, False)
else:
    img_centerpoints = calc.get_centerpoints(args.path, True)

# normalize x points
x_list = [point[0] for point in img_centerpoints]
x_max = max(x_list)
x_min = min(x_list)
x_list_normalized = [(val - x_min) / (x_max - x_min) * 480 + 10 for val in x_list]
# normalize y points
y_list = [point[1] for point in img_centerpoints]
y_max = max(y_list)
y_min = min(y_list)
y_list_normalized = [(val - y_min) / (y_max - y_min) * 480 + 10 for val in y_list]
norm_img_centerpoints = list(zip(x_list_normalized, y_list_normalized))

# take action indicated by command-line argument
# store point set in database
# TODO: store more info
if args.action.lower() == "store":
    try:
        print("Opening file...")
        with open("StorageJSON.json", "r+") as file:
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
        with open("StorageJSON.json", "w") as file:
            json.dump(data, file, indent=4)
        print("Point data added.")
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")

# try to match point set with set in database
elif args.action.lower() == "match":
    try:
        with open("StorageJSON.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Points file not found.")
        exit(0)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
        exit(0)

    point_obj_list = data["stored_graphs"]

    # create image of new points
    pts = np.array(norm_img_centerpoints, np.int32)
    new_img = np.zeros((500, 500, 3), np.uint8)
    cv2.polylines(new_img, [pts], True, (0, 255, 255))

    orb = cv2.ORB_create(nfeatures=5000)

    avg_distance_dict = {}
    for obj in point_obj_list:

        # make new image with lines drawn between ordered points
        pts = np.array(obj["points"], np.int32)
        shape_img = np.zeros((500, 500, 3), np.uint8)
        cv2.polylines(shape_img, [pts], True, (0, 255, 255))
        # shape_img = cv2.rotate(shape_img, cv2.ROTATE_180)
        cv2.imshow(f"Stored point ID: {obj["id"]}", shape_img)

        # compute matches between images
        new_keypoints, new_descriptors = orb.detectAndCompute(new_img, None)
        stored_keypoints, stored_descriptors = orb.detectAndCompute(shape_img, None)
        matcher = cv2.BFMatcher()
        matches = matcher.match(new_descriptors, stored_descriptors)

        avg_distance = 0
        for match in matches:
            avg_distance += match.distance

        avg_distance = avg_distance / len(matches)

        avg_distance_dict[obj["id"]] = avg_distance

        # show matches
        final_img = cv2.drawMatches(new_img, new_keypoints, shape_img, stored_keypoints, matches[:20], None)
        final_img = cv2.resize(final_img, (1000, 650))

        cv2.imshow(f"Comparison with ID {obj["id"]}", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(avg_distance_dict)
    print(f"Closest match: {min(avg_distance_dict, key=avg_distance_dict.get)}")

elif args.action.lower() == "test":
    pass
else:
    print("Invalid action")