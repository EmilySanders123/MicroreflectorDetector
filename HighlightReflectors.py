#!/usr/bin/env python3
import argparse
import ast
import json
from operator import itemgetter
from sys import displayhook
from types import new_class

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.bezierTools import segmentPointAtT

from CenterpointCalculator import CenterpointCalculator
from RatioCalculator import RatioCalculator

# get command-line arguments
parser = argparse.ArgumentParser(description="Highlights microreflectors in images of secure text/dendrites")
parser.add_argument("path", help="Path to the image")
parser.add_argument("action", help="Action to perform.  Options: store, match, test")
parser.add_argument("id", help="Unique ID of the image")
args = parser.parse_args()

# calculate and store centerpoints for specified image
calc = CenterpointCalculator()
if args.action == "store" or args.action == "match" or args.action == "display" or args.action == "test_ratio":
    img_centerpoints = calc.get_centerpoints(args.path, False)
else:
    img_centerpoints = calc.get_centerpoints(args.path, True)

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

# calculate ratios between all points
ratioCalc = RatioCalculator()
new_point_ratios = ratioCalc.generate_constellation_ratios(norm_img_centerpoints)

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
                new_point_obj = {"id": args.id, "point_ratios": new_point_ratios}

                data["stored_graphs"].append(new_point_obj)
                print("Point data added.")
            else:
                print("Point with ID already exists in file.")
            file.seek(0)
            json.dump(data, file, indent=4)

    # make new database file if one is not found
    except FileNotFoundError:
        print("File not found.  Adding point data to newly created file...")
        data = {"stored_graphs": []}
        new_point_obj = {"id": args.id, "point_ratios": new_point_ratios}
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

    # store reference points in list
    ref_constellation_obj_list = data["stored_graphs"]

    # stores percentage of reference stars matched and new stars matched, respectively
    ref_stars_percent_list = []
    new_stars_percent_list = []

    # iterate through each entry in the list of stored constellations
    for ref_constellation_entry in ref_constellation_obj_list:
        candidate_new_points = new_point_ratios.copy()
        candidate_ref_points = ref_constellation_entry["point_ratios"].copy()
        matches = []

        # TODO: better drawings
        # blank images to show which points are matched
        new_points_img = np.zeros((500, 500, 3), np.uint8)
        new_points_img[:, :] = [255, 255, 255]
        ref_points_img = np.zeros((500, 500, 3), np.uint8)
        ref_points_img[:, :] = [255, 255, 255]
        separator_bar = np.zeros((500, 5, 3), np.uint8)
        separator_bar[:, :] = [0, 0, 0]

        # check each point in reference constellations
        for ref_ratio_list in ref_constellation_entry["point_ratios"]:
            # compare against each point in new constellation
            for new_ratio_list in candidate_new_points:
                ratio_matches = 0
                for ref_ratio in ref_ratio_list[1:4]:
                        for new_ratio in new_ratio_list[1:4]:
                            # print("Comparing new ratio " + str(new_ratio) + " with ref ratio " + str(ref_ratio))
                            # TODO: tune tolerances
                            if abs(ref_ratio[0] - new_ratio[0]) <= .0005 and abs(ref_ratio[1] - new_ratio[1]) <= .0005:
                                ratio_matches += 1

                if ratio_matches == 3:
                    matches.append(new_ratio_list)
                    candidate_new_points.remove(new_ratio_list)
                    candidate_ref_points.remove(ref_ratio_list)
                    # draw matched new point on image
                    cv2.circle(new_points_img, center=(int(new_ratio_list[0][0]), int(new_ratio_list[0][1])), radius=2, color=(10, 150, 0), thickness=-1)
                    cv2.circle(ref_points_img, center=(int(ref_ratio_list[0][0]), int(ref_ratio_list[0][1])), radius=2, color=(10, 150, 0), thickness=-1)

        for unmatched_pt in candidate_new_points:
            cv2.circle(new_points_img, center=(int(unmatched_pt[0][0]), int(unmatched_pt[0][1])), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.putText(new_points_img, "New points", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

        for unmatched_pt in candidate_ref_points:
            cv2.circle(ref_points_img, center=(int(unmatched_pt[0][0]), int(unmatched_pt[0][1])), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.putText(ref_points_img, "Ref points", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

        double_image = np.concatenate((new_points_img, separator_bar, ref_points_img), axis=1)

        # record overall percentage of new points and reference points that were matched
        percent_ref_stars_matched = len(matches) / len(ref_constellation_entry["point_ratios"]) * 100
        percent_new_stars_matched = len(matches) / len(norm_img_centerpoints) * 100
        ref_stars_percent_list.append((ref_constellation_entry["id"], percent_ref_stars_matched))
        new_stars_percent_list.append((ref_constellation_entry["id"], percent_new_stars_matched))

        print("Point cloud " + ref_constellation_entry["id"] + ":")
        print("Number of new stars: " + str(len(norm_img_centerpoints)))
        print("Number of ref stars: " + str(len(ref_constellation_entry["point_ratios"])))
        print("Matches: " + str(len(matches)))
        print("Percent of new stars matched: " + str(percent_new_stars_matched) + "%")
        print("Percent of reference stars matched: " + str(percent_ref_stars_matched) + "%\n")

        cv2.imshow("Matches found", double_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # the percentages of new stars with a match and reference stars with a match must both be at least 75%
    high_new_match_list = [i for i in new_stars_percent_list if i[1] >= 75]
    if len(high_new_match_list) == 0:
        # there are no matches over 75%
        print("No matches found.")
    else:
        # there is at least one match
        # tiebreak with the highest percent of reference stars matched
        # if that is the same as well, then just pick whichever
        ref_match_list = [ref_percent_tuple for ref_percent_tuple in ref_stars_percent_list if ref_percent_tuple[0] in
                          ([new_percent_tuple[0] for new_percent_tuple in high_new_match_list])]
        max_ref_match = max(ref_match_list, key=itemgetter(1))
        if max_ref_match[1] >= 75:
            print("ID with best match: " + max_ref_match[0])
        else:
            print("No matches found.")

elif args.action.lower() == "display":
    try:
        with open("StorageJSON.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Points file not found.")
        exit(0)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
        exit(0)

    ref_constellation_obj_list = data["stored_graphs"]

    for obj in ref_constellation_obj_list:
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

elif args.action.lower() == "test_ratio":
    ratioCalc = RatioCalculator()
    new_point_ratios = ratioCalc.generate_constellation_ratios(norm_img_centerpoints, draw=True)

        # FORMAT: ref_point_ratios[point] = [(point_x, point_y), (first_dist/second_dist, angle between first and second), (first_dist/third_dist, angle between first and third), (second_dist/third_dist, angle between second and third)]
else:
    print("Invalid action")