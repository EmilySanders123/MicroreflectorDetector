#!/usr/bin/env python3
import argparse
import json
from operator import itemgetter
import cv2
import numpy as np
from CenterpointCalculator import CenterpointCalculator


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
        # read data from file
        with open("StorageJSON.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Points file not found.")
        exit(0)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
        exit(0)

    # store reference points in list
    point_obj_list = data["stored_graphs"]

    # stores percentage of reference stars matched and new stars matched, respectively
    ref_stars_percent_list = []
    new_stars_percent_list = []

    # iterate through each entry in the list of stored constellations
    for i, entry_list in enumerate(point_obj_list):
        candidate_points = norm_img_centerpoints.copy()
        matches = []

        # blank image to show which points are matched
        points_img = np.zeros((500, 500, 3), np.uint8)
        points_img[:, :] = [255, 255, 255]

        # check each point in reference constellation
        ref_list = entry_list["points"].copy()
        for ref_point in entry_list["points"]:
            curr_x = float(ref_point[0])
            curr_y = float(ref_point[1])

            # draw circles around reference point
            cv2.circle(points_img, center=(int(curr_x), int(curr_y)), radius=2, color=(0, 0, 0), thickness=-1)
            cv2.circle(points_img, center=(int(curr_x), int(curr_y)), radius=10, color=(0, 255, 0), thickness=1)

            # check if each new point matches
            for new_point in norm_img_centerpoints:
                new_x = float(new_point[0])
                new_y = float(new_point[1])

                # check if new point is within 10 pixel radius of reference point
                if pow((curr_x - new_x), 2) + pow((curr_y - new_y), 2) <= pow(10, 2):
                    # add matched new point to list of matches
                    matches.append((new_x, new_y))

                    # remove both new and reference points from lists so that they cannot be matched again
                    candidate_points.remove(new_point)
                    ref_list.remove(ref_point)

                    # draw matched new point on image
                    cv2.circle(points_img, center=(int(new_x), int(new_y)), radius=2, color=(255, 255, 0), thickness=-1)

                    # skip to next ref point
                    break

        for unmatched_pt in candidate_points:
            cv2.circle(points_img, center=(int(unmatched_pt[0]), int(unmatched_pt[1])), radius=2, color=(0, 0, 255), thickness=-1)

        # record overall percentage of new points and reference points that were matched
        percent_ref_stars_matched = len(matches) / len(entry_list["points"]) * 100
        percent_new_stars_matched = len(matches) / len(norm_img_centerpoints) * 100
        ref_stars_percent_list.append((entry_list["id"], percent_ref_stars_matched))
        new_stars_percent_list.append((entry_list["id"], percent_new_stars_matched))

        print("Point cloud " + entry_list["id"] + ":")
        print("Matches: " + str(len(matches)))
        print("Percent of new stars matched: " + str(percent_new_stars_matched) + "%")
        print("Percent of reference stars matched: " + str(percent_ref_stars_matched) + "%\n")

        cv2.imshow("Matches found", points_img)
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

elif args.action.lower() == "test":
    pass
else:
    print("Invalid action")