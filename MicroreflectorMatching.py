#!/usr/bin/env python3
import argparse
import json

import cv2
import numpy as np

from CenterpointCalculator import CenterpointCalculator
from MatchCalculator import MatchCalculator
from RatioCalculator import RatioCalculator


def read_storage_data():
    try:
        # open storage file
        with open("StorageJSON.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Points file not found.")
        exit(0)
    except json.decoder.JSONDecodeError:
        print("Could not parse JSON file format.  The file may be corrupted.")
        exit(0)


def main():
    # get command-line arguments
    parser = argparse.ArgumentParser(
        description="Identifies, stores, and matches reflective particles embedded in dendritic identifiers and secure text.")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("-s", "--store", help="Store the constellation information of the given image")
    action_group.add_argument("-m", "--match",
                              help="Calculate the constellation of the given image and find the best match in the stored constellations on record")
    action_group.add_argument("-d", "--display",
                              help="Display the selected constellation with matching ID as points on a graph")
    action_group.add_argument("-t", "--test",
                              help="Graphically represent all star generation steps of the given image, from the raw image to the final constellation")
    action_group.add_argument("-tr", "--test_ratio",
                              help="Graphically represent the ratio generation for all stars in the given imagee")
    args_in = parser.parse_args()

    # if display option is chosen, display selected constellation without processing an image, then exit
    if args_in.display:
        data = read_storage_data()

        ref_constellation_obj_list = data["stored_graphs"]

        for obj in ref_constellation_obj_list:
            if obj["id"] == args_in.display:
                # draw all centerpoints on a new image
                display_img = np.zeros((500, 500, 3), np.uint8)
                display_img[:, :] = [255, 255, 255]
                for i in obj["point_ratios"]:
                    cv2.circle(display_img, (int(i[0][0]), int(i[0][1])), 2, (0, 0, 0), -1)

                cv2.imshow("Chosen image", display_img)
                cv2.waitKey(0)

        exit(0)

    # calculate and store centerpoints for specified image
    calc = CenterpointCalculator()
    if args_in.store:
        img_centerpoints = calc.get_centerpoints(args_in.store, False)
    elif args_in.match:
        img_centerpoints = calc.get_centerpoints(args_in.match, False)
    elif args_in.test:
        img_centerpoints = calc.get_centerpoints(args_in.test, True)
    elif args_in.test_ratio:
        img_centerpoints = calc.get_centerpoints(args_in.test_ratio, True)
    else:
        print("You should not be able to get here")
        exit(0)

    x_list = [point[0] for point in img_centerpoints]
    y_list = [point[1] for point in img_centerpoints]
    x_max = max(x_list)
    x_min = min(x_list)
    x_range = x_max - x_min
    y_max = max(y_list)
    y_min = min(y_list)
    y_range = y_max - y_min

    max_range = max(x_range, y_range)

    # normalize x and y points to range between 10 and 490
    x_list_normalized = [(val - x_min) / max_range * 480 + 10 for val in x_list]
    y_list_normalized = [(val - y_min) / max_range * 480 + 10 for val in y_list]
    norm_img_centerpoints = list(zip(x_list_normalized, y_list_normalized))

    # calculate ratios between all points
    ratio_calc = RatioCalculator()
    new_point_ratios = ratio_calc.generate_constellation_ratios(norm_img_centerpoints)

    # end process if too few points were found to create ratios
    if new_point_ratios is None:
        print("Too few points were found.  Please adjust angle and try again.")
        exit(0)

    # take action indicated by command-line argument
    # store point set in database
    if args_in.store:
        try:
            print("Opening file...")
            with open("StorageJSON.json", "r+") as file:
                print("File found.  Adding new point data...")
                data = json.load(file)
                id_list = [item.get("id") for item in data["stored_graphs"]]
                new_point_obj = {"id": str(int(max(id_list)) + 1), "point_ratios": new_point_ratios}

                # add new point to json data
                data["stored_graphs"].append(new_point_obj)

                # put json data into storage file
                file.seek(0)
                json.dump(data, file, indent=4)

                print("Point data added.")

        # make new database file if one is not found
        except FileNotFoundError:
            print("File not found.  Adding point data to newly created file...")
            data = {"stored_graphs": []}
            new_point_obj = {"id": "1", "point_ratios": new_point_ratios}
            data["stored_graphs"].append(new_point_obj)
            with open("StorageJSON.json", "w") as file:
                json.dump(data, file, indent=4)
            print("Point data added.")

        # quits attempt if json format cannot be read
        except json.decoder.JSONDecodeError:
            print("Could not parse JSON file format.  The file may be corrupted.")

    # try to match point set with set in database
    elif args_in.match:
        # read data from storage file
        data = read_storage_data()

        # store reference points in list
        ref_constellation_obj_list = data["stored_graphs"]

        # find the stored constellation with the highest match percentage
        match_calculator = MatchCalculator(ref_constellation_obj_list)
        match_id = match_calculator.find_matches(norm_img_centerpoints, new_point_ratios)

        # print out matching id or lack thereof
        if match_id is not None:
            print("ID of closest match found: " + match_id)
        else:
            print("No matches found.")

    elif args_in.test:
        pass

    # display ratio calculations for debugging purposes
    elif args_in.test_ratio:
        ratio_calc = RatioCalculator()
        new_point_ratios = ratio_calc.generate_constellation_ratios(norm_img_centerpoints, draw=True)

        # end process if too few points were found to create ratios
        if new_point_ratios is None:
            print("Too few points were found.  Please adjust angle and try again.")
            exit(0)

    # invalid action, should not be possible
    else:
        print("Invalid action")


if __name__ == "__main__":
    main()