from operator import itemgetter

import cv2
import numpy as np


class MatchCalculator:
    ref_constellation_obj_list = None

    def __init__(self, ref_constellation_obj_list: list):
        self.ref_constellation_obj_list = ref_constellation_obj_list

    def find_matches(self, norm_img_centerpoints: list, new_point_ratios: list, debug=False) -> str | None:
        # stores percentage of reference stars matched and new stars matched, respectively
        ref_stars_percent_list = []
        new_stars_percent_list = []

        # iterate through each entry in the list of stored constellations
        for ref_constellation_entry in self.ref_constellation_obj_list:
            candidate_new_points = new_point_ratios.copy()
            candidate_ref_points = ref_constellation_entry["point_ratios"].copy()
            matches = []

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
                            if abs(ref_ratio[0] - new_ratio[0]) <= 0.08 and abs(ref_ratio[1] - new_ratio[1]) <= 0.08:
                                ratio_matches += 1

                    # if all three ratios match, then points are a positive match
                    if ratio_matches == 3:
                        matches.append(new_ratio_list)
                        candidate_new_points.remove(new_ratio_list)
                        candidate_ref_points.remove(ref_ratio_list)
                        # draw matched new point on image
                        cv2.circle(new_points_img, center=(int(new_ratio_list[0][0]), int(new_ratio_list[0][1])),
                                   radius=2, color=(10, 150, 0), thickness=-1)
                        cv2.circle(ref_points_img, center=(int(ref_ratio_list[0][0]), int(ref_ratio_list[0][1])),
                                   radius=2, color=(10, 150, 0), thickness=-1)

                        # if all three points are matched, stop comparing new ratio points because we don't need any more
                        break

            # draw unmatched points on new point image
            for unmatched_pt in candidate_new_points:
                cv2.circle(new_points_img, center=(int(unmatched_pt[0][0]), int(unmatched_pt[0][1])), radius=2,
                           color=(0, 0, 255), thickness=-1)
            cv2.putText(new_points_img, "New points", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

            # draw unmatched points on ref points image
            for unmatched_pt in candidate_ref_points:
                cv2.circle(ref_points_img, center=(int(unmatched_pt[0][0]), int(unmatched_pt[0][1])), radius=2,
                           color=(0, 0, 255), thickness=-1)
            cv2.putText(ref_points_img, "Ref points", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

            # print coordinates next to each point if debugging
            if debug:
                for point in ref_constellation_entry["point_ratios"]:
                    coords = "(" + str(int(point[0][0])) + ", " + str(int(point[0][1])) + ")"
                    cv2.putText(ref_points_img, coords, (int(point[0][0]) + 3, int(point[0][1]) + 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

                for point in new_point_ratios:
                    coords = "(" + str(int(point[0][0])) + ", " + str(int(point[0][1])) + ")"
                    cv2.putText(new_points_img, coords, (int(point[0][0]) + 3, int(point[0][1]) + 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

            # stick all images together
            double_image = np.concatenate((new_points_img, separator_bar, ref_points_img), axis=1)

            # record overall percentage of new points and reference points that were matched
            percent_ref_stars_matched = len(matches) / len(ref_constellation_entry["point_ratios"]) * 100
            percent_new_stars_matched = len(matches) / len(norm_img_centerpoints) * 100
            ref_stars_percent_list.append((ref_constellation_entry["id"], percent_ref_stars_matched))
            new_stars_percent_list.append((ref_constellation_entry["id"], percent_new_stars_matched))

            # print info about match attempt
            print("Point cloud " + ref_constellation_entry["id"] + ":")
            print("Number of new stars: " + str(len(norm_img_centerpoints)))
            print("Number of ref stars: " + str(len(ref_constellation_entry["point_ratios"])))
            print("Matches: " + str(len(matches)))
            print("Percent of new stars matched: " + str(percent_new_stars_matched) + "%")
            print("Percent of reference stars matched: " + str(percent_ref_stars_matched) + "%\n")

            # show results of matching attempt as image
            cv2.imshow("Matches found", double_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # the percentages of new stars with a match and reference stars with a match must both be at least 75%
        high_new_match_list = [i for i in new_stars_percent_list if i[1] >= 75]
        if len(high_new_match_list) == 0:
            # there are no matches over 75%
            return None
        else:
            # there is at least one match
            # tiebreak with the highest percent of reference stars matched
            # if that is the same as well, then just pick whichever
            ref_match_list = [ref_percent_tuple for ref_percent_tuple in ref_stars_percent_list if
                              ref_percent_tuple[0] in
                              ([new_percent_tuple[0] for new_percent_tuple in high_new_match_list])]
            max_ref_match = max(ref_match_list, key=itemgetter(1))
            if max_ref_match[1] >= 75:
                return max_ref_match[0]
            else:
                return None