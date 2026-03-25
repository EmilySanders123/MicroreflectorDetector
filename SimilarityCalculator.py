import cv2
import numpy as np
from operator import itemgetter


class SimilarityCalculator:
    point_obj_list = []

    def __init__(self, point_obj_list: list):
        self.point_obj_list = point_obj_list

    def find_best_match(self, norm_img_centerpoints: list) -> int | None:
        ref_stars_percent_list = []
        new_stars_percent_list = []

        for entry_list in self.point_obj_list:
            candidate_points = norm_img_centerpoints.copy()
            matches = []

            # blank image to show which points are matched
            points_img = np.zeros((500, 500, 3), np.uint8)
            points_img[:, :] = [255, 255, 255]

            # check each point in reference constellation
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

                    # check if new point is within 10 pixel radius of reference point and has not already been matched
                    if pow((curr_x - new_x), 2) + pow((curr_y - new_y), 2) <= pow(10,
                                                                                  2) and new_point in candidate_points:
                        # add matched new point to list of matches
                        matches.append((new_x, new_y))

                        # remove new point from list so that they cannot be matched again
                        candidate_points.remove(new_point)

                        # draw matched new point on image
                        cv2.circle(points_img, center=(int(new_x), int(new_y)), radius=2, color=(255, 255, 0),
                                   thickness=-1)

                        # skip to next ref point
                        break

            # draw all unmatched new points
            for unmatched_pt in candidate_points:
                cv2.circle(points_img, center=(int(unmatched_pt[0]), int(unmatched_pt[1])), radius=2, color=(0, 0, 255),
                           thickness=-1)

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