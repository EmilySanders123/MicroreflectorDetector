import numpy as np


class RatioCalculator:
    def __init__(self):
        # placeholder
        pass

    def closest_three_points(self, point_index: int, point_list: list[tuple[int, int]]):
        curr_point = point_list[point_index]
        print("point analyzed: " + str(curr_point))
        np_norm_img_centerpoints = np.array(point_list)
        np_point = np.array(curr_point)
        distances = np.linalg.norm(np_norm_img_centerpoints - np_point, axis=1)

        # TODO: account for < 4 stars total
        # remove current ref point from list (always has same index as current point)
        np_norm_img_centerpoints = np.delete(np_norm_img_centerpoints, point_index, axis=0)
        distances = np.delete(distances, point_index)

        # save and then remove first closest point
        closest_index = np.argmin(distances)
        closest_point = (float(np_norm_img_centerpoints[closest_index][0]), float(np_norm_img_centerpoints[closest_index][1]))
        closest_distance = distances[closest_index]
        np_norm_img_centerpoints = np.delete(np_norm_img_centerpoints, closest_index, axis=0)
        distances = np.delete(distances, closest_index)

        # save and then remove second closest point
        second_closest_index = np.argmin(distances)
        second_closest_point = (float(np_norm_img_centerpoints[second_closest_index][0]), float(np_norm_img_centerpoints[second_closest_index][1]))
        second_closest_distance = distances[second_closest_index]
        np_norm_img_centerpoints = np.delete(np_norm_img_centerpoints, second_closest_index, axis=0)
        distances = np.delete(distances, second_closest_index)

        # save third closest point
        third_closest_index = np.argmin(distances)
        third_closest_point = (float(np_norm_img_centerpoints[third_closest_index][0]), float(np_norm_img_centerpoints[third_closest_index][1]))
        third_closest_distance = distances[third_closest_index]

        return [closest_point, second_closest_point, third_closest_point]