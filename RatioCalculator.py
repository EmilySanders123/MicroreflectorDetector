import math

import cv2
import numpy as np


class RatioCalculator:

    def generate_constellation_ratios(self, centerpoints_list: list[tuple[int, int]], draw=False) -> list | None:
        # kick out if less than 4 points found (unlikely to happen)
        if len(centerpoints_list) < 4:
            return None

        ref_point_neighbors = {}
        ref_point_ratios = []
        for point_index, point in enumerate(centerpoints_list):
            ref_point_neighbors[point] = self.__closest_three_points(point_index, centerpoints_list)

            ref_point_ratios.append(self.__generate_point_ratios(point, ref_point_neighbors[point]))

        if draw:
            # display closest three points to each point
            for point, neighbors in ref_point_neighbors.items():
                display_img = np.zeros((500, 500, 3), np.uint8)
                display_img[:, :] = [255, 255, 255]
                for all_point in centerpoints_list:
                    cv2.circle(display_img, (int(all_point[0]), int(all_point[1])), 3, (0, 0, 0), -1)

                for neighbor_point in neighbors:
                    cv2.line(display_img, (int(point[0]), int(point[1])),
                             (int(neighbor_point[0]), int(neighbor_point[1])), (0, 0, 0), 1)
                    cv2.circle(display_img, (int(neighbor_point[0]), int(neighbor_point[1])), 3, (0, 150, 15), -1)

                cv2.circle(display_img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

                cv2.imshow("Closest points to " + str(point), display_img)
                cv2.waitKey(0)

        return ref_point_ratios

    def __closest_three_points(self, point_index: int, point_list: list[tuple[int, int]]) -> list[tuple[float, float]]:
        curr_point = point_list[point_index]
        np_norm_img_centerpoints = np.array(point_list)
        np_point = np.array(curr_point)
        distances = np.linalg.norm(np_norm_img_centerpoints - np_point, axis=1)

        # remove current ref point from list (always has same index as current point)
        np_norm_img_centerpoints = np.delete(np_norm_img_centerpoints, point_index, axis=0)
        distances = np.delete(distances, point_index)

        # save and then remove first-closest point
        closest_index = np.argmin(distances)
        closest_point = (float(np_norm_img_centerpoints[closest_index][0]), float(np_norm_img_centerpoints[closest_index][1]))
        np_norm_img_centerpoints = np.delete(np_norm_img_centerpoints, closest_index, axis=0)
        distances = np.delete(distances, closest_index)

        # save and then remove second-closest point
        second_closest_index = np.argmin(distances)
        second_closest_point = (float(np_norm_img_centerpoints[second_closest_index][0]), float(np_norm_img_centerpoints[second_closest_index][1]))
        np_norm_img_centerpoints = np.delete(np_norm_img_centerpoints, second_closest_index, axis=0)
        distances = np.delete(distances, second_closest_index)

        # save third-closest point
        third_closest_index = np.argmin(distances)
        third_closest_point = (float(np_norm_img_centerpoints[third_closest_index][0]), float(np_norm_img_centerpoints[third_closest_index][1]))

        return [closest_point, second_closest_point, third_closest_point]

    def __generate_point_ratios(self, centerpoint: tuple[float, float], neighbors: list[tuple[float, float]]):
        center_to_first_dist = math.sqrt((neighbors[0][0] - centerpoint[0]) ** 2 + (neighbors[0][1] - centerpoint[1]) ** 2)
        center_to_second_dist = math.sqrt((neighbors[1][0] - centerpoint[0]) ** 2 + (neighbors[1][1] - centerpoint[1]) ** 2)
        center_to_third_dist = math.sqrt((neighbors[2][0] - centerpoint[0]) ** 2 + (neighbors[2][1] - centerpoint[1]) ** 2)
        first_to_second_dist = math.sqrt((neighbors[0][0] - neighbors[1][0]) ** 2 + (neighbors[0][1] - neighbors[1][1]) ** 2)
        first_to_third_dist = math.sqrt((neighbors[0][0] - neighbors[2][0]) ** 2 + (neighbors[0][1] - neighbors[2][1]) ** 2)
        second_to_third_dist = math.sqrt((neighbors[1][0] - neighbors[2][0]) ** 2 + (neighbors[1][1] - neighbors[2][1]) ** 2)

        first_over_second_dist = center_to_first_dist / center_to_second_dist
        first_over_third_dist = center_to_first_dist / center_to_third_dist
        second_over_third_dist = center_to_second_dist / center_to_third_dist

        first_to_second_angle = math.acos((center_to_first_dist ** 2 + center_to_second_dist ** 2 - first_to_second_dist ** 2) / (2 * center_to_first_dist * center_to_second_dist))
        first_to_third_angle = math.acos((center_to_first_dist ** 2 + center_to_third_dist ** 2 - first_to_third_dist ** 2) / (2 * center_to_first_dist * center_to_third_dist))
        second_to_third_angle = math.acos((center_to_second_dist ** 2 + center_to_third_dist ** 2 - second_to_third_dist ** 2) / (2 * center_to_second_dist * center_to_third_dist))

        # FORMAT: ref_point_ratios[point] = [(point_x, point_y), (first_dist/second_dist, angle between first and second), (first_dist/third_dist, angle between first and third), (second_dist/third_dist, angle between second and third)]
        return [centerpoint, (first_over_second_dist, first_to_second_angle), (first_over_third_dist, first_to_third_angle), (second_over_third_dist,  second_to_third_angle)]