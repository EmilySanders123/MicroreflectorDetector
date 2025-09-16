import math
from threading import Thread

import cv2
import numpy as np

class CenterpointCalculator:
    def __init__(self):
        # placeholder
        pass

    def get_centerpoints(self, img_name: str, show_imgs: bool) -> list:
        """
        Finds unique centerpoints of bright points within an image in (x, y) format and returns them as a list ordered
        by the distance from the center of the point cluster.
        :param img_name: Path to image to be processed
        :param show_imgs: Display images of processing steps
        :return: Set of unique centerpoints of bright points within an image
        """
        
        # open image to be processed
        im = cv2.imread(img_name)

        # mask entire image except for brightest points
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        min_diff = 255 - np.array(gray_im).max()
        gray_im += min_diff
        ret, thresh_im = cv2.threshold(gray_im, 155, 255, cv2.THRESH_BINARY)
        # TODO: mess with image brightening values?  change contrast?

        # generate contours from filtered image
        contours, hierarchies = cv2.findContours(thresh_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # find set of centerpoints for each contour
        centerpoints = set()
        for i in contours:
            point = self.__find_contour_center(i)
            if point:
                centerpoints.add(point)

        # find center coordinate of all centerpoints listed
        centerpoints_list = list(centerpoints)
        center_x = 0
        center_y = 0
        for i in range(len(centerpoints_list)):
            center_x += centerpoints_list[i][0]
            center_y += centerpoints_list[i][1]
        center_x //= len(centerpoints_list)
        center_y //= len(centerpoints_list)

        # sort centerpoints list by proximity to center so they are always in the same order
        centerpoints_list.sort(key=lambda p: math.sqrt((p[0] - center_x)**2 * (p[1] - center_y)**2))

        # display images in separate thread so program doesn't pause until they are closed
        if show_imgs:
            thread = Thread(target=self.__display_images, args=(im, thresh_im, centerpoints_list, center_x, center_y))
            thread.start()

        return centerpoints_list

    def __find_contour_center(self, contour: np.ndarray) -> tuple[int, int] | None:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        else:
            return None

    def __display_images(self, og_img: np.ndarray, processed_img: np.ndarray, centerpoints: list, center_x: int, center_y: int) -> None:
        # draw all centerpoints on a new image
        im_height, im_width, _ = og_img.shape
        points_img = np.zeros((im_height, im_width, 3), np.uint8)
        points_img[:, :] = [255, 255, 255]
        for i in centerpoints:
            cv2.circle(points_img, i, 2, (0, 0, 255), -1)

        # make new image with lines drawn between ordered points
        pts = np.array(centerpoints, np.int32)
        shape_img = np.zeros((im_height, im_width, 3), np.uint8)
        cv2.polylines(shape_img, [pts], True, (0, 255, 255))

        # display images
        cv2.imshow("Original image", og_img)
        cv2.imshow("Processed image", processed_img)
        cv2.imshow("Detected points", points_img)
        cv2.imshow("Point shape", shape_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()