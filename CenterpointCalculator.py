from threading import Thread

import cv2
import numpy as np

class CenterpointCalculator:
    def __init__(self):
        # placeholder
        pass

    def get_centerpoints(self, img_name: str) -> set:
        # find image to be processed
        im = cv2.imread(img_name)

        # mask entire image except for brightest points
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        min_diff = 255 - np.array(gray_im).max()
        gray_im += min_diff
        ret, thresh_im = cv2.threshold(gray_im, 155, 255, cv2.THRESH_BINARY)
        # TODO: mess with image brightening?

        # generate contours from filtered image
        contours, hierarchies = cv2.findContours(thresh_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # find set of centerpoints for each contour
        centerpoints = set()
        for i in contours:
            point = self.__find_contour_center(i)
            if point:
                centerpoints.add(point)

        # draw all centerpoints on a new image
        im_height, im_width, _ = im.shape
        points_im = np.zeros((im_height, im_width, 3), np.uint8)
        points_im[:, :] = [255, 255, 255]
        for i in centerpoints:
            cv2.circle(points_im, i, 2, (0, 0, 255), -1)

        # display images in separate thread so program doesn't pause until they are closed
        thread = Thread(target=self.__display_images, args=(im, thresh_im, points_im))
        thread.start()

        return centerpoints

    def __find_contour_center(self, contour: np.ndarray) -> tuple[int, int] | None:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        else:
            return None

    def __display_images(self, og_img: np.ndarray, processed_img: np.ndarray, detected_points: np.ndarray) -> None:
        cv2.imshow("Original image", og_img)
        cv2.imshow("Processed image", processed_img)
        cv2.imshow("Detected points", detected_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()