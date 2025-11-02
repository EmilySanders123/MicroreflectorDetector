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
        selected_contours = []
        for i in contours:
            if cv2.arcLength(i, True) < 50:
                point = self.__find_contour_center(i)
                if point:
                    centerpoints.add(point)
                    selected_contours.append(i)

        # display images in separate thread so program doesn't pause until they are closed
        if show_imgs:
            thread = Thread(target=self.__display_images, args=(im, thresh_im, tuple(selected_contours), centerpoints))
            thread.start()

        return list(centerpoints)

    def __find_contour_center(self, contour: np.ndarray) -> tuple[int, int] | None:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        else:
            return None

    def __display_images(self, og_img: np.ndarray, processed_img: np.ndarray, contours: tuple, centerpoints: list) -> None:
        # draw contours on image
        im_height, im_width, _ = og_img.shape
        contour_img = np.zeros((im_height, im_width, 3), np.uint8)
        contour_img[:, :] = [255, 255, 255]
        cv2.drawContours(contour_img, contours, -1, (0, 0, 0), 3)

        # draw all centerpoints on a new image
        points_img = np.zeros((im_height, im_width, 3), np.uint8)
        points_img[:, :] = [255, 255, 255]
        marked_og_img = og_img.copy()
        for i in centerpoints:
            cv2.circle(points_img, i, 2, (0, 0, 255), -1)
            cv2.circle(marked_og_img, i, 2, (0, 0, 255), -1)

        # display images
        cv2.imshow("Original image", og_img)
        cv2.imshow("Processed image", processed_img)
        cv2.imshow('Selected contours', contour_img)
        cv2.imshow("Detected points", points_img)
        cv2.imshow("Marked-up original image", marked_og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()