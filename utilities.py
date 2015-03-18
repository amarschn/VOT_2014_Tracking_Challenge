#!/usr/bin/env python
"""
Utilities
===========
This module contains multiple utility functions:
 get_jpeg => will load jpeg images and return them as an array.
 plot_pixel_position => will plot pixel location given an array.
 rect_resize => will resize a rectangle given an array of points.
"""

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def get_jpeg(path):
    """
    Returns all JPEG files given a path
    :param path:
    """
    image_names = []
    for f in os.listdir(path):
        if f.endswith(".jpg"):
            image_names.append(path + f)
    return image_names


def plot_pixel_position(pos):
    """
    :param pos:
    :return:
    """
    pos = np.array(pos)
    plt.plot(pos[:, 0])
    plt.plot(pos[:, 1])
    plt.legend(['X pixel location', 'Y pixel location'])
    plt.show()

def rect_resize(rect, points, buffer=40):
    """
    Resizes a rectangle based on given points, will grow the rectangle by a buffered amount given the max and min values
    of the point array

    :param rect: rectangle containing
    :param points:
    :param buffer:
    :return:
    """

    # Get the min and max x and y positions
    min_point = min(points)
    max_point = max(points)

    # Define the max and min of x and y to be added or subtracted the buffer, respectively
    min_x = min_point[0] - buffer
    max_x = max_point[0] + buffer
    min_y = min_point[1] - buffer
    max_y = max_point[1] + buffer

    # Grow the rectangle by the mean of the rectangles current position and the mean of the min and max x and y
    #  positions. This keeps the rectangle from re-sizing drastically every frame due to changing feature points
    rect[0] = np.mean([rect[0], min_x])
    rect[1] = np.mean([rect[1], min_y])
    rect[2] = np.mean([rect[2], max_x])
    rect[3] = np.mean([rect[3], max_y])

    # Return the new rectangle
    return np.int32(rect)


if __name__ == '__main__':

    # Load images into array
    imgs = get_jpeg('C:/Users/Drew/Dropbox/Uber_Assignment/uber_cv_car_exercise/car/')

    # Begin video capture of images
    cap = cv2.VideoCapture(imgs[0])
    idx = 0

    # Loop through images, end once at the end of the image array
    while idx < len(imgs):
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Update the index
        idx += 1

    # Close all windows
    cv2.destroyAllWindows()



