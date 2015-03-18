#!/usr/bin/env python
"""
ORB Detector and Tracker
========================
Runs an ORB detector and car tracker based on an initial car rectangle position value.
"""

import cv2
import numpy as np

import utilities


def orb(imgs, car_rect=[5, 160, 50, 195], feature_params=dict(nfeatures=40), display_image=True,  write_video=False, video_name='ORB.avi'):
    """
    Tracks a car given an initial rectangle position of the car and displays tracking rectangle and points

    :param imgs: string array containing image file locations
    :param car_rect: Initial position of the car in array format [X min, Y min, X max, Y max]
    :param feature_params: The feature parameters determining the ORB feature detector
    :param display_image: flag determining whether or not to display the image stream
    :param write_video: Flag determining whether or not to record image stream
    :param video_name: File name of image stream
    :return: array of average locations of the car rect
    """

    # Initialize the average position of the rect to an empty array and the index to 0
    ave_pos = []
    frame_idx = 0

    # If write_video flag, record a video
    if write_video:
        height, width, _ = cv2.imread(imgs[0]).shape
        writer = cv2.VideoWriter(video_name, -1, 20, (width, height))

    # Loop through all images
    while frame_idx < len(imgs):

        # Read the image
        frame = cv2.imread(imgs[frame_idx], 0)

        # Initialize ORB feature detector
        orb = cv2.ORB(**feature_params)

        # Detect keypoints in the current image with ORB
        key_points = orb.detect(frame, None)
        key_points, _ = orb.compute(frame, key_points)

        # Initialize empty array that will contain points detected on the car
        car_points = []

        # Loop through all key points detected by ORB and resize the car
        # rectangle based on the points contained by the rectangle
        for p in key_points:
            # Set x and y to integers
            (x, y) = np.int32(p.pt)

            # If the given point is contained within the current car rectangle
            # add it to the car_points array
            if x > car_rect[0] and x < car_rect[2] and y > car_rect[1] and y < car_rect[3]:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                car_points.append((x, y))

        # If there are any points in the car rectangle, resize the rectangle based on those points
        if len(car_points) > 0:
            car_rect = utilities.rect_resize(car_rect, car_points)

        # Append the center position of the car rectangle to the average position array
        ave_pos.append((np.mean([car_rect[0], car_rect[2]]), np.mean([car_rect[1], car_rect[3]])))

        if display_image:
            cv2.rectangle(frame, (car_rect[0], car_rect[1]), (car_rect[2], car_rect[3]), 255)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Write to video
        if write_video:
            writer.write(frame)

        # Update the frame index
        frame_idx += 1

    cv2.destroyAllWindows()
    if write_video:
        writer.release()
    # Return the average position array
    return ave_pos




if __name__ == '__main__':

    # Load images
    imgs = utilities.get_jpeg('./resources/car/')

    # Get average position
    ave_pos = orb(imgs)

    # Plot the average position
    utilities.plot_pixel_position(ave_pos)