#!/usr/bin/env python
"""
Haar Cascade Classifier
=======================
This module contains the 'cascade' function which will run a Haar classifier
on an image stream and returns the average location of the classifier matches
at each frame in an array.
"""

import cv2
import numpy as np

import utilities


def cascade(imgs, cascade_classifier, display_image=True, write_video=False, video_name='HAAR.avi'):
    """
    Gives the average position of a haar cascade classifier and displays image stream with classifier matches

    :param imgs: string array containing image file locations
    :param cascade_classifier: string containing file location of Haar classifier XML
    :param display_image: flag determining whether or not to display the image stream
    :param write_video: Flag determining whether or not to record image stream
    :param video_name: File name of image stream
    :return: array of average locations of the matches of the Haar classifier
    """

    # Load cascade classifier
    car_cascade = cv2.CascadeClassifier(cascade_classifier)

    # If write_video flag, record a video
    if write_video:
        height, width, _ = cv2.imread(imgs[0]).shape
        writer = cv2.VideoWriter(video_name, -1, 20, (width, height))

    # Initialize the average position to an empty array
    ave_pos = []

    # Initialize frame index
    frame_idx = 0

    # Loop through images, displaying every cascade match in a blue rectangle
    while frame_idx < len(imgs):
        # Load image at current index
        frame = cv2.imread(imgs[frame_idx])

        # Create grayscale image for cascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all classifier matches of grayscale image
        cars = car_cascade.detectMultiScale(gray)

        # Initialize the x,y positions at this frame to empty arrays
        pos_x = []
        pos_y = []

        # Create rectangle for every match
        for (x, y, w, h) in cars:
            # Create a rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            pos_x.append(np.mean([x, x + w]))
            pos_y.append(np.mean([y, y + h]))

        # Only add a new position to the array if there was a classifier match
        if len(cars) > 0:
            ave_pos.append((np.mean(pos_x), np.mean(pos_y)))

        # Display the image if the display image flag is true
        if display_image:
            cv2.imshow('img', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Write to video
        if write_video:
            writer.write(frame)

        # Update the index
        frame_idx += 1

    # Close all windows
    cv2.destroyAllWindows()

    # Release the video
    if write_video:
        writer.release()

    return ave_pos


if __name__ == '__main__':

    # Load images
    imgs = utilities.get_jpeg('./resources/car/')

    # Get average position according to the classifier
    ave_pos = cascade(imgs, './resources/cars3.xml')

    # Plot the average position
    utilities.plot_pixel_position(ave_pos)