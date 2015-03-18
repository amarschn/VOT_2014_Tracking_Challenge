#!/usr/bin/env python
"""
Lucas-Kanade Optical Flow
=========================
Runs LK OF algorithm to update bounding rectangle of the car
"""

import cv2
import numpy as np

import utilities


def lk_optical_flow(imgs, detect_interval=5, car_rect=[5, 160, 50, 195], display_image=True, write_video=False, video_name='LK_OF.avi'):
    """
    Runs Lucas-Kanade optical flow algorithm combined with an assumption of the car location rectangle to track the car
    location.

    :param imgs: string array containing image file locations
    :param detect_interval: interaval determining how often new features to track are calculated using Shi-Tomasi
            detector
    :param car_rect: initial car position defined by rectangle
    :param display_image: flag determining whether or not to display the image stream
    :param write_video: Flag determining whether or not to record image stream
    :param video_name: File name of image stream
    :return: array of average locations of the car rect
    """
    # Load images
    cap = cv2.VideoCapture(imgs[0])

    # If write_video flag, record a video
    if write_video:
        height, width, _ = cv2.imread(imgs[0]).shape
        writer = cv2.VideoWriter(video_name, -1, 20, (width, height))

    # Parameters for ShiTomasi corner detection:
    # maxCorners -
    # qualityLevel - characterizes the minimal accepted quality of the image corners. The parameter value is multiplied
    #                by the best corner quality measure, which is the minimal eigenvalue or the Harris function response
    # minDistance - distance between points
    # blockSize - Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
    feature_params = dict(maxCorners=20, qualityLevel=0.5, minDistance=7, blockSize=10)

    # Parameters for Lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Initialize the array that will contain feature points and the average position
    points = []
    ave_pos = []

    # Initialize the frame index
    frame_idx = 0

    # Loop through the image sequence, defining feature points every detection interval and
    while frame_idx < len(imgs):

        # Read the current frame
        ret, frame = cap.read()

        # Create grayscale image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create copy of image
        vis = frame.copy()

        # Empty the car points
        car_points = []

        # If the length of points to track is greater than 0, then the Lucas-Kanade optical flow algorithm is run on
        # those feature points and the previous feature points
        if len(points) > 0:

            # Define current image (img1), and previous image (img0)
            img0, img1 = prev_gray, frame_gray

            # Define previous points that were tracked
            p0 = np.float32([pt[-1] for pt in points]).reshape(-1, 1, 2)

            # Define new points to track using Lucas-Kanade and the current image and previous image
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

            # Define new points to track using Lucas-Kanade but with the images reversed
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

            # Find the difference between the 2 LK optical flow calculations for each tracking point
            difference = abs(p0-p0r).reshape(-1, 2).max(-1)

            # Define the good points to be those where the difference between the 2 LK optical flow calculations is
            #  less than 1. This will allow us to backward check the points to make sure that all points are good
            good = difference < 1
            
            # Define empty array to be new points to track
            new_points = []

            # Run through all points found in the LK optical flow calculation, determining which points will be tracked
            for pt, (x, y), good_flag in zip(points, p1.reshape(-1, 2), good):
                # If the difference between the 2 LK optical flow calcuations
                # (the one comparing the current image to the previous image, and the one that does the reverse), then
                # move on the to the next point
                if not good_flag:
                    continue
                # Add new points to the points to track
                pt.append((x, y))
                new_points.append(pt)

                # Draw circle around point
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                # If the point lies within the car rectangle, add it to the car points array
                if x > car_rect[0] and x < car_rect[2] and y > car_rect[1] and y < car_rect[3]:
                    car_points.append((x, y))

            # Update the car rectangle if it contains any points
            if len(car_points) > 0:
                car_rect = utilities.rect_resize(car_rect, car_points)

            # Append the center position of the car rectangle to the average position array
            ave_pos.append((np.mean([car_rect[0], car_rect[2]]), np.mean([car_rect[1], car_rect[3]])))

            # Define
            points = new_points

        # If on the detect_interval-th frame, then re-detect features to track using Shi-Tomasi
        if frame_idx % detect_interval == 0:

            # Define image mask
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            
            # Add points currently being tracked to the mask
            for x, y in [np.int32(pt[-1]) for pt in points]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            # Find new features to track using the mask defined
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

            # Add new points to the points array
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    points.append([(x, y)])

        # Set the previous grayscale image to the current grayscale image
        prev_gray = frame_gray


        if display_image:
            cv2.rectangle(vis, (car_rect[0], car_rect[1]), (car_rect[2], car_rect[3]), 255)
            cv2.imshow('lk_track', vis)
            frame_idx += 1
            ch = 0xFF & cv2.waitKey(30)
            if ch == 27:
                break

        # Write to video
        if write_video:
            writer.write(vis)

    # Close the image sequence viewer
    cv2.destroyAllWindows()
    if write_video:
        writer.release()
    return ave_pos

if __name__ == '__main__':
    # Load images
    imgs = utilities.get_jpeg('./resources/car/')

    # Run dense optical flow
    ave_pos = lk_optical_flow(imgs)

    # Plot the car position
    utilities.plot_pixel_position(ave_pos)