#!/usr/bin/env python
"""
Dense Optical Flow
==================
Runs dense optical flow algorithm on image sequence and displays output
"""


import numpy as np
import cv2

def dense_optical_flow(imgs, display_image=True, display_type='hsv', write_video=False, video_name='D_OF.avi'):
    """
    Runs and displays dense optical flow visualization of image sequence

    :param imgs: string array containing image file locations
    :param display_image: flag determining whether or not to display the image stream
    :param display_type: Type of display for image sequence, hsv or vec (hsv is color-based and vec shows a grid of
            vectors)
    :param write_video: Flag determining whether or not to record image stream
    :param video_name: File name of image stream
    :return: None
    """

    # Load images
    cap = cv2.VideoCapture(imgs[0])

    # If write_video flag, record a video
    if write_video:
        height, width, _ = cv2.imread(imgs[0]).shape
        writer = cv2.VideoWriter(video_name, -1, 20, (width, height))

    # Read the first frame and initialize hsv image
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Set initial hsv image to all zeros
    hsv = np.zeros_like(frame1)

    # Set to color
    hsv[..., 1] = 255

    # initialize frame index
    frame_idx = 1

    while frame_idx < len(imgs):

        # Load next frame and grayscale frame
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


        # Arguments:
        # prev - first image
        # next - second image
        # pyr_scale - image scale for pyramid, 0.5 means each lay is half the previous
        # levels - pyramid number of levels
        # winsize - averaging size -> bigger = blurrier
        # iterations - iterations at each pyramid level -> makes for better results but significantly slows speed
        # poly_n - pixel neighborhood size -> bigger = blurrier
        # poly_sigma - gaussian std used for smoothing
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 1, 50, 2, 7, 1.5, 1)

        # Display type
        if display_image:
            if display_type == 'hsv':
                frame = draw_hsv(hsv, flow)
            elif display_type == 'vec':
                frame = draw_vec(next, flow)
            cv2.imshow('Frame', frame)

            # Write to video
            if write_video:
                writer.write(frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Set previous to next and increment index
        prvs = next
        frame_idx += 1

    # Destroy image viewer
    cap.release()
    cv2.destroyAllWindows()
    if write_video:
        writer.release()


def draw_vec(img, flow, step=10):
    """
    Displays vector field on image. This code was taken directly from the "samples" portion of opencv

    :param img: image to display
    :param flow: flow field given by Farneback optical flow algorithm
    :param step: grid spacing
    :return: vis image
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(hsv, flow):
    """
    Displays hsv color field of polar coordinates on image. This code was taken directly from the "samples" portion of opencv
    :param hsv: hsv color image
    :param flow: flow field given by Farneback optical flow algorithm
    :return:
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


if __name__ == '__main__':
    import utilities

    # Load images
    imgs = utilities.get_jpeg('./resources/car/')

    # Run dense optical flow
    dense_optical_flow(imgs, display_type='vec')
