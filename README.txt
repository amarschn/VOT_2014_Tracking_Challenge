VOT 2014 Tracking Challenge

Python package containing images from the VOT 2014 Challenge and source code used in car tracking and visualization.
Much of the code is modified from code samples provided by openCV.

__init__.py :

Contains author name

cascade_tracker.py :

Contains the 'cascade' function which will run a Haar classifier
on an image stream and returns the average location of the classifier matches
at each frame in an array.

dense_optical_flow.py :

Runs dense optical flow algorithm on image sequence and displays output.

lk_tracker.py :

Runs Lucas-Kanade optical flow algorithm to update bounding rectangle of the car

orb_tracker.py :

Runs an ORB detector and car tracker based on an initial car rectangle position value.

utilities.py :

contains multiple utility functions:
 get_jpeg => will load jpeg images and return them as an array.
 plot_pixel_position => will plot pixel location given an array.
 rect_resize => will resize a rectangle given an array of points.