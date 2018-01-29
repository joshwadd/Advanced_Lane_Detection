import numpy as np
import cv2

class PerspectiveTransform(object):
    '''Class to compute birds eye view transformation

    Define a set of source and destination points on straight lane lines.
    This transformation can then be used for all images taken with the camera
    in this position, under the assumption the road is a flat plane
    '''
    def __init__(self):

        #Source points
        bottom_left = [220, 720]
        bottom_right = [1110, 720]
        top_left = [570, 470]
        top_right = [722, 470]



        # Define the source points
        source = np.float32([bottom_left,
                          bottom_right,
                          top_right,
                          top_left])

        #pts = np.array([bottom_left, bottom_right, top_right, top_left], np.int32)
        #pts = pts.reshape((-1,1,2))



        #destination points
        bottom_left = [320, 720]
        bottom_right = [920, 720]
        top_left = [320, 1]
        top_right = [920, 1]

        # Define the destination points
        dst = np.float32([bottom_left,
                         bottom_right,
                         top_right,
                         top_left])

        # Compute the perspective transform
        self.mat = cv2.getPerspectiveTransform(source, dst)
        self.mat_inv = cv2.getPerspectiveTransform(dst, source)

    def warp(self, image):
        return cv2.warpPerspective(image, self.mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    def inverse_warp(self, image):
        return cv2.warpPerspective(image, self.mat_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


