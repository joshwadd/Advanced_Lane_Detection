import numpy as np
import cv2

class CameraCalibration(object):

    def __init__(self, calibration_images, nx, ny):

        self.mtx, self.dist = self.camera_calibration(calibration_images, nx, ny)

    def camera_calibration(self, calibration_images, nx, ny):

        obj_points = [] #Object in 3D space
        img_points = [] #Points in 2D Image plane

        #Prepare the 3D object points belonging to chess
        #board corners eg. (0,0,0) , (1,0,0), (2,0,0)
        corner_points = np.zeros((nx*ny,3), np.float32)
        corner_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        for image in calibration_images:
            img = cv2.imread(image)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #Find the chess board corners in the calibration image
            ret, corners = cv2.findChessboardCorners(gray_img, (nx, ny), None)

            #If the corners are found add to lists
            if ret == True:
                obj_points.append(corner_points)
                img_points.append(corners)

        #Use the lists of image and object points to calibrate
        image_size = (gray_img.shape[1], gray_img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

        return mtx, dist

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)


