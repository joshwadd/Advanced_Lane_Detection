from calibration import CameraCalibration
from perspective import PerspectiveTransform
from line import Line
from window import Window
import numpy as np
import cv2

from edge_detection import detect_edges



class LaneDetection(object):
    """Class to dectect and draw the lanes on a serise of images/frames"""
    def __init__(self, h, w, CALIBRATION_IMAGES, NX, NY, nwindows = 9, margin =100, tol=50):

        #Variables for the image size
        self.h = h
        self.w = w

        #Initlaise objects to perform the calibration and perspective transforms
        self.calibrator = CameraCalibration(CALIBRATION_IMAGES, NX, NY)
        self.perspective_transform = PerspectiveTransform()

        #Initalise line objects to keep track of the properties of the fitted lines
        #overtime
        self.left_line = Line(self.h, self.w)
        self.right_line = Line(self.h, self.w)

        #Initalise window objects to keep track of the
        self.nwindows = nwindows
        self.window_height = np.int(self.h/nwindows)
        self.left_window = []
        self.right_window = []
        self.margin = margin
        self.tol = tol
        self.initalise_windows()



    def process_frame(self, image, debug=True):
        # Step 1: Correct for the camera image distortion
        corrected_image = self.calibrator.undistort(image)
        # Step 2: Binary threshold image to detect the lane lines
        threshold_image = detect_edges(corrected_image)
        # Step 3: Apply a perspective transform to the image for birds eye view
        perspective_image = self.perspective_transform.warp(threshold_image)

        # Step 4: Finding and fitting a polynomial line to the lane lines


        if not(self.left_line.detected) or not(self.right_line.detected):
            self.fit_windows(perspective_image)
            if debug:
                edges_debug = detect_edges(corrected_image, True)
                perspective_debug = self.perspective_transform.warp(edges_debug)
                debug_image = self.debug_visualisation_windows(perspective_debug)
        else:
            self.fit_region(perspective_image)
            if debug:
                edges_debug = detect_edges(corrected_image, True)
                perspective_debug = self.perspective_transform.warp(edges_debug)
                debug_image = self.debug_visualisation_region(perspective_debug)



        if debug:
            overhead_lane = self.draw_lanes(self.perspective_transform.warp(corrected_image),True)
            debug_viewer = cv2.resize(debug_image, (0,0), fx=0.3, fy=0.3)
            overhead_viewer = cv2.resize(overhead_lane, (0,0), fx=0.3, fy=0.3)

            #Make top region of image darker
            corrected_image[:250, :, :] = corrected_image[:250, :, :]*0.4
            debug_h = debug_viewer.shape[0]
            debug_w = debug_viewer.shape[1]
            #Overlay the debug view to the top of the image
            corrected_image[20:20+debug_h, 20:20+debug_w, :] = debug_viewer
            #Overlay the overhead view to the top of the image
            corrected_image[20:20+debug_h, 40+3*20+ 2*debug_w:3*20+3*debug_w+40, :] = overhead_viewer


            #Print the lane distance and radius of curv to the screen
            x_location = 3*20 + debug_w
            self.text_to_image(corrected_image, 'Radius of curvature: {:.1f} m'.format(self.avg_radius_of_curvature()),x_location, 40)
            self.text_to_image(corrected_image, 'Left distance:  {:.3f} m'.format(self.left_line.camera_distance()),x_location, 100)
            self.text_to_image(corrected_image, 'Right distance:  {:.3f} m'.format(self.right_line.camera_distance()),x_location, 160)
            self.text_to_image(corrected_image, 'Center distance:  {:.3f} m'.format(self.left_line.camera_distance() - self.right_line.camera_distance()),x_location, 220)


        identified_lane = self.draw_lanes(corrected_image)

        return identified_lane


    def text_to_image(self, image, text, coordinate_x, coordinate_y):
        cv2.putText(image, text, (coordinate_x, coordinate_y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255))

    def fit_region(self, warped_image ):

        left_fit = self.left_line.average_fits()
        right_fit = self.right_line.average_fits()

        if not(self.left_line.detected) or not(self.right_line.detected):
            print("No current coefficents")
            return

        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]


        self.left_line.compute_line(lefty, leftx)
        self.right_line.compute_line(righty, rightx)

    def initalise_windows(self):

        leftx = np.int(self.w/4)
        rightx = np.int(3*(self.w/4))

        for i in range(self.nwindows):

            left_window = Window(self.h - (i+1)*self.window_height,
                                 self.h - i*self.window_height,
                                 leftx, self.margin, self.tol)
            right_window = Window(self.h - (i+1)*self.window_height,
                                 self.h - i*self.window_height,
                                 rightx, self.margin, self.tol)

            self.left_window.append(left_window)
            self.right_window.append(right_window)


    def fit_windows(self, image_edges):

        #Create a histrogram of the bottom half of the image and find two peaks
        histogram = np.sum(image_edges[int(image_edges.shape[0]/2):,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        # Find the locations of the non-zero pixels
        nonzero = image_edges.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists for the left and right lane pixel indices (might not be required)
        left_inds = []
        right_inds = []

        for i in range(self.nwindows):
            if i == 0:
                self.left_window[i].x_center = leftx_current
                self.right_window[i].x_center = rightx_current
            else:
                self.left_window[i].x_center = self.left_window[i-1].x_next_window
                self.right_window[i].x_center = self.right_window[i-1].x_next_window

            left_inds.append(self.left_window[i].find_pixels(nonzero))
            right_inds.append(self.right_window[i].find_pixels(nonzero))

        # Concatenate the arrays of indices
        left_inds = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_inds]
        lefty = nonzeroy[left_inds]
        rightx = nonzerox[right_inds]
        righty = nonzeroy[right_inds]

        # Find lines to the extracted pixels
        self.left_line.compute_line(lefty, leftx)
        self.right_line.compute_line(righty, rightx)


    def debug_visualisation_windows(self, image, lines=True, windows=True):

        out_img = image


        if windows:
            for window in self.left_window:
                coordinate = window.get_coordinate()
                cv2.rectangle(out_img, coordinate[0], coordinate[1], (0,255,0), 2)
            for window in self.right_window:
                coordinate = window.get_coordinate()
                cv2.rectangle(out_img, coordinate[0], coordinate[1], (0,255,0), 2)
        if lines:

            #Print the average fitted lines
            y, left_fitx = self.left_line.generate_line()
            if left_fitx is not None:
                cv2.polylines(out_img, [np.stack((left_fitx, y)).astype(np.int).T], False, (255,0,255),2)

            y, right_fitx = self.right_line.generate_line()
            if right_fitx is not None:
                cv2.polylines(out_img, [np.stack((right_fitx, y)).astype(np.int).T], False, (255,0,255),2)

             #Print the current fitted lines
            y, left_fit_currentx = self.left_line.generate_line_current()
            if left_fit_currentx is not None:
                cv2.polylines(out_img, [np.stack((left_fit_currentx, y)).astype(np.int).T], False, (255,128,0),2)

            y, right_fit_currentx = self.right_line.generate_line_current()
            if right_fit_currentx is not None:
                cv2.polylines(out_img, [np.stack((right_fit_currentx, y)).astype(np.int).T], False, (255,128,0),2)

        return out_img

    def debug_visualisation_region(self, image, lines=True, region=True):

        out_img = image#*255
        window_img = np.zeros_like(out_img)

        if region:
            y , left_fitx = self.left_line.generate_line_current()
            y , right_fitx = self.right_line.generate_line_current()
            margin = self.margin
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, y]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, y]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            out_img = cv2.addWeighted(out_img, 1.0, window_img, 0.3, 0)

        if lines:
            #Print the average fitted lines
            y, left_fitx = self.left_line.generate_line()
            if left_fitx is not None:
                cv2.polylines(out_img, [np.stack((left_fitx, y)).astype(np.int).T], False, (255,0,255),2)

            y, right_fitx = self.right_line.generate_line()
            if right_fitx is not None:
                cv2.polylines(out_img, [np.stack((right_fitx, y)).astype(np.int).T], False, (255,0,255),2)

             #Print the current fitted lines
            y, left_fit_currentx = self.left_line.generate_line_current()
            if left_fit_currentx is not None:
                cv2.polylines(out_img, [np.stack((left_fit_currentx, y)).astype(np.int).T], False, (255,128,0),2)

            y, right_fit_currentx = self.right_line.generate_line_current()
            if right_fit_currentx is not None:
                cv2.polylines(out_img, [np.stack((right_fit_currentx, y)).astype(np.int).T], False, (255,128,0),2)


        return out_img



    def draw_lanes(self, undist, birdseye=False):


        y, left_fitx = self.left_line.generate_line()
        y, right_fitx = self.right_line.generate_line()

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(undist[:,:,0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        if left_fitx is None or right_fitx is None:
            return undist

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        if not(birdseye):
            lane = self.perspective_transform.inverse_warp(color_warp)
        else:
            lane = color_warp

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        #newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, lane, 0.3, 0)

        return result

    def avg_radius_of_curvature(self):

        return np.average([self.left_line.radius_of_curvature(), self.right_line.radius_of_curvature()])

