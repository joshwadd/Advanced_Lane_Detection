import numpy as np
import cv2

LOOK_BACK = 5

class Line(object):
    def __init__(self, h, w):
        #Was the line detected in the last iteration
        self.detected = False

        #height of image
        self.h = h
        #width of image
        self.w = w

        # cofficents of the last n fits of the line
        self.recent_coefficents = []
        #polynomial coefficents for the most recent fit
        self.current_coefficents = None
        #Radius of curvature of the line in some units

        #counter for the number of lines added
        self.n = 0

        #parameters for converting from pixel to real space
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def compute_line(self, y=None, x=None):

        sufficent_pnts = len(y) > 0 #and np.max(y) - np.min(y) > self.h * .625
        self.detected = sufficent_pnts

        if not(self.detected):
            #print("Line not detected")
            self.current_coefficents = None
            return
        self.current_coefficents = self.fit_points(y,x)



        self.update_history()

    def fit_points(self, y, x):
        return np.polyfit(y, x, 2)

    def update_history(self):


        #Add the current coefficents to the back of the list
        # and pop the first
        if self.n  < LOOK_BACK:
            self.recent_coefficents.append(self.current_coefficents)
        else:
            self.recent_coefficents.pop(0)
            self.recent_coefficents.append(self.current_coefficents)

        self.n += 1

    def average_fits(self):

        if not self.recent_coefficents:
            return None

        recent_coefficents = np.asarray(self.recent_coefficents)
        average_cofficents = np.mean(recent_coefficents, axis=0)
        return average_cofficents

    def generate_line(self):

        y = np.linspace(0, self.h -1, self.h)
        current_coeff = self.average_fits()

        if current_coeff is None:
            return None, None


        line_x = current_coeff[0]*y**2 + current_coeff[1]*y + current_coeff[2]

        return y, line_x

    def generate_line_current(self):


        y = np.linspace(0, self.h -1, self.h)
        current_coeff = self.current_coefficents

        if current_coeff is None:
            return None, None


        line_x = current_coeff[0]*y**2 + current_coeff[1]*y + current_coeff[2]

        return y, line_x



    def camera_distance(self):

        #distance in meters of vechicle center from the line

        y, x = self.generate_line()

        x_bottom = x[self.h-1]

        return np.absolute((self.w //2 -  x_bottom) * self.xm_per_pix)


    def radius_of_curvature(self):


        y, fitx = self.generate_line()

        fit_cr = np.polyfit(y*self.ym_per_pix, fitx*self.xm_per_pix,2)

        return ((1 + (2*fit_cr[0]*720*self.ym_per_pix + fit_cr[1])**2)**1.5)/ np.absolute(2*fit_cr[0])










