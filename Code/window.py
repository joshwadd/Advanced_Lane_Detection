import numpy as np

class Window(object):
    """
    Class to contain details related to each scanning window

    """

    def __init__(self, y_top, y_bottom, x_center, m=100, tol=50):

        self.y_top = y_top
        self.y_bottom = y_bottom
        self.x_center = x_center
        self.x_next_window = x_center
        self.m = m
        self.tol = tol

    def get_coordinate(self):

        return ((self.x_center - self.m, self.y_bottom ),(self.x_center + self.m, self.y_top))


    def find_pixels(self, non_zero, new_x = None):

        nonzeroy = np.array(non_zero[0])
        nonzerox = np.array(non_zero[1])

        if new_x is not None:
            self.x_center = new_x

        good_indices = ((nonzeroy >= self.y_top) & (nonzeroy < self.y_bottom) &
                        (nonzerox >= (self.x_center - self.m)) &  (nonzerox < (self.x_center + self.m))).nonzero()[0]


        if len(good_indices) > self.tol:
            self.x_next_window = np.int(np.mean(nonzerox[good_indices]))
        else:
            self.x_next_window = self.x_center

        return good_indices


