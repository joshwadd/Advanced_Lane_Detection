import numpy as np
import cv2


#Calculate the directional gradient
def abs_sobel_thresh(img_channel, orient='x', sobel_kernel=3 , thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    binary_output = np.uint8(255*abs_sobel/np.max(abs_sobel))
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask

#Calculate the gradient magitude
def mag_thresh(img_channel, sobel_kernel=3, thresh_min=0, thresh_max=255):

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Create a binary mask of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    return binary_output

def value_thresh(image_channel, thresh_min=0, thresh_max=255):
    binary_mask = np.zeros_like(image_channel)
    binary_mask[(image_channel > thresh_min) & (image_channel <= thresh_max)] = 1

    return binary_mask

def colour_thresh(image, thresh_min= np.array([0,0,0]), thresh_max = np.array([255,255,255])):
    colour_mask = cv2.inRange(image, thresh_min, thresh_max)/255
    return colour_mask.astype(np.uint8)


#Calculate the gradient direction
def dir_threshold(img_channel, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):

    # Take both Sobel x and y gradients
    sobel_x = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)

    # Calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    direction = np.absolute(direction)

    # Binary mask where the griadent directions are within the threshold values
    mask = np.zeros_like(direction)
    mask[(direction >= thresh_min) & (direction <= thresh_max)] = 1

    return mask

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #filling pixels inside the polygon defined by "vertices" with 1
    cv2.fillPoly(mask, vertices, 1)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_edges(image, debug_channels=False):
    """Function to output a binary threshold image for detecting lines"""

    # Get image dimensions.
    height = image.shape[0]
    width = image.shape[1]

    vertices = np.array([[(0,height-1),(width/2, int(0.5*height)),(width-1, height-1)]], dtype=np.int32)

    #############
    # Track 1, colour mask filtering
    ##############
    #Convert to HSL colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #Set parametrs for yellow filtering in HSV
    hsv_yellow_low = np.array([0, 100, 100])
    hsv_yellow_high = np.array([50, 255, 255])

    #Set parameters for white filtering in HSV
    hsv_white_low = np.array([18, 0, 180])
    hsv_white_high = np.array([255, 80, 255])

    yellow_mask = colour_thresh(hsv, hsv_yellow_low, hsv_yellow_high)
    white_mask = colour_thresh(hsv, hsv_white_low, hsv_white_high)

    #Combine the white and yellow masks
    colour_mask = np.zeros_like(yellow_mask)
    colour_mask[(yellow_mask ==1) | (white_mask ==1)] = 1

    colour_mask = region_of_interest(colour_mask, vertices)

    ########################
    #Track2: gradient filtering in HLS space
    ##########################
    img_l = hls[:,:,1]
    img_l_grad_x = abs_sobel_thresh(img_l, 'x', 5, 50, 255)
    img_l_grad_y = abs_sobel_thresh(img_l, 'y', 5, 50, 255)

    grad_l_mask = np.zeros_like(img_l)
    grad_l_mask[(img_l_grad_x ==1) | (img_l_grad_y==1)] =1


    img_s = hls[:,:,2]
    img_s_grad_x = abs_sobel_thresh(img_s, 'x', 5, 50, 255)
    img_s_grad_y = abs_sobel_thresh(img_s, 'y', 5, 50, 255)

    grad_s_mask = np.zeros_like(img_s)
    grad_s_mask[(img_s_grad_x ==1) | (img_s_grad_y==1)] = 1

    #Combine the gradient filters for the two channels
    gradient_mask = np.zeros_like(grad_s_mask)
    gradient_mask[(grad_s_mask == 1) | (grad_l_mask == 1)] = 1

    gradient_mask = region_of_interest(gradient_mask, vertices)



    if debug_channels:
        return np.dstack((np.zeros_like(colour_mask), colour_mask, gradient_mask))*255
    else:
        binary_mask = np.zeros_like(colour_mask)
        binary_mask[(colour_mask ==1) | (gradient_mask == 1)] = 1
        return binary_mask



