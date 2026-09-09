################################################################################    
################## SELF DRIVING CAR ENGINEER NANO DEGREE #######################
################################################################################

##### Project two: Computer vision. Advance lane detection
##### by Felipe Rojas
##### In this project, I will use advance techniques for camera calibration
##### image undistortion and correction, image transformation for a bird-view
##### of the road, lane detection and curve calculation

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

######################## First Step. Camera Calibration ########################

# I will use a set of chessboard images for the camera calibration
#The chessboard size is 9x6 for the project instead of 8x6 as in the lesson. 

def camera_cal(chess_size):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_size[0]*chess_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    # The function will return these arrays for the CV2 undistortion function
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # A list with all the chessboard images
    images = glob.glob('camera_cal/calibration*.jpg')
    
    # Cycle through the list
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chess_size[0],chess_size[1]), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (chess_size[0],chess_size[1]), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    return objpoints, imgpoints

####################### Second Step. Image Undistortion #######################

# With the Object points and the Image points obtained from the Camera Calibration
# function to undistort the Road image

def undistort(img, obj_p, img_p):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, img_size,None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Side by side images of the original image with the undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    return undist
    
##################### third Step. Gradient transformation ###################### 

#Here I will combine the different techniques for Color and Gradient Thresholds

# First, I will use the HLS representation of the image. Since the S channel is the
# cleaner for lane detection, I will use that one for the combined Threshold.
# Next, I will use the Absolute sobel Threshold for X axis
# Then, the magnitude threshold using both X and Y sobel
# Finally, the direction gradient
# Combining these thresholds, we have the output gradient Threshold

def hls_threshold(image, thresh_s=(160,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #H = hls[:,:,0]
    #L = hls[:,:,1]
    S = hls[:,:,2]

    s_binary = np.zeros_like(S)
    s_binary[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
    return s_binary
    
def abs_sobel(img, thresh=(0, 255), orient='x'):
    # 1) Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
         # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobelx)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    elif(orient == 'y'):
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
         # 3) Take the absolute value of the derivative or gradient
        abs_sobely = np.absolute(sobely)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
    else:
        print("not a valid orientation")
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_gradient = np.uint8(255*gradient_magnitude/np.max(gradient_magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_gradient)
    binary_output[(scaled_gradient >= mag_thresh[0]) & (scaled_gradient <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(gradient_direction)
    binary_output[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output    

def combined_threshold(img, thresh_s, abs_threshold, mag_threshold, dir_threshold, kernel_size = 3):
    # Apply each of the thresholding functions
    gradx = abs_sobel(img, abs_threshold, orient='x')
    #plt.imshow(gradx, cmap = 'gray')
    grady = abs_sobel(img, abs_threshold, orient='y')
    #plt.imshow(gradx, cmap = 'gray')
    mag_binary = mag_thresh(img, kernel_size, mag_threshold)
    #plt.imshow(mag_binary, cmap = 'gray')
    dir_binary = dir_thresh(img, kernel_size, dir_threshold)
    #plt.imshow(dir_binary, cmap = 'gray')
    s_binary = hls_threshold(img, thresh_s)
    combined = np.zeros_like(dir_binary)
    combined[ ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1) ] = 1
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return combined

################### Fourth Step. Perspective Transformation ###################

# After I got the Combined Threshold, I will perform a Perspective Transform
# The goal of this step is to obtain a Bird-view image

# First, I will define the area of interest. This is the road that will be transformed
# to the bird-view
def AreaofInterest(img, botleft,topleft, topright, botright):
    
    # Now I will draw a polygon to see which is the Area that I will transform
    vertices = np.array([[botleft,topleft, topright, botright]], 'int32')
    image_copy = np.copy(img)
    cv2.polylines(image_copy, vertices, True, color = (255,0,0), thickness = 2)
    return image_copy

# After I have the Area of the Road I want to Transform, I will use this function
# to get the Bird-View image of the Road
def persp_transfor(img, src_points, dst_points):
    imshape = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points) 
    warped = cv2.warpPerspective(img, M, imshape, flags=cv2.INTER_LINEAR)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Perspective Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return warped

########################## Fifth Step. Lane detection #########################

# Now that I have the Perspective Transformation with the Gradient Images I will use
# the Histogram of White points added to "see" where the lanes probably are
# For the First Frame of the Video, I will search the whole image for lane points to fit 
# in the polynomial function
# After that, the lane roads will remain almost the same, so I will pass the
# left and right fit points

def find_lane_pixels(image,  nwin=9, margin = 100, min_pix = 50):
    hist = np.sum(warped_gradient[warped_gradient.shape[0]//2:,:], axis=0)
    plt.plot(hist)
    # Create the Output image variable
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int(img.shape[0]//nwin)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwin):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                      (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                      (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_pix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img    

def fit_polynomial(image,  nwin=2, margin = 100, min_pix = 50):
    
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(image,  nwin, margin, min_pix)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    

    # Plots the left and right polynomials on the lane lines
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)

    return out_img, left_fit, right_fit

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fit, right_fit, ploty

def search_around_poly(binary_warped, left_fit, right_fit, margin = 100):
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
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

    # Fit new polynomials
    left_fit, right_fit, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result, left_fit, right_fit
      
def draw_lanes(og_image, grad_image, left_fit, right_fit, src_points, dst_points):
    ploty = np.linspace(0, og_image.shape[0]-1, num = og_image.shape[0] )
    mask = np.zeros_like(grad_image).astype(np.uint8)
    color_mask = np.dstack((mask, mask, mask))
    # For the function fillPolly, we need an array of the points
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_mask, np.int_([pts]), (0,255, 0))
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_mask, Minv, (og_image.shape[1], og_image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(og_image, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    return result, ploty

def curv_offset(og_image, left_fit, right_fit, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    ploty = np.linspace(0, og_image.shape[0]-1, num = og_image.shape[0] )
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    #### Implement the calculation of R_curve (radius of curvature) #####
    left_curv = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curv = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    average_rad = (left_curv + right_curv)/2
    average_rad = "Radius of Curvature: %.2f m" % average_rad
    cv2.putText(og_image,average_rad , (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,0), thickness=3)
    
    car_pos_y = og_image.shape[0]
    #I will look for X position when the Y in at the max value (the car position, or the bottom of the image)
    left_fit_bottom = left_fit[0]*car_pos_y**2 + left_fit[1]*car_pos_y + left_fit[2]
    right_fit_bottom = right_fit[0]*car_pos_y**2 + right_fit[1]*car_pos_y + right_fit[2]
    # Now I will get the average of both X values to get the Center of the lane.
    lane_center = (left_fit_bottom + right_fit_bottom)/2.
    # Assuming the center of the image is where the car is, I will compare it to the Center of the lane
    #If the result is positive, the Car offset is to the right, if the result is negative the car offset is to the left
    offset_pix = og_image.shape[1]/2 - lane_center 
    # Scale the result to meters
    offset_m = offset_pix*xm_per_pix
    offset_m = "Car Offset:%.2f m" %offset_m
    cv2.putText(og_image,offset_m , (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,0), thickness=3)
    plt.imshow(og_image)
    return og_image

# Finally, I will define a function that compiles all the functions

def advance_lane_detec(img):
    # 1) Image undistortion
    undist_image = undistort(img, obj_p, img_p)
    # 2) Combined Threshold
    gradient_binary = combined_threshold(img, thresh_s = (140,200), abs_threshold=(30, 200), mag_threshold=(70, 170), dir_threshold=(0.3, 1.1), kernel_size=3)
    # 3) Transform to a Bird-View Perspective 
    # 3.1) define the points for the Road polygon and draw it
    imshape = (img.shape[1], img.shape[0])
    bottom_left = [int(imshape[0]*0.195), int(imshape[1]*0.94)]
    bottom_right = [int(imshape[0]*0.85), int(imshape[1]*0.94)]
    top_left = [int(imshape[0]*0.465), int(imshape[1]*0.62)]
    top_right = [int(imshape[0]*0.535), int(imshape[1]*0.62)]
    src_points = np.float32([bottom_left, top_left, top_right, bottom_right])
    dst_points = np.float32([[bottom_left[0], 720], [bottom_left[0], 0], [bottom_right[0], 0], [bottom_right[0], 720]])
    # 3.2) Optinally, I could drat the Source points to see how they fit the Road image
    #aoi_image = AreaofInterest(undist_image, bottom_left,top_left, top_right, bottom_right)
    #plt.imshow(aoi_image2)
    # 3.3) Now, with the Source points and Destination Points, I will do the Perspective transformation
    warped_gradient = persp_transfor(gradient_binary, src_points, dst_points)
    # 3.4) Next step, is to find the lane pixels in the image
    ploty = np.linspace(0, gradient_binary.shape[0]-1, num = gradient_binary.shape[0] )
    out_image, left_fit, right_fit = fit_polynomial(img, nwin=9, margin = 80, min_pix = 50)
    
    # 3.5) now with the information from the lane detection I will draw the lanes in the road image
    lanes_image = draw_lanes(undist_image, gradient_binary, left_fit, right_fit, src_points, dst_points)
    final_image = curv_offset(undist_image, left_fit, right_fit, ploty)
    return final_image
    
################################################################################    
############# Testing the functions for Advance Lane Detection #################
################################################################################

# First, let's use the camera calibration function!


global frames,obj_p, img_p,leftx, lefty, rightx, righty
frames = 0
obj_p, img_p = camera_cal(chess_size = (9,6))   #The chess joints are 9 horizontal and 6 vertical


# The object points and image points will be used from now on
img = mpimg.imread('test_images/straight_lines1.jpg')
final_image1 = advance_lane_detec(image1)
mpimg.imsave("output_images/straight_lines1_output.png", final_image1)

image2 = mpimg.imread('test_images/straight_lines2.jpg')
final_image2 = advance_lane_detec(image2, obj_p, img_p)
mpimg.imsave("output_images/straight_lines2_output.png", final_image2)

image3 = mpimg.imread('test_images/test1.jpg')
final_image3 = advance_lane_detec(image3, obj_p, img_p)
mpimg.imsave("output_images/test1_output.png", final_image3)

image4 = mpimg.imread('test_images/test2.jpg')
final_image4 = advance_lane_detec(image4, obj_p, img_p)
mpimg.imsave("output_images/test2_output.png", final_image4)

image5 = mpimg.imread('test_images/test3.jpg')
final_image5 = advance_lane_detec(image5, obj_p, img_p)
mpimg.imsave("output_images/test3_output.png", final_image5)

image6 = mpimg.imread('test_images/test4.jpg')
final_image6 = advance_lane_detec(image6, obj_p, img_p)
mpimg.imsave("output_images/test4_output.png", final_image6)

image7 = mpimg.imread('test_images/test5.jpg')
final_image7 = advance_lane_detec(image7, obj_p, img_p)
mpimg.imsave("output_images/test5_output.png", final_image7)

image8 = mpimg.imread('test_images/test6.jpg')
final_image8 = advance_lane_detec(image8, obj_p, img_p)
mpimg.imsave("output_images/test6_output.png", final_image8)



from moviepy.editor import VideoFileClip

video_output = 'output_images/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(advance_lane_detec) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)