import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class FindLanes:
    def __init__(self):
        self.objpoints, self.imgpoints = self.calibrate()
        self.lr_fit = None
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None
        self.left_curverad = None
        self.right_curverad = None
        self.curvature_difference = None
        self.center = None
        self.leftx_base = None
        self.rightx_base = None
        self.base_difference = 0 

        self.src0 = np.float32(
            [
                [800,500], # top r
                [1150,720], # bot r
                [200,720], # bot l
                [525,500] # top l
            ]
        )

        self.dst0 = np.float32(
            [
                [1150, 300], # top r
                [1150,720], # bot r
                [175,720], # bot l
                [175,300] # top l
            ]
        )

        self.src = np.float32(
                    [[280,  700],  # Bottom left
                     [595,  460],  # Top left
                     [725,  460],  # Top right
                     [1125, 700]]) # Bottom right

        self.dst = np.float32(
            [[250,  720],  # Bottom left
             [250,    0],  # Top left
             [1065,   0],  # Top right
             [1065, 720]]) # Bottom right 

        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def calibrate(self):
        calibration_images = glob.glob('camera_cal/calibration*.jpg')

        imgpoints = []
        objpoints = []
        
        nx,ny = (9,6)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        for fname in calibration_images:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            ret, corners = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (nx, ny), None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        return objpoints, imgpoints

    def cal_undistort(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        
        return undist

    def saturation_threshold(self, img, thresh=(170, 255)):
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # Apply a threshold to the S channel
        S = hls[:,:,2]
        # Return a binary image of threshold result
        binary_output = np.zeros_like(S) # placeholder line
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1

        return binary_output

    def abs_sobel_threshold(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        o = {"x" : 0, "y" : 0}
        o[orient] += 1
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, o['x'], o['y'], ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def mag_threshold(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = {"x" : cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel), "y" : cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) }
        mag = np.sqrt(np.power(sobel["x"],2) + np.power(sobel["y"],2))
        scaled_sobel = np.uint8(255*mag/np.max(mag))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = {"x" : cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel), "y" : cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) }
        direction = np.arctan2(np.absolute(sobel['y']), np.absolute(sobel['x']))

        binary_output = np.zeros_like(direction)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

        return binary_output

    def color_gradient(self, img):
        saturation = self.saturation_threshold(img, thresh=(170, 255))
        x_gradient = self.abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(50, 200))
        y_gradient = self.abs_sobel_threshold(img, orient='y', sobel_kernel=3, thresh=(50, 200))
        magnitude = self.mag_threshold(img, sobel_kernel=9, mag_thresh=(30,100))
        direction = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

        combined = np.zeros_like(direction)
        combined[((x_gradient == 1) & (y_gradient == 1)) | ((magnitude == 1) & (direction == 1)) | (saturation == 1)] = 1
        
        return combined

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])

        self.mtx = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, self.mtx, img_size, flags=cv2.INTER_LINEAR)

        return warped, self.mtx

    def lower_histogram(self, img):
        # Sum across the bottom half image pixels vertically, (axis=0). 
        return np.sum(img[img.shape[0]//2:,:], axis=0)

    def find_lane_pixels(self, img):
        # Take a histogram of the bottom half of the image
        histogram = self.lower_histogram(img)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        current_base_difference = abs(rightx_base - leftx_base)

        if not current_base_difference < self.base_difference - 100:
            self.leftx_base = leftx_base
            self.rightx_base = rightx_base
            self.base_difference = current_base_difference


        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of the windows - based on the number of windows and the image height
        window_height = np.int(img.shape[0]//nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated later for each window in nwindows
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
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

    def fit_polynomial(self, img):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        # Fit a second order polynomial to each using `np.polyfit`
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        try:
            self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx = 1*self.ploty**2 + 1*self.ploty
            self.right_fitx = 1*self.ploty**2 + 1*self.ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        #return (left_fit, right_fit), (left_fitx, ploty), (right_fitx, ploty), out_img
        return out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
         ### Fit a second order polynomial to each with np.polyfit() ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### TO-DO: Calc both polynomials using self.ploty, left_fit and right_fit ###
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        
        return self.left_fitx, self.right_fitx, self.ploty

    def search_around_poly(self, img):
        if self.left_fit is None or self.right_fit is None:
            return self.fit_polynomial(img)
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        margin = 150

        # Grab activated pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if (leftx.size == 0 or rightx.size == 0):
            #print("sizes are 0")
            return img #self.fit_polynomial(img)

        # Fit new polynomials
        self.fit_poly(img.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, 
                                  self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, 
                                  self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        return result

    def curvature(self, img):        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(self.ploty)
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.right_fitx*xm_per_pix, 2)
        
        # Calculation of R_curve (radius of curvature)
        self.left_curvature =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self.curvature_difference = self.right_curverad - self.left_curvature
        
        # Calculate vehicle center
        """You can assume the camera is mounted at the center of the car, 
        such that the lane center is the midpoint at the bottom of the image 
        between the two lines you've detected. The offset of the lane center 
        from the center of the image (converted from pixels to meters) 
        is your distance from the center of the lane."""
        left_lane_bottom = (self.left_fit[0]*y_eval)**2 + self.left_fit[0]*y_eval + self.left_fit[2]
        right_lane_bottom = (self.right_fit[0]*y_eval)**2 + self.right_fit[0]*y_eval + self.right_fit[2]                       
        lane_center = (left_lane_bottom + right_lane_bottom)/2.
        self.center = (lane_center - img.shape[1]//2) * xm_per_pix 

        # Checking that they have similar curvature
        # Checking that they are separated by approximately the right distance horizontally
        # Checking that they are roughly parallel
        
        return self.left_curvature, self.right_curverad, self.center

    def measure_curvature_real(self, img):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.mean(self.ploty)
        
        # Calculation of R_curve (radius of curvature)
        self.left_curverad = ((1 + (2*self.left_fit[0]*y_eval*ym_per_pix + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        self.right_curverad = ((1 + (2*self.right_fit[0]*y_eval*ym_per_pix + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])

        left_lane_bottom = (self.left_fit[0]*y_eval)**2 + self.left_fit[0]*y_eval + self.left_fit[2]
        right_lane_bottom = (self.right_fit[0]*y_eval)**2 + self.right_fit[0]*y_eval + self.right_fit[2]                       
        lane_center = (left_lane_bottom + right_lane_bottom)/2.
        self.center = (lane_center - img.shape[1]//2) * xm_per_pix 

        self.curvature_difference = self.right_curverad - self.left_curverad
        
        return self.left_curverad, self.right_curverad, self.center

    def lane_display(self, frame, warped):

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (frame.shape[1], warped.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
        
        font_family = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        
        cv2.putText(result, 'Left: {:.0f} m'.format(self.left_curverad), (900, 50), font_family, font_size, (255, 255, 255), 2)
        cv2.putText(result, 'Right: {:.0f} m'.format(self.right_curverad), (900, 100), font_family, font_size, (255, 255, 255), 2)
        cv2.putText(result, 'Center: {:.0f} m'.format(self.center), (900, 150), font_family, font_size, (255, 255, 255), 2)
            
        return result

    def __call__(self, frame):

        # Use the calibration parameters to remove distortion
        undist = self.cal_undistort(frame)

        # Combine color and gradient layers to highlight lane lines
        combined = self.color_gradient(undist)

        # Top-down perspective transform
        warped, self.mtx = self.warp(combined)
                
        out_img = self.search_around_poly(warped)

        self.measure_curvature_real(out_img)

        if self.curvature_difference > abs(100):
            out_img = self.fit_polynomial(warped)


        result = self.lane_display(frame, warped)

        # Checking that they have similar curvature
        # Checking that they are separated by approximately the right distance horizontally
        # Checking that they are roughly parallel
            
        return result

