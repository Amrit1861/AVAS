import cv2
import numpy as np
import edge_detection as edge
import matplotlib.pyplot as plt

filename=r"C:/folder9.6/college/IVth year/major-minor project/custom dataset/train/JPEGImages/10 (17).jpg"

def binary_array(array, thresh, value=0):
    lower_bound, upper_bound = thresh
    if value == 0:

        binary = np.ones_like(array,dtype=np.uint8)
        binary[(array < lower_bound) | (array > upper_bound)] = 0
    else:
        binary = np.zeros_like(array,dtype=np.uint8)
        binary[(array >= lower_bound) & (array <= upper_bound)] = 1
    return binary

def blur_gaussian(channel, ksize=3):
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag = np.uint8(255 * mag / np.max(mag))
    lower_bound, upper_bound = thresh
    return binary_array(mag, (lower_bound, upper_bound))

def sobel(img_channel, orient='x', sobel_kernel=3):
    if orient == 'x':
        sobel_output = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel_output = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return np.abs(sobel_output)

def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)





class Lane:
    def __init__(self,orig_frame):
        if orig_frame is None:
            raise ValueError("Error: Original frame is None. Check if the image path is correct.")
        self.orig_frame=orig_frame
        self.lane_line_markings=None
        self.warped_frame=None
        self.transformation_matrix=None
        self.inv_transformation_matrix=None

        self.orig_image_size=self.orig_frame.shape[::-1][1:]
        width=self.orig_image_size[0]
        height=self.orig_image_size[1]
        self.width=width
        self.height=height


        self.roi_points=np.float32([
            (2000,4000),
            (1000,2000),
            (7000,2000),
            (6000,4000)
        ])

        self.padding=int(0.25*width)
        self.desired_roi_points=np.float32([
            [self.padding,0],
            [self.padding,self.orig_image_size[1]],
            [self.orig_image_size[0]-self.padding, self.orig_image_size[1]],
            [self.orig_image_size[0]-self.padding,0]
        ])

        self.histogram=None
        self.no_of_windows=10
        self.margin=int((1/12)*width)
        self.minpix=int((1/24)*width)

        self.left_fit=None
        self.right_fit=None
        self.left_lane_inds=None
        self.right_lane_inds=None
        self.ploty=None
        self.left_fitx=None
        self.right_fitx=None
        self.leftx=None
        self.rightx=None
        self.lefty=None
        self.righty=None

        self.YM_PER_PIX=10.0/1000
        self.XM_PER_PIX=3.7/781

        self.left_curvem=None
        self.right_curvem=None
        self.center_offset=None
    def calculate_car_position(self,print_to_terminal=False):
        car_location=self.orig_frame.shape[1]/2
        height=self.orig_frame.shape[0]
        if self.left_fit is None or self.right_fit is None:
            if print_to_terminal:
                print("Lane lines not detected. Unable to calculate car position.")
            self.center_offset = None  # Assign a None or default value to indicate failure
            return None
        bottom_left=self.left_fit[0]*height**2+self.left_fit[1]*height+self.left_fit[2]
        bottom_right=self.right_fit[0]*height**2+self.right_fit[1]*height+self.right_fit[2]
        center_lane=(bottom_right-bottom_left)/2+bottom_left
        center_offset=(np.abs(car_location)-np.abs(center_lane))*self.XM_PER_PIX*100
        if print_to_terminal==True:
            print(str(center_offset)+'cm')
        self.center_offset=center_offset
        return center_offset

    def calculate_curvature(self,print_to_terminal=False):
        if self.lefty is None or self.leftx is None or self.righty is None or self.rightx is None or self.ploty is None:
            print(
                "Error: Missing lane line data (left_fitx, right_fitx, or ploty is None). Ensure the sliding window method is run.")
            return None, None
        y_eval=np.max(self.ploty)
        try:
            left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * self.XM_PER_PIX, 2)
            right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * self.XM_PER_PIX, 2)
        except ValueError as e:
            print(f"Polynomial fit error: {e}")
            return None, None
        left_fit_cr=np.polyfit(self.lefty*self.YM_PER_PIX,self.leftx*(self.XM_PER_PIX),2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (self.XM_PER_PIX), 2)
        left_curvem=((1+(2*left_fit_cr[0]*y_eval*self.YM_PER_PIX+left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5)/np.absolute(2*right_fit_cr[0])

        if print_to_terminal:
            print(f"Left Curvature: {left_curvem:.2f} m, Right Curvature: {right_curvem:.2f} m")
        self.left_curvem=left_curvem
        self.right_curvem=right_curvem
        return left_curvem,right_curvem

    def warp_perspective(self):
        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)

        # Warp the original frame to a top-down view
        self.warped_frame = cv2.warpPerspective(self.orig_frame, self.transformation_matrix, (self.width, self.height))

    def calculate_histogram(self,frame=None,plot=True):
        if frame is None:
            frame=self.warped_frame
        if frame is None:
            print("Error: Frame is None. Cannot calculate histogram.")
            return None

        self.histogram=np.sum(frame[int(frame.shape[0]/2):,:],axis=0)

        if plot:
            figure,(ax1,ax2)=plt.subplots(2,1)
            figure.set_size_inches(10,5)
            ax1.imshow(frame,cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def display_curvature_offset(self,frame=None,plot=False):
        image_copy=self.orig_frame.copy()
        if frame is None:
            image_copy=self.orig_frame.copy()
        else:
            image_copy=frame

        left_curvem = self.left_curvem if self.left_curvem is not None else 0.0
        right_curvem = self.right_curvem if self.right_curvem is not None else 0.0
        avg_curvem = (left_curvem + right_curvem) / 2

        cv2.putText(image_copy,
                    'Curve Radius: {:.2f}m'.format(avg_curvem),
                    (int((5 / 600) * self.width), int((20 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (float((0.5 / 600) * self.width)),
                    (255, 255, 255), 2, cv2.LINE_AA)

        offset_text = 'N/A' if self.center_offset is None else str(self.center_offset)[:7] + 'cm'
        cv2.putText(image_copy,
                    'Center Offset: ' + offset_text,
                    (int((5 / 600) * self.width), int((40 / 338) * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (float((0.5 / 600) * self.width)),
                    (255, 255, 255), 2, cv2.LINE_AA)


        if plot==True:
            cv2.imshow("Image with Curvature abd Offset",image_copy)
        return image_copy

    def get_lane_line_previous_window(self,left_fit,right_fit,plot=False):
        margin=self.margin

        nonzero=self.warped_frame.nonzero()
        nonzeroy=np.array(nonzero[0])
        nonzerox=np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0] * (
                nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                                  nonzerox < (left_fit[0] * (
                                  nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (
                nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                                   nonzerox < (right_fit[0] * (
                                   nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        leftx=nonzerox[left_lane_inds]
        lefty=nonzeroy[left_lane_inds]
        rightx=nonzerox[right_lane_inds]
        righty=nonzeroy[right_lane_inds]

        self.leftx=leftx
        self.rightx=rightx
        self.lefty=lefty
        self.righty=righty

        left_fit=np.polyfit(lefty,leftx,2)
        right_fit=np.polyfit(righty,rightx,2)
        self.left_fit=left_fit
        self.right_fit=right_fit

        ploty=np.linspace(
            0,self.warped_frame.shape[0]-1,
            self.warped_frame.shape[0])
        left_fitx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.ploty=ploty
        self.left_fitx=left_fitx
        self.right_fitx=right_fitx

        if plot==True:
            out_img=np.dstack((self.warped_frame, self.warped_frame,
                               (self.warped_frame)))*255
            window_img=np.zeros_like(out_img)

            out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]]=[255,0,0]
            out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]]=[0,0,255]

            margin=self.margin
            left_line_window1=np.array([np.transpose(np.vstack([
                left_fitx-margin,ploty
            ]))])
            left_line_window2 = np.array([np.flipud(np.vstack([
                left_fitx + margin, ploty
            ]))])
            left_line_pts=np.hstack((left_line_window1,left_line_window2))
            right_line_window1=np.array([np.transpose(np.vstack([right_fitx-margin,ploty]))])
            right_line_window2 = np.array([np.flipud(np.vstack([right_fitx + margin, ploty]))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))



            cv2.fillPoly(window_img,np.int_([left_line_pts]),(0,255,0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result=cv2.addWeighted(out_img,1,window_img,0.3,0)


            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def get_lane_line_indices_sliding_windows(self,plot=False):
        margin=self.margin
        frame_sliding_window=self.warped_frame.copy()
        window_height=np.int(self.warped_frame.shape[0]/self.no_of_windows)
        nonzero=self.warped_frame.nonzero()
        nonzeroy=np.array(nonzero[0])
        nonzerox=np.array(nonzero[1])
        left_lane_inds=[]
        right_lane_inds=[]

        leftx_base,rightx_base=self.histogram_peak()
        leftx_current=leftx_base
        rightx_current=rightx_base

        no_of_windows=self.no_of_windows

        for window in range(no_of_windows):
            win_y_low=self.warped_frame.shape[0]-(window+1)*window_height
            win_y_high=self.warped_frame.shape[0]-window*window_height
            win_xleft_low=leftx_current-margin
            win_xleft_high=leftx_current+margin
            win_xright_low=rightx_current-margin
            win_xright_high=rightx_current+margin
            cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),
                          (win_xleft_high,win_y_high),(255,255,255),2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (255, 255, 255), 2)

            good_left_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&
                            (nonzerox>=win_xleft_low)&
                            (nonzerox<win_xleft_high)).nonzero()[0]
            good_right_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&
                             (nonzerox>=win_xright_low)&
                             (nonzerox<win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit
        if plot==True:
            ploty=np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
            left_fitx=left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            out_img=np.dstack((
                frame_sliding_window,frame_sliding_window,(
                frame_sliding_window )))*255

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]]=[255,0,0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] =[0, 0, 255]

            figure,(ax1,ax2,ax3)=plt.subplot(3,1)
            figure.set_size_inches(10,10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame,cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window,cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx,ploty,color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped frame with sliding windows")
            ax3.set_title("Detected Lanes with Sliding Windows")
            plt.show()
        return self.left_fit,self.right_fit
    def get_line_markings(self,frame=None):
        if frame is None:
            frame=self.orig_frame
        hls=cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
        _,sxbinary=edge.threshold(hls[:,:,1],thresh=(120,255))
        sxbinary = binary_array(hls[:, :, 1], thresh=(120, 255))
        sxbinary=edge.blur_gaussian(sxbinary,ksize=3)
        s_channel=hls[:,:,2]
        _,s_binary=edge.threshold(s_channel,(80,255))
        _,r_thresh=edge.threshold(frame[:,:,2],thresh=(120,255))
        rs_binary=cv2.bitwise_and(s_binary,r_thresh)
        road_edges = mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
        self.lane_line_markings=cv2.bitwise_and(s_binary,r_thresh)
        return self.lane_line_markings


    def histogram_peak(self):
        midpoint=np.int(self.histogram.shape[0]/2)
        leftx_base=np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[:midpoint])+midpoint
        return leftx_base,rightx_base
    def overlay_lane_lines(self,plot=False):
        if self.left_fitx is None or self.right_fitx is None or self.ploty is None:
            print("Error: Missing lane line data (left_fitx, right_fitx, or ploty is None).")
            return self.orig_frame
        warp_zero=np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp=np.dstack((warp_zero,warp_zero,warp_zero))
        pts_left=np.array([np.transpose(np.vstack([self.left_fitx,self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
            self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                self.orig_frame.shape[
                    1], self.orig_frame.shape[0]))

        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if plot == True:
                # Plot the figures
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")
            ax2.set_title("Original Frame With Lane Overlay")
            plt.show()

        return result
    def perspective_transform(self,frame=None,plot=False):
        if frame is None:
            frame=self.lane_line_markings
        self.transformation_matrix=cv2.getPerspectiveTransform(
            self.roi_points,self.desired_roi_points)

        self.inv_transformation_matrix=cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points)

        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=(
                cv2.INTER_LINEAR))

        (thresh, binary_warped) = cv2.threshold(
                    self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                        self.desired_roi_points]), True, (147, 20, 255), 3)
            while(1):
                cv2.imshow('Warped Image', warped_plot)
                if cv2.waitKey(0):
                    break
            cv2.destroyAllWindows()
        return  self.warped_frame
    def plot_roi(self,frame=None,plot=False):
        if plot==False:
            return
        if frame is None:
            frame=self.orig_frame.copy()
        this_image=cv2.polylines(frame,np.int32([self.roi_points]),True,(147,20,255),3)

        while(1):
            cv2.imshow('ROI Image', this_image)
            if cv2.waitKey(0):
                break
        cv2.destroyAllWindows()


def main():
    filename=r"C:/folder9.6/college/IVth year/major-minor project/custom dataset/train/JPEGImages/10 (17).jpg"
    # Load a frame (or image)
    original_frame = cv2.imread(filename)

    # Create a Lane object
    lane_obj = Lane(orig_frame=original_frame)

    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()

    # Plot the region of interest on the image
    lane_obj.plot_roi(plot=True)

    # Perform the perspective transform to generate a bird's eye view
    # If Plot == True, show image with new region of interest
    warped_frame = lane_obj.perspective_transform(plot=True)

    # Generate the image histogram to serve as a starting point
    # for finding lane line pixels
    histogram = lane_obj.calculate_histogram(plot=True)

    # Find lane line pixels using the sliding window method

    # Fill in the lane line


    # Overlay lines on the original frame
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=True)

    # Calculate lane line curvature (left and right lane lines)
    lane_obj.calculate_curvature(print_to_terminal=True)



    # Calculate center offset
    lane_obj.calculate_car_position(print_to_terminal=True)

    # Display curvature and center offset on image
    frame_with_lane_lines2 = lane_obj.display_curvature_offset(
        frame=frame_with_lane_lines, plot=True)

    # Create the output file name by removing the '.jpg' part
    size = len(filename)
    new_filename = filename[:size - 4]
    new_filename = new_filename + '_thresholded1.jpg'

    # Save the new image in the working directory
    cv2.imwrite(new_filename, lane_line_markings)

    # Display the image
    cv2.imshow("Image", lane_line_markings)

    # Display the window until any key is pressed
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()


main()

















