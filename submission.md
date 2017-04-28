
## Advanced Lane Finding Project
## by Reinaldo Maslim

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## Undistorting camera image calibration

First the object points in 3D cartesian coordinate of the chessboard has to be given, with the assumption of the chessboard on x-y plane with z=0. As the same chessboard is used for all images, the obj points are the same for all test images. Then for each image, using the cv2.findChessboardCorners() function to provide corners of each grid in the chessboard. Combining both object points and corners for all images in two lists, we then feed these lists to calibrateCamera() function to compute camera calibration and distortion coefficients. Afterwards, subsequent images can be corrected by undistort() function.


```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs=cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#gray.shape[::-1] returns image height and width like 960, 1280. can also be img.shape[0:2]
#dist:distortion coeff, mtx:camera matrix 3D-2D, rvecs-tvecs: rotation and translation vectors from world to image

dst=cv2.undistort(img, mtx, dist, None, mtx)
```


```python
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
img = cv2.imread('../camera_cal/calibration1.jpg')
dst=cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imwrite('../camera_cal/undist1.jpg',dst)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_2_0.png)


## Create thresholded binary image

I used color transform the images into HLS color space. By using the saturation S-channel, a binary image is formed by thresholding the colors in this channel. On the other hand, I also performed Sobel along x-axis to grayscale image to preserve vertical gradient lines and eliminate horizontal lines. 

These two binary images are combined together as an input to the next pipeline.


```python
def undistort(image_raw):
    
    image=cv2.undistort(image_raw, mtx, dist, None, mtx)

    return image
```


```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately

    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 3) Calculate the magnitude
    abs_sobelxy=np.sqrt(np.add(np.square(abs_sobelx), np.square(abs_sobely)))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1]) ]= 1
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad=np.arctan2(abs_sobely, abs_sobelx)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(grad)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(grad >= thresh[0]) & (grad <=thresh[1]) ]= 1
    return binary_output


def get_combined_color(img):

    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]   

    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 200
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    
    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    #color_binary = np.dstack((u_binary, s_binary, l_binary))
    
    color_binary = np.zeros_like(s_binary)
    color_binary[(l_binary == 1) | (b_binary == 1)| (s_binary==1)] = 1   
    
    
    return color_binary

```

##  Region of interest masking and perspective transform

Perpective transform is applied using a warp() function with image input. The source src and destination dst polygon is hardcoded as follows:

```
    src=np.float32(
        [[image.shape[1]/2-100, image.shape[0]-270],
        [image.shape[1]/2+100, image.shape[0]-270],
        [image.shape[1]/2+650, image.shape[0]],
        [image.shape[1]/2-650, image.shape[0]]])

    dst=np.float32([
        [image.shape[1]*1/7, 0],
        [image.shape[1]*6/7, 0],
        [image.shape[1]*6/7, image.shape[0]],
        [image.shape[1]*1/7, image.shape[0]]])

```

The perpective transform is verified by checking that straight road will correspond to straight vertical lines on the warped image. Additionally, I mask the warped image by region_of_interest() function that removes pixels information outside this polygon. The vertices of this polygon is set as:

```
    vertices = np.array([[(img.shape[1]/2-80, img.shape[0]-280),
            (img.shape[1]/2+80, img.shape[0]-280),
            (img.shape[1]/2+670, img.shape[0]),
            (img.shape[1]/2-670, img.shape[0])]], dtype=np.int32)
  
```
By doing this, we eliminate bushes or walls that lies close to the lane lines.


```python
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 1

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def warp(img):
    global N, Ninv
    img_size=(img.shape[1], image.shape[0])
    
    src=np.float32(
        [[image.shape[1]/2-100, image.shape[0]-270],
        [image.shape[1]/2+100, image.shape[0]-270],
        [image.shape[1]/2+650, image.shape[0]],
        [image.shape[1]/2-650, image.shape[0]]])
    
    dst=np.float32([
        [image.shape[1]*1/7, 0],
        [image.shape[1]*6/7, 0],
        [image.shape[1]*6/7, image.shape[0]],
        [image.shape[1]*1/7, image.shape[0]]])
    
    N=cv2.getPerspectiveTransform(src, dst)

    Ninv=cv2.getPerspectiveTransform(dst, src)

    warped=cv2.warpPerspective(img, N, img_size, flags=cv2.INTER_LINEAR)
    
    return warped


```

## Identifying lane line pixels and fitting polynomial

To identify lane-line pixels, sliding window is adopted with 9 windows with width expanded size of margin to left and right. First, the starting positions of window is determined from left and right peaks of histogram along x-axis summed from lower 30% of the image. If sufficient number of pixels are inside this window, then we recenter the window. Both windows are propagated upwards while recentering to the mean x-value at every steps. 

After pixels belonging to both lanes are found, polyfit function is applied to obtain a 2nd order polynomial for each lane. 


```python
def blind_search(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom 0.3 of the image
    histogram = np.sum(binary_warped[np.int(0.7*binary_warped.shape[0]):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8)*255
    #print(np.max(out_img))
    #plt.imshow(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    pointfour = np.int(0.4*histogram.shape[0])
    pointsix=np.int(0.6*histogram.shape[0])
    leftx_base = np.argmax(histogram[:pointfour]) #get the peak from first half(left)
    rightx_base = np.argmax(histogram[pointsix:]) + pointsix #get peak from secondhalf(right)
    
    # Choose the number of sliding windows
    nwindows = 9 
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin =  90
    # Set minimum number of pixels found to recenter window
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    return left_fit, right_fit, left_fitx, right_fitx, out_img
    

```

## Computing radius of curvature and position of car from the center

Radius of curvature of each lane can be computed from the resulting polynomial. As both polynomials previously fitted is in pixels unit, we have to convert it into meters by these corresponding conversions.

```
    y-axis
    720 pixels = 30 m

    x-axis
    700 pixels = 3.7 m
```

Therefore, the radius of curvature can be obtained as:
```
    radius = ((1 + (2*fit_curve[0]*y_eval + fit_curve[1])**2)**1.5)/np.absolute(2*fit_curve[0])
```

Meanwhile, the position of car from the center can also be calculated by comparing the mid section of left-right lanes to the image center. Similarly, we have to convert px units to m.


```
    position = image.shape[1]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    offset_from_center=(position - center)*xm_per_pix

```


```python
def find_radius(ploty, fitx):
    
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    
    fit_curve = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
    radius = ((1 + (2*fit_curve[0]*y_eval + fit_curve[1])**2)**1.5) \
                                 /np.absolute(2*fit_curve[0])
    return radius

def find_centre(pts):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = image.shape[1]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 500)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 500)][:,1])
    
    
    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    
    return (position - center)*xm_per_pix
```


```python
def draw_poly(image, binary_warped, left_fitx, right_fitx):
    y_len=720
    ploty = np.linspace(0, y_len-1, y_len)
    # Draw green lane onto original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane as green onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Ninv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistort(image), 1, newwarp, 0.3, 0)

    left_curverad=find_radius(ploty, left_fitx)
    right_curverad=find_radius(ploty, right_fitx)
    
    pts = np.argwhere(newwarp[:,:,1])
    position=find_centre(pts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text="Left curve %f, right curve %f" %(left_curverad,right_curverad)
    cv2.putText(result,text,(100,100), font, 1, (255,255,255), 2)
    
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    result=cv2.putText(result, text, (100, 50), font,  1, (255,255,255), 2)
    
    return result
```


```python
def prepare(image_raw):
    image=cv2.undistort(image_raw, mtx, dist, None, mtx)
    kernel_size = 5
    img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # Convert to HLS color space and separate the S channel
    ksize=5
    sxbinary=abs_sobel_thresh(img, thresh=(20, 100))
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 120))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(.65, 1.05))

    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    combined_color=get_combined_color(img)

    combined_binary=np.zeros_like(combined)
    combined_binary[(combined_color > 0) | (combined > 0)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    #green is combined gradient_thresh, blue is s_binary
    color_binary = np.dstack((np.zeros_like(gray), combined, combined_color))

    vertices = np.array([[(img.shape[1]/2-80, img.shape[0]-280),
            (img.shape[1]/2+80, img.shape[0]-280),
            (img.shape[1]/2+670, img.shape[0]),
            (img.shape[1]/2-670, img.shape[0])]], dtype=np.int32)

    masked_img=region_of_interest(combined_binary, vertices)
    
    warped_img=np.absolute(warp(masked_img))
    
    return warped_img
```


```python
#Test prepare() function and pipeline()

y_len=720
ploty = np.linspace(0, y_len-1, y_len)

for i in range(1,6):
    fname = '../test_images/test{}.jpg'.format(i)
    image = cv2.imread(fname)
    image= cv2.resize(image, (1280, 720)) 
    result = prepare(image)
    _, _, left_fitx, right_fitx, out_img=  blind_search(result)  
    
    
    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    #print(np.max(image))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(out_img)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    #ax2.xlim(0, img.shape[1])
    #ax2.ylim(img.shape[0], 0)
    
    ax2.set_title('Prepare result', fontsize=40)
 
    ax3.imshow(draw_poly(image, result, left_fitx, right_fitx))
    #ax3.imshow(out_img)
    ax3.set_title('Pipeline image', fontsize=40)
   
   
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
for i in range(1,11):
    fname = '../test_images/hard{}.jpg'.format(i)
    image = cv2.imread(fname)
    image= cv2.resize(image, (1280, 720)) 
    result = prepare(image)
    _, _, left_fitx, right_fitx, out_img= blind_search(result)  
    

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    #print(np.max(image))
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(out_img)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    #ax2.xlim(0, img.shape[1])
    #ax2.ylim(img.shape[0], 0)
    
    ax2.set_title('Prepare Result', fontsize=40)
    
    ax3.imshow(draw_poly(image, result, left_fitx, right_fitx))
    ax3.set_title('Pipeline image', fontsize=40)
 
   
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    

```


![png](output_14_0.png)



![png](output_14_1.png)



![png](output_14_2.png)



![png](output_14_3.png)



![png](output_14_4.png)



![png](output_14_5.png)



![png](output_14_6.png)



![png](output_14_7.png)



![png](output_14_8.png)



![png](output_14_9.png)



![png](output_14_10.png)



![png](output_14_11.png)



![png](output_14_12.png)



![png](output_14_13.png)



![png](output_14_14.png)



```python
#Normal pipeline without considering previous frames

def pipeline(image):
    
    prepared_result=prepare(image)
    _, _, left_fitx, right_fitx, _=  blind_search(prepared_result) 
    
    result=draw_poly(image, prepared_result, left_fitx, right_fitx)
    
    return result
    
```

## Advanced pipeline

In order to eliminate false lanes, lanes information from the previous frames can be utilized to predict, smoothen, and check current lanes. To do this, line class is defined to store information for left and right lanes. 

The class holds several important variables of the previous frames' lanes and setters functions to update variables.


```python
# Define a class to receive the characteristics of each line detection
class Line():
    #lanes polynomials are in pixels
    def __init__(self):
        #number of iterations to be averaged
        self.n_iterations=30    
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line, size=n x image.shape[0]
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations, size= image.shape[0]
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #counter for sanity
        self.sanity_count=0
        
    def setRecentXFitted(self, recent_xfitted):
        self.recent_xfitted = recent_xfitted
        
    def setBestX(self, bestx):
        self.bestx = bestx        

    def setBestFit(self, best_fit):
        self.best_fit = best_fit
        
    def setCurrentFit(self, current_fit):
        self.current_fit = current_fit
        
    def setRadius(self, radius_of_curvature):
        self.radius_of_curvature = radius_of_curvature
        
    def setDetected(self, detected):
        self.detected = detected
        
    def setSanityCount(self, sanity_count):
        self.sanity_count = sanity_count
        
print("line class is defined")
```

    line class is defined


## Sanity check

Sanity check consist of three layers, namely:

1. Curvature similarity: 
    comparing radius of curvature of current to previous lane

2. Offset from lane: 
    thresholding error between curvature centroids of current and previous lane

3. Parallelness to previous best fit: 
    calculate mean squared error between current and previous lane

## Smoothen and update lane

If a new lane is detected, update lane parameters. Calculate the smoothened lane by averaging curve points over the last n iterations, then fit a polynomial to this averaged curve. 

If no new lane is detected, use old best polynomial averaged over last n iterations.


```python
#define necessary auxillary functions

def check_sanity(lane, fitx):
    if lane.radius_of_curvature is None:
        #newly reset line
        return True
    
    ploty = np.linspace(0, len(fitx)-1, len(fitx))
    
    curve_threshold=[0.5, 2]
    horizontal_threshold=0.8
    mse_threshold=3000
    
    #check for curvature similarity
    current_radius=find_radius(ploty, fitx)
    #print("----")
    #print(current_radius)
    #print(lane.radius_of_curvature)
    #print("----")
    if (abs(current_radius/lane.radius_of_curvature)<curve_threshold[1] and abs(current_radius/lane.radius_of_curvature)>curve_threshold[0]) and (current_radius<30000 and current_radius>200):
        #similar curvature
        mid_position = image.shape[1]/2
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        #check horizontal distance
        if abs(np.mean(fitx)-np.mean(lane.bestx))*xm_per_pix<horizontal_threshold:
            #separated correctly
            #check if roughly parallel
            mse = ((fitx - lane.bestx) ** 2).mean(axis=0)
            #print("mse")
            #print(mse)
            if mse<mse_threshold:
                #good fit
                return True
            else:
                #print("not parallel")
                #not parallel enough
                return False
        else:
            #print("too far offset")
            #too far offset
            return False
    else:
        #print("dissimilar curvature")
        #dissimilar curvature
        return False
    
    
def update_and_smoothen(lane, detected, newx):
    #newx is fitx result from look_ahead_search or blind_search if detected lane is sane
    y_len=720
    ploty = np.linspace(0, y_len-1, y_len)
    
    #update detected status, detected also means sane
    lane.setDetected(detected)
    
    if detected:
        #update lane parameters
        if len(lane.recent_xfitted)>=lane.n_iterations:
            #remove the first x_set on the list
            lane.recent_xfitted.pop(0)
        #add the newest x_set to the back
        lane.recent_xfitted.append(newx)
        
        lane.setRecentXFitted(lane.recent_xfitted)
        
        #using the latest n x values, update bestx as mean along axis-0 
        lane.bestx=np.mean(np.asarray(lane.recent_xfitted), 0)
        lane.setBestX(lane.bestx)
        #from bestx calculate best_fit polynomial coefficients
        lane.setBestFit(np.polyfit(ploty, lane.bestx, 2))
        smooth_fitx = lane.best_fit[0]*ploty**2 + lane.best_fit[1]*ploty + lane.best_fit[2]
        
        
        lane.setCurrentFit(np.polyfit(ploty, newx, 2))
        lane.setRadius(find_radius(ploty, smooth_fitx))
        
        return smooth_fitx
    else:
        old_smooth_fitx = lane.best_fit[0]*ploty**2 + lane.best_fit[1]*ploty + lane.best_fit[2]
        lane.setSanityCount(lane.sanity_count+1)
        
        return old_smooth_fitx
    
```

## Look ahead search

Once lanes are detected on the previous frame, we don't need to search blindly but instead search around these lanes. To perform this, I create a filter with margin offset from previous lane fit. Then, the binary birdview image is masked with this filter before performing blind_search() over this masked binary image. 

If either previous lane fits are not available, I did not mask the corresponding half.


```python
def look_ahead_search(binary_warped, left_fit, right_fit):
    
    margin=75
    mask=np.zeros_like(binary_warped)
    #mask the image based on left and right polynomials
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        cv2.fillPoly(mask, np.int_([left_line_pts]), 1)
    else:
        mask[:, :int(mask.shape[1]/2)]=1
        
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(mask, np.int_([right_line_pts]), 1)
    else:
        mask[:, int(mask.shape[1]/2):]=1

    
    masked_binary=cv2.bitwise_and(binary_warped, mask)
    
    #feed masked image to blind search
    left_fit, right_fit, left_fitx, right_fitx, _=blind_search(masked_binary)

    return left_fit, right_fit, left_fitx, right_fitx
```

## Get fit 

To get right and left fit, as their information are independent of another; sanity check, look ahead search, or blind search must be performed separately for each of them. The get_fit() function acts as decision making in all scenarios to use previous functions to get current lanes.

If lane on either side has not been detected for the past sane_threshold frames, the lane is reset as a new Line().


```python
def get_fit(binary_warped, lane, mode):
    global left_lane, right_lane
    sane_threshold=10
    #mode 0=left, 1=right
    #print(lane.detected)
    if lane.detected is True:
        #if line previously detected
        #use look ahead search
        left_fit, right_fit, left_fitx, right_fitx=look_ahead_search(binary_warped, left_lane.best_fit, right_lane.best_fit)
        
        if mode==0:
            fitx=left_fitx
        else:
            fitx=right_fitx
            
        if check_sanity(lane, fitx):
            smoothen_fitx=update_and_smoothen(lane, True, fitx)
            #print("look ahead")
            return smoothen_fitx
        
        #use blind search 
        left_fit, right_fit, left_fitx, right_fitx, _=blind_search(binary_warped)
        
        if mode==0:
            fitx=left_fitx
        else:
            fitx=right_fitx
            
        if check_sanity(lane, fitx):
            smoothen_fitx=update_and_smoothen(lane, True, fitx)
            #print("blind")
            return smoothen_fitx
        #print("old")
        old_smoothen_fitx=update_and_smoothen(lane, False, None)
        return old_smoothen_fitx
    
    else:
        #if lane not detected previously
        if lane.sanity_count>sane_threshold:
            #reset line
            if mode==0:
                left_lane=Line()
            else:
                right_lane=Line()
        
        #use blind search 
        left_fit, right_fit, left_fitx, right_fitx, _=blind_search(binary_warped)
        
        if mode==0:
            fitx=left_fitx
        else:
            fitx=right_fitx
            
        if check_sanity(lane, fitx):
            smoothen_fitx=update_and_smoothen(lane, True, fitx)
            #print("blind")
            return smoothen_fitx
        
        #print("old")
        old_smoothen_fitx=update_and_smoothen(lane, False, None)
        return old_smoothen_fitx  
```


```python
left_lane=Line()
right_lane=Line()

def advanced_pipeline(image):
    
    #prepare image by warping and masking into binary image
    binary_warped=prepare(image)
    
    left_fitx=get_fit(binary_warped, left_lane, 0)
    right_fitx=get_fit(binary_warped, right_lane, 1)
    
    
    result=draw_poly(image, binary_warped, left_fitx, right_fitx)
    
    return result
    
```

## Normal pipeline vs Advanced pipeline

The following cell outputs results from both normal pipeline and advanced pipeline(that takes previous frames into consideration). The images are sequenced with short time-steps for advanced pipeline to work. 

From the results, it is noticed that advance pipeline results in a smoother and stable than normal pipeline.


```python
for i in range(1,26):
    fname = '../test_images/Screenshot from project_video.mp4 - {}.png'.format(i)
    image = cv2.imread(fname)
    #print(image.shape)
    image= cv2.resize(image, (1280, 720)) 

    # Plot the result
    f, (ax2, ax3) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    #ax1.imshow(image)
    #ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(pipeline(image))    
    ax2.set_title('Normal pipeline result', fontsize=40)
    
    ax3.imshow(advanced_pipeline(image))
    ax3.set_title('Advanced pipeline result', fontsize=40)
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    

    
```

    /home/rm/miniconda2/envs/carnd-term1/lib/python3.5/site-packages/matplotlib/pyplot.py:524: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      max_open_warning, RuntimeWarning)



![png](output_26_1.png)



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)



![png](output_26_5.png)



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)



![png](output_26_9.png)



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)



![png](output_26_13.png)



![png](output_26_14.png)



![png](output_26_15.png)



![png](output_26_16.png)



![png](output_26_17.png)



![png](output_26_18.png)



![png](output_26_19.png)



![png](output_26_20.png)



![png](output_26_21.png)



![png](output_26_22.png)



![png](output_26_23.png)



![png](output_26_24.png)



![png](output_26_25.png)


## Generate result videos

First video: normal pipeline on project_video.mp4
Second video: advanced pipeline project_video.mp4
Third video: normal pipeline on challenge_video.mp4
Fourth video: advanced pipeline on challenge_video.mp4


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
mask_with_poly=False
```


```python
project_output = '../project_video_result_normal.mp4'
#clip1 = VideoFileClip("../project_video.mp4")
#project_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
#%time project_clip.write_videofile(project_output, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))
```





<video width="960" height="540" controls>
  <source src="../project_video_result_normal.mp4">
</video>





```python
#reset lines
left_lane=Line()
right_lane=Line()


project_output = '../project_video_result_advanced.mp4'
#clip1 = VideoFileClip("../project_video.mp4")
#project_clip = clip1.fl_image(advanced_pipeline) #NOTE: this function expects color images!!
#%time project_clip.write_videofile(project_output, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))
```





<video width="960" height="540" controls>
  <source src="../project_video_result_advanced.mp4">
</video>




## Challenge video


In the section below, I attempted the challenge video with new sets of parameters (region_of_interest, warping, color thresholding, and Line() parameter). From trial and error, the results are shown in below as:

Normal pipeline:
challenge_video_result_normal.mp4

Advanced pipeline:
challenge_video_result_advanced.mp4

The normal pipeline outputs on an image sequence can also be seen below. Notice that the last image could not identify the right lane due to the abscence of detectable lane. 

From the videos, we can see that the advanced pipeline outputs a more reasonable and stable result.



```python
def warp(img):
    global N, Ninv
    img_size=(img.shape[1], image.shape[0])
    
    src=np.float32(
        [[image.shape[1]/2-100, image.shape[0]-250],
        [image.shape[1]/2+100, image.shape[0]-250],
        [image.shape[1]/2+650, image.shape[0]],
        [image.shape[1]/2-650, image.shape[0]]])
    
    dst=np.float32([
        [image.shape[1]*1/7, 0],
        [image.shape[1]*6/7, 0],
        [image.shape[1]*6/7, image.shape[0]],
        [image.shape[1]*1/7, image.shape[0]]])
    
    N=cv2.getPerspectiveTransform(src, dst)

    Ninv=cv2.getPerspectiveTransform(dst, src)

    warped=cv2.warpPerspective(img, N, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

def get_combined_color(img):

    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]   

    # Threshold color channel
    s_thresh_min = 80
    s_thresh_max = 230
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    
    l_thresh_min = 200
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
    

    #color_binary = np.dstack((u_binary, s_binary, l_binary))
    
    color_binary = np.zeros_like(s_binary)
    color_binary[(l_binary == 1) | (b_binary == 1)| (s_binary==1)] = 1   
    
    
    return color_binary

def find_radius(ploty, fitx):
    
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    
    fit_curve = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
    radius = ((1 + (2*fit_curve[0]*y_eval + fit_curve[1])**2)**1.5) \
                                 /np.absolute(2*fit_curve[0])
    return radius

def find_centre(pts):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = image.shape[1]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 500)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 500)][:,1])
    
    
    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    
    return (position - center)*xm_per_pix

#change mask
def prepare(image_raw):
    image=cv2.undistort(image_raw, mtx, dist, None, mtx)
    kernel_size = 5
    img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # Convert to HLS color space and separate the S channel
    ksize=5
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(.20, 1.15))

    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1))| ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary == 1) & (dir_binary == 1))] = 1
    combined_color=get_combined_color(img)

    combined_binary=np.zeros_like(combined)
    combined_binary[(combined_color > 0) | (combined > 0)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    #green is combined gradient_thresh, blue is s_binary
    color_binary = np.dstack((np.zeros_like(gray), combined, combined_color))

    vertices = np.array([[(img.shape[1]/2-20, img.shape[0]-270),
            (img.shape[1]/2+80, img.shape[0]-270),
            (img.shape[1]/2+520, img.shape[0]),
            (img.shape[1]/2-440, img.shape[0])]], dtype=np.int32)
    
    masked_img=region_of_interest(combined_binary, vertices)
    
    warped_img=np.absolute(warp(masked_img))
    
    #warped_img=np.absolute(warp(combined_binary))
    
    return warped_img


def blind_search(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom 0.3 of the image
    histogram = np.sum(binary_warped[np.int(0.7*binary_warped.shape[0]):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8)*255
    #print(np.max(out_img))
    #plt.imshow(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    pointfour = np.int(0.4*histogram.shape[0])
    pointsix=np.int(0.6*histogram.shape[0])
    leftx_base = np.argmax(histogram[:pointfour]) #get the peak from first half(left)
    rightx_base = np.argmax(histogram[pointsix:]) + pointsix #get peak from secondhalf(right)
    
    # Choose the number of sliding windows
    nwindows = 9 
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin =  110
    # Set minimum number of pixels found to recenter window
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    return left_fit, right_fit, left_fitx, right_fitx, out_img


def draw_poly(image, binary_warped, left_fitx, right_fitx):
    y_len=720
    ploty = np.linspace(0, y_len-1, y_len)
    # Draw green lane onto original image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane as green onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Ninv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistort(image), 1, newwarp, 0.3, 0)

    left_curverad=find_radius(ploty, left_fitx)
    right_curverad=find_radius(ploty, right_fitx)
    
    pts = np.argwhere(newwarp[:,:,1])
    position=find_centre(pts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text="Left curve %f, right curve %f" %(left_curverad,right_curverad)
    cv2.putText(result,text,(100,100), font, 1, (255,255,255), 2)
    
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    result=cv2.putText(result, text, (100, 50), font,  1, (255,255,255), 2)
    
    return result



def pipeline(image):
    
    prepared_result=prepare(image)
    _, _, left_fitx, right_fitx,out_img=  blind_search(prepared_result) 
    
    result=draw_poly(image, prepared_result, left_fitx, right_fitx)
    
    #vertices = np.array([[(img.shape[1]/2-20, img.shape[0]-270),
    #        (img.shape[1]/2+80, img.shape[0]-270),
    #        (img.shape[1]/2+520, img.shape[0]),
    #        (img.shape[1]/2-440, img.shape[0])]], dtype=np.int32)
    
    #result=region_of_interest(image, vertices)
    #out_img = np.dstack((prepared_result,prepared_result, prepared_result)).astype(np.uint8)*255
    #result=out_img
    return result
```


```python
y_len=720
ploty = np.linspace(0, y_len-1, y_len)

for i in range(1,15):
    fname = '../test_images/Screenshot from challenge_video.mp4 - {}.png'.format(i)
    image = cv2.imread(fname)
    image= cv2.resize(image, (1280, 720)) 
    
    result = prepare(image)
    _, _, left_fitx, right_fitx, out_img=  blind_search(result)  
    
    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(out_img)
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax2.set_title('Prepare result', fontsize=40)
 

    ax3.imshow(pipeline(image))
    ax3.set_title('Pipeline image', fontsize=40)
   
   
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_35_0.png)



![png](output_35_1.png)



![png](output_35_2.png)



![png](output_35_3.png)



![png](output_35_4.png)



![png](output_35_5.png)



![png](output_35_6.png)



![png](output_35_7.png)



![png](output_35_8.png)



![png](output_35_9.png)



![png](output_35_10.png)



![png](output_35_11.png)



![png](output_35_12.png)



![png](output_35_13.png)



```python
project_output = '../challenge_video_normal_result.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")
project_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
%time project_clip.write_videofile(project_output, audio=False)
```

    [MoviePy] >>>> Building video ../challenge_video_normal_result.mp4
    [MoviePy] Writing video ../challenge_video_normal_result.mp4


    
      0%|          | 0/485 [00:00<?, ?it/s][A
      0%|          | 1/485 [00:00<02:07,  3.79it/s][A
      0%|          | 2/485 [00:00<02:07,  3.79it/s][A
      1%|          | 3/485 [00:00<02:02,  3.92it/s][A
      1%|          | 4/485 [00:00<01:59,  4.02it/s][A
      1%|          | 5/485 [00:01<02:01,  3.96it/s][A
      1%|          | 6/485 [00:01<02:01,  3.95it/s][A
      1%|▏         | 7/485 [00:01<01:57,  4.08it/s][A
      2%|▏         | 8/485 [00:01<01:54,  4.17it/s][A
      2%|▏         | 9/485 [00:02<01:57,  4.06it/s][A
      2%|▏         | 10/485 [00:02<01:53,  4.19it/s][A
      2%|▏         | 11/485 [00:02<01:50,  4.28it/s][A
      2%|▏         | 12/485 [00:02<01:49,  4.34it/s][A
      3%|▎         | 13/485 [00:03<01:47,  4.37it/s][A
      3%|▎         | 14/485 [00:03<01:46,  4.40it/s][A
      3%|▎         | 15/485 [00:03<01:46,  4.40it/s][A
      3%|▎         | 16/485 [00:03<01:45,  4.43it/s][A
      4%|▎         | 17/485 [00:04<01:49,  4.26it/s][A
      4%|▎         | 18/485 [00:04<01:49,  4.26it/s][A
      4%|▍         | 19/485 [00:04<01:54,  4.06it/s][A
      4%|▍         | 20/485 [00:04<02:00,  3.85it/s][A
      4%|▍         | 21/485 [00:05<02:02,  3.79it/s][A
      5%|▍         | 22/485 [00:05<02:02,  3.79it/s][A
      5%|▍         | 23/485 [00:05<01:55,  3.99it/s][A
      5%|▍         | 24/485 [00:05<01:51,  4.13it/s][A
      5%|▌         | 25/485 [00:06<01:47,  4.26it/s][A
      5%|▌         | 26/485 [00:06<01:45,  4.34it/s][A
      6%|▌         | 27/485 [00:06<01:43,  4.41it/s][A
      6%|▌         | 28/485 [00:06<01:42,  4.46it/s][A
      6%|▌         | 29/485 [00:06<01:41,  4.51it/s][A
      6%|▌         | 30/485 [00:07<01:44,  4.34it/s][A
      6%|▋         | 31/485 [00:07<01:43,  4.40it/s][A
      7%|▋         | 32/485 [00:07<01:41,  4.44it/s][A
      7%|▋         | 33/485 [00:07<01:41,  4.47it/s][A
      7%|▋         | 34/485 [00:08<01:40,  4.47it/s][A
      7%|▋         | 35/485 [00:08<01:44,  4.32it/s][A
      7%|▋         | 36/485 [00:08<01:48,  4.14it/s][A
      8%|▊         | 37/485 [00:08<01:44,  4.28it/s][A
      8%|▊         | 38/485 [00:08<01:42,  4.36it/s][A
      8%|▊         | 39/485 [00:09<01:41,  4.39it/s][A
      8%|▊         | 40/485 [00:09<01:42,  4.35it/s][A
      8%|▊         | 41/485 [00:09<01:46,  4.18it/s][A
      9%|▊         | 42/485 [00:09<01:44,  4.23it/s][A
      9%|▉         | 43/485 [00:10<01:53,  3.90it/s][A
      9%|▉         | 44/485 [00:10<01:53,  3.88it/s][A
      9%|▉         | 45/485 [00:10<01:50,  3.98it/s][A
      9%|▉         | 46/485 [00:11<01:52,  3.91it/s][A
     10%|▉         | 47/485 [00:11<01:53,  3.85it/s][A
     10%|▉         | 48/485 [00:11<01:49,  3.98it/s][A
     10%|█         | 49/485 [00:11<01:50,  3.95it/s][A
     10%|█         | 50/485 [00:12<01:47,  4.05it/s][A
     11%|█         | 51/485 [00:12<01:45,  4.10it/s][A
     11%|█         | 52/485 [00:12<01:47,  4.04it/s][A
     11%|█         | 53/485 [00:12<01:46,  4.06it/s][A
     11%|█         | 54/485 [00:12<01:44,  4.14it/s][A
     11%|█▏        | 55/485 [00:13<01:42,  4.19it/s][A
     12%|█▏        | 56/485 [00:13<01:40,  4.26it/s][A
     12%|█▏        | 57/485 [00:13<01:44,  4.11it/s][A
     12%|█▏        | 58/485 [00:13<01:41,  4.19it/s][A
     12%|█▏        | 59/485 [00:14<01:42,  4.15it/s][A
     12%|█▏        | 60/485 [00:14<01:43,  4.12it/s][A
     13%|█▎        | 61/485 [00:14<01:45,  4.01it/s][A
     13%|█▎        | 62/485 [00:14<01:43,  4.09it/s][A
     13%|█▎        | 63/485 [00:15<01:41,  4.15it/s][A
     13%|█▎        | 64/485 [00:15<01:42,  4.09it/s][A
     13%|█▎        | 65/485 [00:15<01:43,  4.06it/s][A
     14%|█▎        | 66/485 [00:15<01:41,  4.11it/s][A
     14%|█▍        | 67/485 [00:16<01:39,  4.20it/s][A
     14%|█▍        | 68/485 [00:16<01:37,  4.27it/s][A
     14%|█▍        | 69/485 [00:16<01:36,  4.31it/s][A
     14%|█▍        | 70/485 [00:16<01:35,  4.36it/s][A
     15%|█▍        | 71/485 [00:17<01:34,  4.40it/s][A
     15%|█▍        | 72/485 [00:17<01:38,  4.19it/s][A
     15%|█▌        | 73/485 [00:17<01:38,  4.20it/s][A
     15%|█▌        | 74/485 [00:17<01:36,  4.27it/s][A
     15%|█▌        | 75/485 [00:17<01:34,  4.32it/s][A
     16%|█▌        | 76/485 [00:18<01:35,  4.29it/s][A
     16%|█▌        | 77/485 [00:18<01:34,  4.30it/s][A
     16%|█▌        | 78/485 [00:18<01:37,  4.16it/s][A
     16%|█▋        | 79/485 [00:18<01:35,  4.25it/s][A
     16%|█▋        | 80/485 [00:19<01:36,  4.18it/s][A
     17%|█▋        | 81/485 [00:19<01:37,  4.12it/s][A
     17%|█▋        | 82/485 [00:19<01:37,  4.13it/s][A
     17%|█▋        | 83/485 [00:19<01:39,  4.03it/s][A
     17%|█▋        | 84/485 [00:20<01:39,  4.04it/s][A
     18%|█▊        | 85/485 [00:20<01:38,  4.05it/s][A
     18%|█▊        | 86/485 [00:20<01:35,  4.17it/s][A
     18%|█▊        | 87/485 [00:20<01:34,  4.22it/s][A
     18%|█▊        | 88/485 [00:21<01:36,  4.12it/s][A
     18%|█▊        | 89/485 [00:21<01:34,  4.19it/s][A
     19%|█▊        | 90/485 [00:21<01:35,  4.14it/s][A
     19%|█▉        | 91/485 [00:21<01:34,  4.17it/s][A
     19%|█▉        | 92/485 [00:22<01:36,  4.07it/s][A
     19%|█▉        | 93/485 [00:22<01:37,  4.00it/s][A
     19%|█▉        | 94/485 [00:22<01:34,  4.12it/s][A
     20%|█▉        | 95/485 [00:22<01:33,  4.16it/s][A
     20%|█▉        | 96/485 [00:23<01:36,  4.03it/s][A
     20%|██        | 97/485 [00:23<01:36,  4.02it/s][A
     20%|██        | 98/485 [00:23<01:34,  4.09it/s][A
     20%|██        | 99/485 [00:23<01:31,  4.20it/s][A
     21%|██        | 100/485 [00:24<01:33,  4.12it/s][A
     21%|██        | 101/485 [00:24<01:34,  4.06it/s][A
     21%|██        | 102/485 [00:24<01:33,  4.08it/s][A
     21%|██        | 103/485 [00:24<01:32,  4.13it/s][A
     21%|██▏       | 104/485 [00:25<01:35,  3.98it/s][A
     22%|██▏       | 105/485 [00:25<01:34,  4.02it/s][A
     22%|██▏       | 106/485 [00:25<01:32,  4.11it/s][A
     22%|██▏       | 107/485 [00:25<01:31,  4.14it/s][A
     22%|██▏       | 108/485 [00:25<01:31,  4.10it/s][A
     22%|██▏       | 109/485 [00:26<01:29,  4.18it/s][A
     23%|██▎       | 110/485 [00:26<01:31,  4.09it/s][A
     23%|██▎       | 111/485 [00:26<01:29,  4.19it/s][A
     23%|██▎       | 112/485 [00:26<01:32,  4.01it/s][A
     23%|██▎       | 113/485 [00:27<01:30,  4.09it/s][A
     24%|██▎       | 114/485 [00:27<01:31,  4.05it/s][A
     24%|██▎       | 115/485 [00:27<01:31,  4.03it/s][A
     24%|██▍       | 116/485 [00:27<01:30,  4.07it/s][A
     24%|██▍       | 117/485 [00:28<01:30,  4.05it/s][A
     24%|██▍       | 118/485 [00:28<01:29,  4.12it/s][A
     25%|██▍       | 119/485 [00:28<01:27,  4.18it/s][A
     25%|██▍       | 120/485 [00:28<01:29,  4.10it/s][A
     25%|██▍       | 121/485 [00:29<01:31,  3.99it/s][A
     25%|██▌       | 122/485 [00:29<01:29,  4.06it/s][A
     25%|██▌       | 123/485 [00:29<01:27,  4.13it/s][A
     26%|██▌       | 124/485 [00:29<01:27,  4.12it/s][A
     26%|██▌       | 125/485 [00:30<01:28,  4.09it/s][A
     26%|██▌       | 126/485 [00:30<01:30,  3.97it/s][A
     26%|██▌       | 127/485 [00:30<01:30,  3.95it/s][A
     26%|██▋       | 128/485 [00:30<01:27,  4.09it/s][A
     27%|██▋       | 129/485 [00:31<01:28,  4.00it/s][A
     27%|██▋       | 130/485 [00:31<01:27,  4.04it/s][A
     27%|██▋       | 131/485 [00:31<01:25,  4.15it/s][A
     27%|██▋       | 132/485 [00:31<01:26,  4.07it/s][A
     27%|██▋       | 133/485 [00:32<01:28,  3.96it/s][A
     28%|██▊       | 134/485 [00:32<01:28,  3.97it/s][A
     28%|██▊       | 135/485 [00:32<01:25,  4.10it/s][A
     28%|██▊       | 136/485 [00:32<01:25,  4.07it/s][A
     28%|██▊       | 137/485 [00:33<01:28,  3.92it/s][A
     28%|██▊       | 138/485 [00:33<01:25,  4.05it/s][A
     29%|██▊       | 139/485 [00:33<01:24,  4.10it/s][A
     29%|██▉       | 140/485 [00:33<01:27,  3.94it/s][A
     29%|██▉       | 141/485 [00:34<01:28,  3.88it/s][A
     29%|██▉       | 142/485 [00:34<01:26,  3.95it/s][A
     29%|██▉       | 143/485 [00:34<01:23,  4.09it/s][A
     30%|██▉       | 144/485 [00:34<01:24,  4.03it/s][A
     30%|██▉       | 145/485 [00:35<01:23,  4.08it/s][A
     30%|███       | 146/485 [00:35<01:23,  4.06it/s][A
     30%|███       | 147/485 [00:35<01:27,  3.86it/s][A
     31%|███       | 148/485 [00:35<01:26,  3.88it/s][A
     31%|███       | 149/485 [00:36<01:25,  3.91it/s][A
     31%|███       | 150/485 [00:36<01:23,  3.99it/s][A
     31%|███       | 151/485 [00:36<01:23,  3.99it/s][A
     31%|███▏      | 152/485 [00:36<01:24,  3.92it/s][A
     32%|███▏      | 153/485 [00:37<01:25,  3.90it/s][A
     32%|███▏      | 154/485 [00:37<01:24,  3.93it/s][A
     32%|███▏      | 155/485 [00:37<01:26,  3.80it/s][A
     32%|███▏      | 156/485 [00:37<01:26,  3.82it/s][A
     32%|███▏      | 157/485 [00:38<01:26,  3.78it/s][A
     33%|███▎      | 158/485 [00:38<01:23,  3.90it/s][A
     33%|███▎      | 159/485 [00:38<01:29,  3.65it/s][A
     33%|███▎      | 160/485 [00:39<01:31,  3.53it/s][A
     33%|███▎      | 161/485 [00:39<01:32,  3.49it/s][A
     33%|███▎      | 162/485 [00:39<01:35,  3.37it/s][A
     34%|███▎      | 163/485 [00:40<01:34,  3.42it/s][A
     34%|███▍      | 164/485 [00:40<01:34,  3.39it/s][A
     34%|███▍      | 165/485 [00:40<01:29,  3.59it/s][A
     34%|███▍      | 166/485 [00:40<01:26,  3.68it/s][A
     34%|███▍      | 167/485 [00:41<01:27,  3.64it/s][A
     35%|███▍      | 168/485 [00:41<01:29,  3.55it/s][A
     35%|███▍      | 169/485 [00:41<01:24,  3.76it/s][A
     35%|███▌      | 170/485 [00:41<01:19,  3.95it/s][A
     35%|███▌      | 171/485 [00:42<01:20,  3.91it/s][A
     35%|███▌      | 172/485 [00:42<01:18,  3.99it/s][A
     36%|███▌      | 173/485 [00:42<01:18,  3.98it/s][A
     36%|███▌      | 174/485 [00:42<01:16,  4.07it/s][A
     36%|███▌      | 175/485 [00:43<01:15,  4.11it/s][A
     36%|███▋      | 176/485 [00:43<01:15,  4.10it/s][A
     36%|███▋      | 177/485 [00:43<01:15,  4.07it/s][A
     37%|███▋      | 178/485 [00:43<01:17,  3.96it/s][A
     37%|███▋      | 179/485 [00:44<01:15,  4.04it/s][A
     37%|███▋      | 180/485 [00:44<01:16,  3.98it/s][A
     37%|███▋      | 181/485 [00:44<01:16,  3.97it/s][A
     38%|███▊      | 182/485 [00:44<01:15,  4.00it/s][A
     38%|███▊      | 183/485 [00:45<01:15,  4.00it/s][A
     38%|███▊      | 184/485 [00:45<01:16,  3.93it/s][A
     38%|███▊      | 185/485 [00:45<01:16,  3.93it/s][A
     38%|███▊      | 186/485 [00:45<01:14,  4.03it/s][A
     39%|███▊      | 187/485 [00:46<01:13,  4.03it/s][A
     39%|███▉      | 188/485 [00:46<01:14,  3.99it/s][A
     39%|███▉      | 189/485 [00:46<01:12,  4.11it/s][A
     39%|███▉      | 190/485 [00:46<01:10,  4.19it/s][A
     39%|███▉      | 191/485 [00:47<01:14,  3.97it/s][A
     40%|███▉      | 192/485 [00:47<01:12,  4.04it/s][A
     40%|███▉      | 193/485 [00:47<01:11,  4.08it/s][A
     40%|████      | 194/485 [00:47<01:09,  4.20it/s][A
     40%|████      | 195/485 [00:48<01:11,  4.06it/s][A
     40%|████      | 196/485 [00:48<01:10,  4.07it/s][A
     41%|████      | 197/485 [00:48<01:09,  4.17it/s][A
     41%|████      | 198/485 [00:48<01:12,  3.94it/s][A
     41%|████      | 199/485 [00:49<01:12,  3.96it/s][A
     41%|████      | 200/485 [00:49<01:10,  4.04it/s][A
     41%|████▏     | 201/485 [00:49<01:08,  4.16it/s][A
     42%|████▏     | 202/485 [00:49<01:07,  4.21it/s][A
     42%|████▏     | 203/485 [00:49<01:08,  4.10it/s][A
     42%|████▏     | 204/485 [00:50<01:08,  4.10it/s][A
     42%|████▏     | 205/485 [00:50<01:06,  4.23it/s][A
     42%|████▏     | 206/485 [00:50<01:05,  4.24it/s][A
     43%|████▎     | 207/485 [00:50<01:06,  4.17it/s][A
     43%|████▎     | 208/485 [00:51<01:05,  4.20it/s][A
     43%|████▎     | 209/485 [00:51<01:06,  4.14it/s][A
     43%|████▎     | 210/485 [00:51<01:07,  4.06it/s][A
     44%|████▎     | 211/485 [00:51<01:09,  3.93it/s][A
     44%|████▎     | 212/485 [00:52<01:07,  4.02it/s][A
     44%|████▍     | 213/485 [00:52<01:06,  4.07it/s][A
     44%|████▍     | 214/485 [00:52<01:04,  4.17it/s][A
     44%|████▍     | 215/485 [00:52<01:05,  4.12it/s][A
     45%|████▍     | 216/485 [00:53<01:05,  4.11it/s][A
     45%|████▍     | 217/485 [00:53<01:05,  4.10it/s][A
     45%|████▍     | 218/485 [00:53<01:04,  4.14it/s][A
     45%|████▌     | 219/485 [00:53<01:07,  3.97it/s][A
     45%|████▌     | 220/485 [00:54<01:05,  4.03it/s][A
     46%|████▌     | 221/485 [00:54<01:04,  4.10it/s][A
     46%|████▌     | 222/485 [00:54<01:04,  4.07it/s][A
     46%|████▌     | 223/485 [00:54<01:04,  4.06it/s][A
     46%|████▌     | 224/485 [00:55<01:05,  3.99it/s][A
     46%|████▋     | 225/485 [00:55<01:03,  4.11it/s][A
     47%|████▋     | 226/485 [00:55<01:04,  4.03it/s][A
     47%|████▋     | 227/485 [00:55<01:04,  3.98it/s][A
     47%|████▋     | 228/485 [00:56<01:03,  4.04it/s][A
     47%|████▋     | 229/485 [00:56<01:03,  4.03it/s][A
     47%|████▋     | 230/485 [00:56<01:02,  4.08it/s][A
     48%|████▊     | 231/485 [00:56<01:01,  4.14it/s][A
     48%|████▊     | 232/485 [00:57<01:02,  4.06it/s][A
     48%|████▊     | 233/485 [00:57<01:01,  4.08it/s][A
     48%|████▊     | 234/485 [00:57<01:00,  4.12it/s][A
     48%|████▊     | 235/485 [00:57<01:02,  4.01it/s][A
     49%|████▊     | 236/485 [00:58<01:03,  3.91it/s][A
     49%|████▉     | 237/485 [00:58<01:02,  3.99it/s][A
     49%|████▉     | 238/485 [00:58<00:59,  4.14it/s][A
     49%|████▉     | 239/485 [00:58<00:59,  4.16it/s][A
     49%|████▉     | 240/485 [00:59<00:57,  4.26it/s][A
     50%|████▉     | 241/485 [00:59<00:57,  4.24it/s][A
     50%|████▉     | 242/485 [00:59<00:56,  4.32it/s][A
     50%|█████     | 243/485 [00:59<00:57,  4.23it/s][A
     50%|█████     | 244/485 [00:59<00:58,  4.14it/s][A
     51%|█████     | 245/485 [01:00<00:58,  4.12it/s][A
     51%|█████     | 246/485 [01:00<00:57,  4.16it/s][A
     51%|█████     | 247/485 [01:00<00:59,  4.00it/s][A
     51%|█████     | 248/485 [01:00<00:58,  4.07it/s][A
     51%|█████▏    | 249/485 [01:01<00:57,  4.08it/s][A
     52%|█████▏    | 250/485 [01:01<00:56,  4.18it/s][A
     52%|█████▏    | 251/485 [01:01<00:57,  4.08it/s][A
     52%|█████▏    | 252/485 [01:01<00:57,  4.07it/s][A
     52%|█████▏    | 253/485 [01:02<00:55,  4.14it/s][A
     52%|█████▏    | 254/485 [01:02<00:56,  4.06it/s][A
     53%|█████▎    | 255/485 [01:02<00:58,  3.94it/s][A
     53%|█████▎    | 256/485 [01:02<00:58,  3.92it/s][A
     53%|█████▎    | 257/485 [01:03<00:57,  3.94it/s][A
     53%|█████▎    | 258/485 [01:03<00:56,  4.00it/s][A
     53%|█████▎    | 259/485 [01:03<00:57,  3.95it/s][A
     54%|█████▎    | 260/485 [01:03<00:57,  3.91it/s][A
     54%|█████▍    | 261/485 [01:04<00:55,  4.04it/s][A
     54%|█████▍    | 262/485 [01:04<00:53,  4.15it/s][A
     54%|█████▍    | 263/485 [01:04<00:53,  4.11it/s][A
     54%|█████▍    | 264/485 [01:04<00:52,  4.21it/s][A
     55%|█████▍    | 265/485 [01:05<00:52,  4.21it/s][A
     55%|█████▍    | 266/485 [01:05<00:51,  4.22it/s][A
     55%|█████▌    | 267/485 [01:05<00:52,  4.19it/s][A
     55%|█████▌    | 268/485 [01:05<00:50,  4.26it/s][A
     55%|█████▌    | 269/485 [01:06<00:50,  4.31it/s][A
     56%|█████▌    | 270/485 [01:06<00:49,  4.36it/s][A
     56%|█████▌    | 271/485 [01:06<00:49,  4.30it/s][A
     56%|█████▌    | 272/485 [01:06<00:50,  4.23it/s][A
     56%|█████▋    | 273/485 [01:07<00:49,  4.26it/s][A
     56%|█████▋    | 274/485 [01:07<00:49,  4.22it/s][A
     57%|█████▋    | 275/485 [01:07<00:50,  4.19it/s][A
     57%|█████▋    | 276/485 [01:07<00:50,  4.15it/s][A
     57%|█████▋    | 277/485 [01:07<00:49,  4.23it/s][A
     57%|█████▋    | 278/485 [01:08<00:49,  4.22it/s][A
     58%|█████▊    | 279/485 [01:08<00:51,  4.03it/s][A
     58%|█████▊    | 280/485 [01:08<00:50,  4.06it/s][A
     58%|█████▊    | 281/485 [01:08<00:50,  4.01it/s][A
     58%|█████▊    | 282/485 [01:09<00:49,  4.09it/s][A
     58%|█████▊    | 283/485 [01:09<00:48,  4.14it/s][A
     59%|█████▊    | 284/485 [01:09<00:48,  4.12it/s][A
     59%|█████▉    | 285/485 [01:09<00:47,  4.18it/s][A
     59%|█████▉    | 286/485 [01:10<00:48,  4.12it/s][A
     59%|█████▉    | 287/485 [01:10<00:47,  4.16it/s][A
     59%|█████▉    | 288/485 [01:10<00:47,  4.16it/s][A
     60%|█████▉    | 289/485 [01:10<00:49,  3.96it/s][A
     60%|█████▉    | 290/485 [01:11<00:49,  3.96it/s][A
     60%|██████    | 291/485 [01:11<00:50,  3.86it/s][A
     60%|██████    | 292/485 [01:11<00:49,  3.86it/s][A
     60%|██████    | 293/485 [01:11<00:50,  3.77it/s][A
     61%|██████    | 294/485 [01:12<00:49,  3.89it/s][A
     61%|██████    | 295/485 [01:12<00:48,  3.92it/s][A
     61%|██████    | 296/485 [01:12<00:46,  4.06it/s][A
     61%|██████    | 297/485 [01:12<00:44,  4.18it/s][A
     61%|██████▏   | 298/485 [01:13<00:45,  4.13it/s][A
     62%|██████▏   | 299/485 [01:13<00:44,  4.13it/s][A
     62%|██████▏   | 300/485 [01:13<00:45,  4.08it/s][A
     62%|██████▏   | 301/485 [01:13<00:44,  4.11it/s][A
     62%|██████▏   | 302/485 [01:14<00:44,  4.13it/s][A
     62%|██████▏   | 303/485 [01:14<00:44,  4.13it/s][A
     63%|██████▎   | 304/485 [01:14<00:43,  4.20it/s][A
     63%|██████▎   | 305/485 [01:14<00:43,  4.13it/s][A
     63%|██████▎   | 306/485 [01:15<00:43,  4.09it/s][A
     63%|██████▎   | 307/485 [01:15<00:42,  4.19it/s][A
     64%|██████▎   | 308/485 [01:15<00:41,  4.25it/s][A
     64%|██████▎   | 309/485 [01:15<00:42,  4.12it/s][A
     64%|██████▍   | 310/485 [01:16<00:42,  4.07it/s][A
     64%|██████▍   | 311/485 [01:16<00:42,  4.13it/s][A
     64%|██████▍   | 312/485 [01:16<00:41,  4.17it/s][A
     65%|██████▍   | 313/485 [01:16<00:42,  4.07it/s][A
     65%|██████▍   | 314/485 [01:17<00:42,  4.01it/s][A
     65%|██████▍   | 315/485 [01:17<00:41,  4.10it/s][A
     65%|██████▌   | 316/485 [01:17<00:40,  4.18it/s][A
     65%|██████▌   | 317/485 [01:17<00:41,  4.01it/s][A
     66%|██████▌   | 318/485 [01:18<00:41,  4.07it/s][A
     66%|██████▌   | 319/485 [01:18<00:40,  4.11it/s][A
     66%|██████▌   | 320/485 [01:18<00:39,  4.16it/s][A
     66%|██████▌   | 321/485 [01:18<00:39,  4.12it/s][A
     66%|██████▋   | 322/485 [01:19<00:39,  4.08it/s][A
     67%|██████▋   | 323/485 [01:19<00:40,  4.04it/s][A
     67%|██████▋   | 324/485 [01:19<00:39,  4.10it/s][A
     67%|██████▋   | 325/485 [01:19<00:40,  3.96it/s][A
     67%|██████▋   | 326/485 [01:20<00:39,  4.02it/s][A
     67%|██████▋   | 327/485 [01:20<00:38,  4.09it/s][A
     68%|██████▊   | 328/485 [01:20<00:38,  4.10it/s][A
     68%|██████▊   | 329/485 [01:20<00:38,  4.10it/s][A
     68%|██████▊   | 330/485 [01:20<00:37,  4.13it/s][A
     68%|██████▊   | 331/485 [01:21<00:36,  4.23it/s][A
     68%|██████▊   | 332/485 [01:21<00:35,  4.29it/s][A
     69%|██████▊   | 333/485 [01:21<00:35,  4.27it/s][A
     69%|██████▉   | 334/485 [01:21<00:36,  4.18it/s][A
     69%|██████▉   | 335/485 [01:22<00:35,  4.19it/s][A
     69%|██████▉   | 336/485 [01:22<00:35,  4.20it/s][A
     69%|██████▉   | 337/485 [01:22<00:35,  4.15it/s][A
     70%|██████▉   | 338/485 [01:22<00:35,  4.16it/s][A
     70%|██████▉   | 339/485 [01:23<00:34,  4.21it/s][A
     70%|███████   | 340/485 [01:23<00:35,  4.07it/s][A
     70%|███████   | 341/485 [01:23<00:35,  4.11it/s][A
     71%|███████   | 342/485 [01:23<00:34,  4.14it/s][A
     71%|███████   | 343/485 [01:24<00:34,  4.16it/s][A
     71%|███████   | 344/485 [01:24<00:33,  4.16it/s][A
     71%|███████   | 345/485 [01:24<00:33,  4.17it/s][A
     71%|███████▏  | 346/485 [01:24<00:33,  4.10it/s][A
     72%|███████▏  | 347/485 [01:25<00:33,  4.07it/s][A
     72%|███████▏  | 348/485 [01:25<00:33,  4.06it/s][A
     72%|███████▏  | 349/485 [01:25<00:33,  4.08it/s][A
     72%|███████▏  | 350/485 [01:25<00:33,  3.98it/s][A
     72%|███████▏  | 351/485 [01:26<00:32,  4.07it/s][A
     73%|███████▎  | 352/485 [01:26<00:32,  4.07it/s][A
     73%|███████▎  | 353/485 [01:26<00:32,  4.03it/s][A
     73%|███████▎  | 354/485 [01:26<00:32,  3.99it/s][A
     73%|███████▎  | 355/485 [01:27<00:31,  4.07it/s][A
     73%|███████▎  | 356/485 [01:27<00:30,  4.17it/s][A
     74%|███████▎  | 357/485 [01:27<00:30,  4.19it/s][A
     74%|███████▍  | 358/485 [01:27<00:29,  4.28it/s][A
     74%|███████▍  | 359/485 [01:27<00:29,  4.28it/s][A
     74%|███████▍  | 360/485 [01:28<00:28,  4.34it/s][A
     74%|███████▍  | 361/485 [01:28<00:28,  4.35it/s][A
     75%|███████▍  | 362/485 [01:28<00:28,  4.31it/s][A
     75%|███████▍  | 363/485 [01:28<00:29,  4.17it/s][A
     75%|███████▌  | 364/485 [01:29<00:29,  4.04it/s][A
     75%|███████▌  | 365/485 [01:29<00:30,  3.91it/s][A
     75%|███████▌  | 366/485 [01:29<00:30,  3.95it/s][A
     76%|███████▌  | 367/485 [01:29<00:28,  4.10it/s][A
     76%|███████▌  | 368/485 [01:30<00:28,  4.17it/s][A
     76%|███████▌  | 369/485 [01:30<00:28,  4.12it/s][A
     76%|███████▋  | 370/485 [01:30<00:28,  4.08it/s][A
     76%|███████▋  | 371/485 [01:30<00:27,  4.20it/s][A
     77%|███████▋  | 372/485 [01:31<00:27,  4.10it/s][A
     77%|███████▋  | 373/485 [01:31<00:27,  4.08it/s][A
     77%|███████▋  | 374/485 [01:31<00:26,  4.14it/s][A
     77%|███████▋  | 375/485 [01:31<00:27,  4.06it/s][A
     78%|███████▊  | 376/485 [01:32<00:26,  4.12it/s][A
     78%|███████▊  | 377/485 [01:32<00:26,  4.01it/s][A
     78%|███████▊  | 378/485 [01:32<00:26,  4.05it/s][A
     78%|███████▊  | 379/485 [01:32<00:25,  4.18it/s][A
     78%|███████▊  | 380/485 [01:33<00:25,  4.19it/s][A
     79%|███████▊  | 381/485 [01:33<00:24,  4.19it/s][A
     79%|███████▉  | 382/485 [01:33<00:24,  4.15it/s][A
     79%|███████▉  | 383/485 [01:33<00:24,  4.20it/s][A
     79%|███████▉  | 384/485 [01:34<00:23,  4.25it/s][A
     79%|███████▉  | 385/485 [01:34<00:24,  4.12it/s][A
     80%|███████▉  | 386/485 [01:34<00:24,  4.05it/s][A
     80%|███████▉  | 387/485 [01:34<00:23,  4.18it/s][A
     80%|████████  | 388/485 [01:34<00:22,  4.25it/s][A
     80%|████████  | 389/485 [01:35<00:22,  4.21it/s][A
     80%|████████  | 390/485 [01:35<00:23,  4.12it/s][A
     81%|████████  | 391/485 [01:35<00:22,  4.15it/s][A
     81%|████████  | 392/485 [01:35<00:22,  4.09it/s][A
     81%|████████  | 393/485 [01:36<00:22,  4.08it/s][A
     81%|████████  | 394/485 [01:36<00:22,  4.01it/s][A
     81%|████████▏ | 395/485 [01:36<00:22,  4.07it/s][A
     82%|████████▏ | 396/485 [01:36<00:21,  4.15it/s][A
     82%|████████▏ | 397/485 [01:37<00:21,  4.04it/s][A
     82%|████████▏ | 398/485 [01:37<00:21,  3.95it/s][A
     82%|████████▏ | 399/485 [01:37<00:21,  4.07it/s][A
     82%|████████▏ | 400/485 [01:37<00:20,  4.13it/s][A
     83%|████████▎ | 401/485 [01:38<00:20,  4.06it/s][A
     83%|████████▎ | 402/485 [01:38<00:20,  4.05it/s][A
     83%|████████▎ | 403/485 [01:38<00:19,  4.10it/s][A
     83%|████████▎ | 404/485 [01:38<00:19,  4.07it/s][A
     84%|████████▎ | 405/485 [01:39<00:19,  4.09it/s][A
     84%|████████▎ | 406/485 [01:39<00:19,  4.13it/s][A
     84%|████████▍ | 407/485 [01:39<00:18,  4.22it/s][A
     84%|████████▍ | 408/485 [01:39<00:17,  4.29it/s][A
     84%|████████▍ | 409/485 [01:40<00:17,  4.27it/s][A
     85%|████████▍ | 410/485 [01:40<00:18,  4.16it/s][A
     85%|████████▍ | 411/485 [01:40<00:18,  4.10it/s][A
     85%|████████▍ | 412/485 [01:40<00:17,  4.13it/s][A
     85%|████████▌ | 413/485 [01:41<00:17,  4.12it/s][A
     85%|████████▌ | 414/485 [01:41<00:18,  3.93it/s][A
     86%|████████▌ | 415/485 [01:41<00:17,  4.01it/s][A
     86%|████████▌ | 416/485 [01:41<00:16,  4.10it/s][A
     86%|████████▌ | 417/485 [01:42<00:16,  4.03it/s][A
     86%|████████▌ | 418/485 [01:42<00:16,  4.08it/s][A
     86%|████████▋ | 419/485 [01:42<00:15,  4.13it/s][A
     87%|████████▋ | 420/485 [01:42<00:15,  4.21it/s][A
     87%|████████▋ | 421/485 [01:43<00:15,  4.07it/s][A
     87%|████████▋ | 422/485 [01:43<00:15,  4.11it/s][A
     87%|████████▋ | 423/485 [01:43<00:14,  4.17it/s][A
     87%|████████▋ | 424/485 [01:43<00:14,  4.22it/s][A
     88%|████████▊ | 425/485 [01:43<00:14,  4.14it/s][A
     88%|████████▊ | 426/485 [01:44<00:14,  4.06it/s][A
     88%|████████▊ | 427/485 [01:44<00:13,  4.16it/s][A
     88%|████████▊ | 428/485 [01:44<00:13,  4.16it/s][A
     88%|████████▊ | 429/485 [01:44<00:13,  4.15it/s][A
     89%|████████▊ | 430/485 [01:45<00:12,  4.23it/s][A
     89%|████████▉ | 431/485 [01:45<00:12,  4.19it/s][A
     89%|████████▉ | 432/485 [01:45<00:13,  4.00it/s][A
     89%|████████▉ | 433/485 [01:45<00:13,  3.98it/s][A
     89%|████████▉ | 434/485 [01:46<00:12,  4.04it/s][A
     90%|████████▉ | 435/485 [01:46<00:12,  4.13it/s][A
     90%|████████▉ | 436/485 [01:46<00:11,  4.16it/s][A
     90%|█████████ | 437/485 [01:46<00:11,  4.10it/s][A
     90%|█████████ | 438/485 [01:47<00:11,  4.13it/s][A
     91%|█████████ | 439/485 [01:47<00:11,  4.15it/s][A
     91%|█████████ | 440/485 [01:47<00:11,  4.01it/s][A
     91%|█████████ | 441/485 [01:47<00:10,  4.01it/s][A
     91%|█████████ | 442/485 [01:48<00:10,  4.07it/s][A
     91%|█████████▏| 443/485 [01:48<00:10,  4.00it/s][A
     92%|█████████▏| 444/485 [01:48<00:10,  4.06it/s][A
     92%|█████████▏| 445/485 [01:48<00:09,  4.05it/s][A
     92%|█████████▏| 446/485 [01:49<00:09,  4.15it/s][A
     92%|█████████▏| 447/485 [01:49<00:09,  4.19it/s][A
     92%|█████████▏| 448/485 [01:49<00:08,  4.28it/s][A
     93%|█████████▎| 449/485 [01:49<00:08,  4.23it/s][A
     93%|█████████▎| 450/485 [01:50<00:08,  4.17it/s][A
     93%|█████████▎| 451/485 [01:50<00:07,  4.26it/s][A
     93%|█████████▎| 452/485 [01:50<00:07,  4.27it/s][A
     93%|█████████▎| 453/485 [01:50<00:07,  4.19it/s][A
     94%|█████████▎| 454/485 [01:51<00:07,  4.13it/s][A
     94%|█████████▍| 455/485 [01:51<00:07,  4.10it/s][A
     94%|█████████▍| 456/485 [01:51<00:07,  4.12it/s][A
     94%|█████████▍| 457/485 [01:51<00:06,  4.03it/s][A
     94%|█████████▍| 458/485 [01:52<00:06,  4.08it/s][A
     95%|█████████▍| 459/485 [01:52<00:06,  4.08it/s][A
     95%|█████████▍| 460/485 [01:52<00:06,  4.05it/s][A
     95%|█████████▌| 461/485 [01:52<00:05,  4.10it/s][A
     95%|█████████▌| 462/485 [01:52<00:05,  4.08it/s][A
     95%|█████████▌| 463/485 [01:53<00:05,  4.20it/s][A
     96%|█████████▌| 464/485 [01:53<00:04,  4.22it/s][A
     96%|█████████▌| 465/485 [01:53<00:04,  4.15it/s][A
     96%|█████████▌| 466/485 [01:53<00:04,  4.12it/s][A
     96%|█████████▋| 467/485 [01:54<00:04,  4.11it/s][A
     96%|█████████▋| 468/485 [01:54<00:04,  3.99it/s][A
     97%|█████████▋| 469/485 [01:54<00:03,  4.05it/s][A
     97%|█████████▋| 470/485 [01:54<00:03,  3.98it/s][A
     97%|█████████▋| 471/485 [01:55<00:03,  4.04it/s][A
     97%|█████████▋| 472/485 [01:55<00:03,  4.09it/s][A
     98%|█████████▊| 473/485 [01:55<00:02,  4.18it/s][A
     98%|█████████▊| 474/485 [01:55<00:02,  4.21it/s][A
     98%|█████████▊| 475/485 [01:56<00:02,  4.06it/s][A
     98%|█████████▊| 476/485 [01:56<00:02,  4.15it/s][A
     98%|█████████▊| 477/485 [01:56<00:01,  4.25it/s][A
     99%|█████████▊| 478/485 [01:56<00:01,  4.30it/s][A
     99%|█████████▉| 479/485 [01:57<00:01,  4.23it/s][A
     99%|█████████▉| 480/485 [01:57<00:01,  4.22it/s][A
     99%|█████████▉| 481/485 [01:57<00:00,  4.30it/s][A
     99%|█████████▉| 482/485 [01:57<00:00,  4.35it/s][A
    100%|█████████▉| 483/485 [01:57<00:00,  4.35it/s][A
    100%|█████████▉| 484/485 [01:58<00:00,  4.32it/s][A
    100%|██████████| 485/485 [01:58<00:00,  4.18it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: ../challenge_video_normal_result.mp4 
    
    CPU times: user 7min 21s, sys: 1.38 s, total: 7min 22s
    Wall time: 1min 59s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))
```





<video width="960" height="540" controls>
  <source src="../challenge_video_normal_result.mp4">
</video>





```python
def advanced_pipeline(image):
    
    #prepare image by warping and masking into binary image
    binary_warped=prepare(image)
    
    left_fitx=get_fit(binary_warped, left_lane, 0)
    right_fitx=get_fit(binary_warped, right_lane, 1)
    
    
    result=draw_poly(image, binary_warped, left_fitx, right_fitx)
    
    return result
```


```python
#reset lines
left_lane=Line()
right_lane=Line()

project_output = '../challenge_video_result_advanced.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")
project_clip = clip1.fl_image(advanced_pipeline) #NOTE: this function expects color images!!
%time project_clip.write_videofile(project_output, audio=False)
```

    [MoviePy] >>>> Building video ../challenge_video_result_advanced.mp4
    [MoviePy] Writing video ../challenge_video_result_advanced.mp4


    
      0%|          | 0/485 [00:00<?, ?it/s][A
      0%|          | 1/485 [00:00<02:13,  3.63it/s][A
      0%|          | 2/485 [00:00<02:14,  3.59it/s][A
      1%|          | 3/485 [00:00<02:10,  3.69it/s][A
      1%|          | 4/485 [00:01<02:05,  3.85it/s][A
      1%|          | 5/485 [00:01<02:01,  3.95it/s][A
      1%|          | 6/485 [00:01<01:59,  4.01it/s][A
      1%|▏         | 7/485 [00:01<02:01,  3.92it/s][A
      2%|▏         | 8/485 [00:02<02:02,  3.91it/s][A
      2%|▏         | 9/485 [00:02<02:02,  3.88it/s][A
      2%|▏         | 10/485 [00:02<02:04,  3.81it/s][A
      2%|▏         | 11/485 [00:02<02:05,  3.78it/s][A
      2%|▏         | 12/485 [00:03<02:03,  3.84it/s][A
      3%|▎         | 13/485 [00:03<02:02,  3.87it/s][A
      3%|▎         | 14/485 [00:03<02:09,  3.62it/s][A
      3%|▎         | 15/485 [00:03<02:09,  3.62it/s][A
      3%|▎         | 16/485 [00:04<02:12,  3.55it/s][A
      4%|▎         | 17/485 [00:04<02:08,  3.64it/s][A
      4%|▎         | 18/485 [00:04<02:06,  3.68it/s][A
      4%|▍         | 19/485 [00:05<02:03,  3.77it/s][A
      4%|▍         | 20/485 [00:05<02:01,  3.81it/s][A
      4%|▍         | 21/485 [00:05<02:02,  3.79it/s][A
      5%|▍         | 22/485 [00:05<02:00,  3.84it/s][A
      5%|▍         | 23/485 [00:06<02:00,  3.82it/s][A
      5%|▍         | 24/485 [00:06<02:02,  3.76it/s][A
      5%|▌         | 25/485 [00:06<02:01,  3.78it/s][A
      5%|▌         | 26/485 [00:06<02:00,  3.82it/s][A
      6%|▌         | 27/485 [00:07<02:01,  3.77it/s][A
      6%|▌         | 28/485 [00:07<02:00,  3.79it/s][A
      6%|▌         | 29/485 [00:07<02:02,  3.73it/s][A
      6%|▌         | 30/485 [00:07<02:02,  3.71it/s][A
      6%|▋         | 31/485 [00:08<02:01,  3.73it/s][A
      7%|▋         | 32/485 [00:08<02:03,  3.66it/s][A
      7%|▋         | 33/485 [00:08<02:02,  3.69it/s][A
      7%|▋         | 34/485 [00:09<02:02,  3.67it/s][A
      7%|▋         | 35/485 [00:09<02:00,  3.73it/s][A
      7%|▋         | 36/485 [00:09<02:01,  3.68it/s][A
      8%|▊         | 37/485 [00:09<02:03,  3.63it/s][A
      8%|▊         | 38/485 [00:10<02:00,  3.72it/s][A
      8%|▊         | 39/485 [00:10<01:59,  3.74it/s][A
      8%|▊         | 40/485 [00:10<01:59,  3.71it/s][A
      8%|▊         | 41/485 [00:10<01:57,  3.77it/s][A
      9%|▊         | 42/485 [00:11<01:56,  3.79it/s][A
      9%|▉         | 43/485 [00:11<02:06,  3.48it/s][A
      9%|▉         | 44/485 [00:11<02:04,  3.54it/s][A
      9%|▉         | 45/485 [00:12<02:01,  3.62it/s][A
      9%|▉         | 46/485 [00:12<02:03,  3.55it/s][A
     10%|▉         | 47/485 [00:12<02:02,  3.56it/s][A
     10%|▉         | 48/485 [00:12<01:59,  3.65it/s][A
     10%|█         | 49/485 [00:13<02:02,  3.57it/s][A
     10%|█         | 50/485 [00:13<01:58,  3.68it/s][A
     11%|█         | 51/485 [00:13<01:59,  3.64it/s][A
     11%|█         | 52/485 [00:13<02:02,  3.54it/s][A
     11%|█         | 53/485 [00:14<02:00,  3.58it/s][A
     11%|█         | 54/485 [00:14<01:59,  3.60it/s][A
     11%|█▏        | 55/485 [00:14<02:02,  3.52it/s][A
     12%|█▏        | 56/485 [00:15<02:05,  3.41it/s][A
     12%|█▏        | 57/485 [00:15<02:02,  3.48it/s][A
     12%|█▏        | 58/485 [00:15<02:00,  3.54it/s][A
     12%|█▏        | 59/485 [00:15<01:58,  3.58it/s][A
     12%|█▏        | 60/485 [00:16<02:01,  3.49it/s][A
     13%|█▎        | 61/485 [00:16<02:04,  3.40it/s][A
     13%|█▎        | 62/485 [00:16<02:08,  3.28it/s][A
     13%|█▎        | 63/485 [00:17<02:04,  3.40it/s][A
     13%|█▎        | 64/485 [00:17<02:01,  3.46it/s][A
     13%|█▎        | 65/485 [00:17<02:00,  3.48it/s][A
     14%|█▎        | 66/485 [00:18<01:56,  3.59it/s][A
     14%|█▍        | 67/485 [00:18<01:55,  3.62it/s][A
     14%|█▍        | 68/485 [00:18<01:52,  3.71it/s][A
     14%|█▍        | 69/485 [00:18<01:55,  3.60it/s][A
     14%|█▍        | 70/485 [00:19<01:58,  3.50it/s][A
     15%|█▍        | 71/485 [00:19<01:59,  3.45it/s][A
     15%|█▍        | 72/485 [00:19<02:03,  3.33it/s][A
     15%|█▌        | 73/485 [00:20<02:16,  3.01it/s][A
     15%|█▌        | 74/485 [00:20<02:15,  3.02it/s][A
     15%|█▌        | 75/485 [00:20<02:21,  2.90it/s][A
     16%|█▌        | 76/485 [00:21<02:25,  2.82it/s][A
     16%|█▌        | 77/485 [00:21<02:28,  2.74it/s][A
     16%|█▌        | 78/485 [00:21<02:26,  2.78it/s][A
     16%|█▋        | 79/485 [00:22<02:31,  2.68it/s][A
     16%|█▋        | 80/485 [00:22<02:28,  2.72it/s][A
     17%|█▋        | 81/485 [00:23<02:22,  2.83it/s][A
     17%|█▋        | 82/485 [00:23<02:24,  2.80it/s][A
     17%|█▋        | 83/485 [00:23<02:20,  2.86it/s][A
     17%|█▋        | 84/485 [00:24<02:11,  3.04it/s][A
     18%|█▊        | 85/485 [00:24<02:02,  3.28it/s][A
     18%|█▊        | 86/485 [00:24<02:00,  3.32it/s][A
     18%|█▊        | 87/485 [00:24<01:56,  3.41it/s][A
     18%|█▊        | 88/485 [00:25<01:56,  3.41it/s][A
     18%|█▊        | 89/485 [00:25<01:56,  3.39it/s][A
     19%|█▊        | 90/485 [00:25<01:52,  3.50it/s][A
     19%|█▉        | 91/485 [00:25<01:48,  3.63it/s][A
     19%|█▉        | 92/485 [00:26<01:47,  3.65it/s][A
     19%|█▉        | 93/485 [00:26<01:51,  3.52it/s][A
     19%|█▉        | 94/485 [00:26<01:50,  3.54it/s][A
     20%|█▉        | 95/485 [00:27<01:47,  3.62it/s][A
     20%|█▉        | 96/485 [00:27<01:44,  3.72it/s][A
     20%|██        | 97/485 [00:27<01:44,  3.70it/s][A
     20%|██        | 98/485 [00:27<01:43,  3.73it/s][A
     20%|██        | 99/485 [00:28<01:45,  3.66it/s][A
     21%|██        | 100/485 [00:28<01:46,  3.63it/s][A
     21%|██        | 101/485 [00:28<01:41,  3.78it/s][A
     21%|██        | 102/485 [00:28<01:37,  3.92it/s][A
     21%|██        | 103/485 [00:29<01:38,  3.87it/s][A
     21%|██▏       | 104/485 [00:29<01:40,  3.79it/s][A
     22%|██▏       | 105/485 [00:29<01:42,  3.71it/s][A
     22%|██▏       | 106/485 [00:29<01:40,  3.77it/s][A
     22%|██▏       | 107/485 [00:30<01:37,  3.90it/s][A
     22%|██▏       | 108/485 [00:30<01:39,  3.80it/s][A
     22%|██▏       | 109/485 [00:30<01:40,  3.74it/s][A
     23%|██▎       | 110/485 [00:31<01:38,  3.80it/s][A
     23%|██▎       | 111/485 [00:31<01:40,  3.73it/s][A
     23%|██▎       | 112/485 [00:31<01:40,  3.71it/s][A
     23%|██▎       | 113/485 [00:31<01:42,  3.61it/s][A
     24%|██▎       | 114/485 [00:32<01:41,  3.66it/s][A
     24%|██▎       | 115/485 [00:32<01:41,  3.64it/s][A
     24%|██▍       | 116/485 [00:32<01:40,  3.67it/s][A
     24%|██▍       | 117/485 [00:32<01:39,  3.70it/s][A
     24%|██▍       | 118/485 [00:33<01:36,  3.81it/s][A
     25%|██▍       | 119/485 [00:33<01:39,  3.66it/s][A
     25%|██▍       | 120/485 [00:33<01:41,  3.61it/s][A
     25%|██▍       | 121/485 [00:34<01:41,  3.57it/s][A
     25%|██▌       | 122/485 [00:34<01:36,  3.76it/s][A
     25%|██▌       | 123/485 [00:34<01:34,  3.84it/s][A
     26%|██▌       | 124/485 [00:34<01:35,  3.78it/s][A
     26%|██▌       | 125/485 [00:35<01:32,  3.89it/s][A
     26%|██▌       | 126/485 [00:35<01:30,  3.95it/s][A
     26%|██▌       | 127/485 [00:35<01:32,  3.88it/s][A
     26%|██▋       | 128/485 [00:35<01:30,  3.96it/s][A
     27%|██▋       | 129/485 [00:36<01:27,  4.05it/s][A
     27%|██▋       | 130/485 [00:36<01:28,  4.03it/s][A
     27%|██▋       | 131/485 [00:36<01:30,  3.92it/s][A
     27%|██▋       | 132/485 [00:36<01:30,  3.89it/s][A
     27%|██▋       | 133/485 [00:37<01:36,  3.65it/s][A
     28%|██▊       | 134/485 [00:37<01:33,  3.76it/s][A
     28%|██▊       | 135/485 [00:37<01:32,  3.79it/s][A
     28%|██▊       | 136/485 [00:37<01:32,  3.78it/s][A
     28%|██▊       | 137/485 [00:38<01:32,  3.75it/s][A
     28%|██▊       | 138/485 [00:38<01:35,  3.63it/s][A
     29%|██▊       | 139/485 [00:38<01:36,  3.58it/s][A
     29%|██▉       | 140/485 [00:39<01:38,  3.49it/s][A
     29%|██▉       | 141/485 [00:39<01:35,  3.61it/s][A
     29%|██▉       | 142/485 [00:39<01:30,  3.80it/s][A
     29%|██▉       | 143/485 [00:39<01:30,  3.76it/s][A
     30%|██▉       | 144/485 [00:40<01:30,  3.78it/s][A
     30%|██▉       | 145/485 [00:40<01:29,  3.82it/s][A
     30%|███       | 146/485 [00:40<01:26,  3.91it/s][A
     30%|███       | 147/485 [00:40<01:30,  3.75it/s][A
     31%|███       | 148/485 [00:41<01:32,  3.64it/s][A
     31%|███       | 149/485 [00:41<01:31,  3.69it/s][A
     31%|███       | 150/485 [00:41<01:30,  3.70it/s][A
     31%|███       | 151/485 [00:42<01:38,  3.40it/s][A
     31%|███▏      | 152/485 [00:42<01:41,  3.27it/s][A
     32%|███▏      | 153/485 [00:42<01:44,  3.19it/s][A
     32%|███▏      | 154/485 [00:43<01:41,  3.27it/s][A
     32%|███▏      | 155/485 [00:43<01:38,  3.35it/s][A
     32%|███▏      | 156/485 [00:43<01:39,  3.30it/s][A
     32%|███▏      | 157/485 [00:43<01:33,  3.49it/s][A
     33%|███▎      | 158/485 [00:44<01:29,  3.64it/s][A
     33%|███▎      | 159/485 [00:44<01:28,  3.68it/s][A
     33%|███▎      | 160/485 [00:44<01:30,  3.58it/s][A
     33%|███▎      | 161/485 [00:44<01:26,  3.74it/s][A
     33%|███▎      | 162/485 [00:45<01:25,  3.76it/s][A
     34%|███▎      | 163/485 [00:45<01:27,  3.66it/s][A
     34%|███▍      | 164/485 [00:45<01:29,  3.59it/s][A
     34%|███▍      | 165/485 [00:46<01:30,  3.52it/s][A
     34%|███▍      | 166/485 [00:46<01:31,  3.50it/s][A
     34%|███▍      | 167/485 [00:46<01:27,  3.63it/s][A
     35%|███▍      | 168/485 [00:46<01:27,  3.61it/s][A
     35%|███▍      | 169/485 [00:47<01:24,  3.72it/s][A
     35%|███▌      | 170/485 [00:47<01:22,  3.80it/s][A
     35%|███▌      | 171/485 [00:47<01:28,  3.53it/s][A
     35%|███▌      | 172/485 [00:47<01:27,  3.56it/s][A
     36%|███▌      | 173/485 [00:48<01:25,  3.65it/s][A
     36%|███▌      | 174/485 [00:48<01:23,  3.71it/s][A
     36%|███▌      | 175/485 [00:48<01:24,  3.68it/s][A
     36%|███▋      | 176/485 [00:49<01:22,  3.72it/s][A
     36%|███▋      | 177/485 [00:49<01:21,  3.80it/s][A
     37%|███▋      | 178/485 [00:49<01:22,  3.71it/s][A
     37%|███▋      | 179/485 [00:49<01:21,  3.75it/s][A
     37%|███▋      | 180/485 [00:50<01:21,  3.75it/s][A
     37%|███▋      | 181/485 [00:50<01:20,  3.76it/s][A
     38%|███▊      | 182/485 [00:50<01:20,  3.76it/s][A
     38%|███▊      | 183/485 [00:50<01:17,  3.89it/s][A
     38%|███▊      | 184/485 [00:51<01:19,  3.80it/s][A
     38%|███▊      | 185/485 [00:51<01:18,  3.81it/s][A
     38%|███▊      | 186/485 [00:51<01:17,  3.88it/s][A
     39%|███▊      | 187/485 [00:51<01:17,  3.86it/s][A
     39%|███▉      | 188/485 [00:52<01:19,  3.73it/s][A
     39%|███▉      | 189/485 [00:52<01:20,  3.68it/s][A
     39%|███▉      | 190/485 [00:52<01:17,  3.81it/s][A
     39%|███▉      | 191/485 [00:52<01:18,  3.76it/s][A
     40%|███▉      | 192/485 [00:53<01:18,  3.72it/s][A
     40%|███▉      | 193/485 [00:53<01:17,  3.76it/s][A
     40%|████      | 194/485 [00:53<01:16,  3.81it/s][A
     40%|████      | 195/485 [00:54<01:16,  3.77it/s][A
     40%|████      | 196/485 [00:54<01:16,  3.79it/s][A
     41%|████      | 197/485 [00:54<01:14,  3.86it/s][A
     41%|████      | 198/485 [00:54<01:14,  3.86it/s][A
     41%|████      | 199/485 [00:55<01:13,  3.90it/s][A
     41%|████      | 200/485 [00:55<01:16,  3.73it/s][A
     41%|████▏     | 201/485 [00:55<01:17,  3.66it/s][A
     42%|████▏     | 202/485 [00:55<01:14,  3.78it/s][A
     42%|████▏     | 203/485 [00:56<01:14,  3.80it/s][A
     42%|████▏     | 204/485 [00:56<01:12,  3.87it/s][A
     42%|████▏     | 205/485 [00:56<01:12,  3.87it/s][A
     42%|████▏     | 206/485 [00:56<01:13,  3.79it/s][A
     43%|████▎     | 207/485 [00:57<01:12,  3.86it/s][A
     43%|████▎     | 208/485 [00:57<01:13,  3.75it/s][A
     43%|████▎     | 209/485 [00:57<01:11,  3.86it/s][A
     43%|████▎     | 210/485 [00:57<01:09,  3.94it/s][A
     44%|████▎     | 211/485 [00:58<01:10,  3.91it/s][A
     44%|████▎     | 212/485 [00:58<01:08,  4.00it/s][A
     44%|████▍     | 213/485 [00:58<01:07,  4.04it/s][A
     44%|████▍     | 214/485 [00:58<01:08,  3.93it/s][A
     44%|████▍     | 215/485 [00:59<01:10,  3.83it/s][A
     45%|████▍     | 216/485 [00:59<01:09,  3.89it/s][A
     45%|████▍     | 217/485 [00:59<01:08,  3.89it/s][A
     45%|████▍     | 218/485 [00:59<01:08,  3.88it/s][A
     45%|████▌     | 219/485 [01:00<01:07,  3.92it/s][A
     45%|████▌     | 220/485 [01:00<01:08,  3.88it/s][A
     46%|████▌     | 221/485 [01:00<01:07,  3.90it/s][A
     46%|████▌     | 222/485 [01:01<01:10,  3.72it/s][A
     46%|████▌     | 223/485 [01:01<01:08,  3.82it/s][A
     46%|████▌     | 224/485 [01:01<01:06,  3.93it/s][A
     46%|████▋     | 225/485 [01:01<01:07,  3.86it/s][A
     47%|████▋     | 226/485 [01:02<01:07,  3.81it/s][A
     47%|████▋     | 227/485 [01:02<01:08,  3.79it/s][A
     47%|████▋     | 228/485 [01:02<01:07,  3.83it/s][A
     47%|████▋     | 229/485 [01:02<01:07,  3.77it/s][A
     47%|████▋     | 230/485 [01:03<01:07,  3.79it/s][A
     48%|████▊     | 231/485 [01:03<01:05,  3.88it/s][A
     48%|████▊     | 232/485 [01:03<01:05,  3.89it/s][A
     48%|████▊     | 233/485 [01:03<01:04,  3.89it/s][A
     48%|████▊     | 234/485 [01:04<01:03,  3.94it/s][A
     48%|████▊     | 235/485 [01:04<01:04,  3.90it/s][A
     49%|████▊     | 236/485 [01:04<01:03,  3.94it/s][A
     49%|████▉     | 237/485 [01:04<01:03,  3.91it/s][A
     49%|████▉     | 238/485 [01:05<01:06,  3.70it/s][A
     49%|████▉     | 239/485 [01:05<01:07,  3.65it/s][A
     49%|████▉     | 240/485 [01:05<01:12,  3.40it/s][A
     50%|████▉     | 241/485 [01:06<01:15,  3.23it/s][A
     50%|████▉     | 242/485 [01:06<01:19,  3.08it/s][A
     50%|█████     | 243/485 [01:06<01:22,  2.94it/s][A
     50%|█████     | 244/485 [01:07<01:21,  2.97it/s][A
     51%|█████     | 245/485 [01:07<01:19,  3.02it/s][A
     51%|█████     | 246/485 [01:07<01:15,  3.18it/s][A
     51%|█████     | 247/485 [01:08<01:09,  3.42it/s][A
     51%|█████     | 248/485 [01:08<01:04,  3.65it/s][A
     51%|█████▏    | 249/485 [01:08<01:04,  3.64it/s][A
     52%|█████▏    | 250/485 [01:08<01:04,  3.62it/s][A
     52%|█████▏    | 251/485 [01:09<01:08,  3.41it/s][A
     52%|█████▏    | 252/485 [01:09<01:12,  3.21it/s][A
     52%|█████▏    | 253/485 [01:09<01:13,  3.17it/s][A
     52%|█████▏    | 254/485 [01:10<01:18,  2.96it/s][A
     53%|█████▎    | 255/485 [01:10<01:21,  2.81it/s][A
     53%|█████▎    | 256/485 [01:11<01:25,  2.68it/s][A
     53%|█████▎    | 257/485 [01:11<01:28,  2.57it/s][A
     53%|█████▎    | 258/485 [01:11<01:23,  2.72it/s][A
     53%|█████▎    | 259/485 [01:12<01:21,  2.76it/s][A
     54%|█████▎    | 260/485 [01:12<01:18,  2.86it/s][A
     54%|█████▍    | 261/485 [01:12<01:23,  2.69it/s][A
     54%|█████▍    | 262/485 [01:13<01:20,  2.77it/s][A
     54%|█████▍    | 263/485 [01:13<01:20,  2.74it/s][A
     54%|█████▍    | 264/485 [01:14<01:25,  2.60it/s][A
     55%|█████▍    | 265/485 [01:14<01:29,  2.45it/s][A
     55%|█████▍    | 266/485 [01:14<01:28,  2.47it/s][A
     55%|█████▌    | 267/485 [01:15<01:25,  2.54it/s][A
     55%|█████▌    | 268/485 [01:15<01:20,  2.68it/s][A
     55%|█████▌    | 269/485 [01:15<01:19,  2.70it/s][A
     56%|█████▌    | 270/485 [01:16<01:17,  2.79it/s][A
     56%|█████▌    | 271/485 [01:16<01:16,  2.78it/s][A
     56%|█████▌    | 272/485 [01:17<01:15,  2.82it/s][A
     56%|█████▋    | 273/485 [01:17<01:16,  2.76it/s][A
     56%|█████▋    | 274/485 [01:17<01:26,  2.43it/s][A
     57%|█████▋    | 275/485 [01:18<01:25,  2.45it/s][A
     57%|█████▋    | 276/485 [01:18<01:24,  2.46it/s][A
     57%|█████▋    | 277/485 [01:19<01:22,  2.52it/s][A
     57%|█████▋    | 278/485 [01:19<01:18,  2.65it/s][A
     58%|█████▊    | 279/485 [01:19<01:16,  2.68it/s][A
     58%|█████▊    | 280/485 [01:20<01:15,  2.70it/s][A
     58%|█████▊    | 281/485 [01:20<01:16,  2.65it/s][A
     58%|█████▊    | 282/485 [01:20<01:16,  2.67it/s][A
     58%|█████▊    | 283/485 [01:21<01:15,  2.69it/s][A
     59%|█████▊    | 284/485 [01:21<01:17,  2.58it/s][A
     59%|█████▉    | 285/485 [01:22<01:17,  2.58it/s][A
     59%|█████▉    | 286/485 [01:22<01:14,  2.67it/s][A
     59%|█████▉    | 287/485 [01:22<01:20,  2.45it/s][A
     59%|█████▉    | 288/485 [01:23<01:17,  2.55it/s][A
     60%|█████▉    | 289/485 [01:23<01:16,  2.56it/s][A
     60%|█████▉    | 290/485 [01:24<01:14,  2.61it/s][A
     60%|██████    | 291/485 [01:24<01:11,  2.73it/s][A
     60%|██████    | 292/485 [01:24<01:08,  2.82it/s][A
     60%|██████    | 293/485 [01:25<01:07,  2.85it/s][A
     61%|██████    | 294/485 [01:25<01:08,  2.79it/s][A
     61%|██████    | 295/485 [01:25<01:08,  2.76it/s][A
     61%|██████    | 296/485 [01:26<01:10,  2.69it/s][A
     61%|██████    | 297/485 [01:26<01:09,  2.69it/s][A
     61%|██████▏   | 298/485 [01:26<01:11,  2.61it/s][A
     62%|██████▏   | 299/485 [01:27<01:11,  2.62it/s][A
     62%|██████▏   | 300/485 [01:27<01:06,  2.79it/s][A
     62%|██████▏   | 301/485 [01:27<01:02,  2.94it/s][A
     62%|██████▏   | 302/485 [01:28<01:02,  2.93it/s][A
     62%|██████▏   | 303/485 [01:28<01:01,  2.98it/s][A
     63%|██████▎   | 304/485 [01:28<01:02,  2.91it/s][A
     63%|██████▎   | 305/485 [01:29<01:06,  2.73it/s][A
     63%|██████▎   | 306/485 [01:29<01:08,  2.63it/s][A
     63%|██████▎   | 307/485 [01:30<01:10,  2.53it/s][A
     64%|██████▎   | 308/485 [01:30<01:10,  2.51it/s][A
     64%|██████▎   | 309/485 [01:30<01:08,  2.59it/s][A
     64%|██████▍   | 310/485 [01:31<01:06,  2.62it/s][A
     64%|██████▍   | 311/485 [01:31<01:06,  2.60it/s][A
     64%|██████▍   | 312/485 [01:32<01:05,  2.64it/s][A
     65%|██████▍   | 313/485 [01:32<01:04,  2.65it/s][A
     65%|██████▍   | 314/485 [01:32<01:06,  2.57it/s][A
     65%|██████▍   | 315/485 [01:33<01:10,  2.41it/s][A
     65%|██████▌   | 316/485 [01:33<01:13,  2.31it/s][A
     65%|██████▌   | 317/485 [01:34<01:11,  2.35it/s][A
     66%|██████▌   | 318/485 [01:34<01:07,  2.46it/s][A
     66%|██████▌   | 319/485 [01:34<01:05,  2.55it/s][A
     66%|██████▌   | 320/485 [01:35<01:02,  2.64it/s][A
     66%|██████▌   | 321/485 [01:35<01:02,  2.64it/s][A
     66%|██████▋   | 322/485 [01:36<01:02,  2.61it/s][A
     67%|██████▋   | 323/485 [01:36<01:00,  2.66it/s][A
     67%|██████▋   | 324/485 [01:36<01:02,  2.59it/s][A
     67%|██████▋   | 325/485 [01:37<01:04,  2.49it/s][A
     67%|██████▋   | 326/485 [01:37<01:02,  2.54it/s][A
     67%|██████▋   | 327/485 [01:38<01:00,  2.61it/s][A
     68%|██████▊   | 328/485 [01:38<00:58,  2.67it/s][A
     68%|██████▊   | 329/485 [01:38<00:57,  2.72it/s][A
     68%|██████▊   | 330/485 [01:39<00:57,  2.68it/s][A
     68%|██████▊   | 331/485 [01:39<00:56,  2.73it/s][A
     68%|██████▊   | 332/485 [01:39<00:56,  2.72it/s][A
     69%|██████▊   | 333/485 [01:40<01:02,  2.45it/s][A
     69%|██████▉   | 334/485 [01:40<01:03,  2.39it/s][A
     69%|██████▉   | 335/485 [01:41<01:00,  2.49it/s][A
     69%|██████▉   | 336/485 [01:41<00:58,  2.53it/s][A
     69%|██████▉   | 337/485 [01:41<00:58,  2.54it/s][A
     70%|██████▉   | 338/485 [01:42<00:55,  2.65it/s][A
     70%|██████▉   | 339/485 [01:42<00:55,  2.63it/s][A
     70%|███████   | 340/485 [01:43<00:54,  2.68it/s][A
     70%|███████   | 341/485 [01:43<00:53,  2.72it/s][A
     71%|███████   | 342/485 [01:43<00:52,  2.71it/s][A
     71%|███████   | 343/485 [01:44<00:58,  2.41it/s][A
     71%|███████   | 344/485 [01:44<01:00,  2.34it/s][A
     71%|███████   | 345/485 [01:45<00:59,  2.37it/s][A
     71%|███████▏  | 346/485 [01:45<00:55,  2.49it/s][A
     72%|███████▏  | 347/485 [01:45<00:54,  2.55it/s][A
     72%|███████▏  | 348/485 [01:46<00:52,  2.60it/s][A
     72%|███████▏  | 349/485 [01:46<00:53,  2.57it/s][A
     72%|███████▏  | 350/485 [01:46<00:51,  2.63it/s][A
     72%|███████▏  | 351/485 [01:47<00:53,  2.52it/s][A
     73%|███████▎  | 352/485 [01:47<00:56,  2.34it/s][A
     73%|███████▎  | 353/485 [01:48<00:54,  2.41it/s][A
     73%|███████▎  | 354/485 [01:48<00:53,  2.45it/s][A
     73%|███████▎  | 355/485 [01:48<00:48,  2.69it/s][A
     73%|███████▎  | 356/485 [01:49<00:47,  2.73it/s][A
     74%|███████▎  | 357/485 [01:49<00:47,  2.68it/s][A
     74%|███████▍  | 358/485 [01:50<00:46,  2.75it/s][A
     74%|███████▍  | 359/485 [01:50<00:44,  2.83it/s][A
     74%|███████▍  | 360/485 [01:50<00:42,  2.94it/s][A
     74%|███████▍  | 361/485 [01:51<00:43,  2.86it/s][A
     75%|███████▍  | 362/485 [01:51<00:46,  2.63it/s][A
     75%|███████▍  | 363/485 [01:51<00:46,  2.62it/s][A
     75%|███████▌  | 364/485 [01:52<00:44,  2.70it/s][A
     75%|███████▌  | 365/485 [01:52<00:45,  2.65it/s][A
     75%|███████▌  | 366/485 [01:53<00:44,  2.67it/s][A
     76%|███████▌  | 367/485 [01:53<00:43,  2.69it/s][A
     76%|███████▌  | 368/485 [01:53<00:42,  2.78it/s][A
     76%|███████▌  | 369/485 [01:54<00:40,  2.87it/s][A
     76%|███████▋  | 370/485 [01:54<00:39,  2.94it/s][A
     76%|███████▋  | 371/485 [01:54<00:43,  2.63it/s][A
     77%|███████▋  | 372/485 [01:55<00:46,  2.43it/s][A
     77%|███████▋  | 373/485 [01:55<00:47,  2.38it/s][A
     77%|███████▋  | 374/485 [01:56<00:43,  2.53it/s][A
     77%|███████▋  | 375/485 [01:56<00:42,  2.59it/s][A
     78%|███████▊  | 376/485 [01:56<00:41,  2.64it/s][A
     78%|███████▊  | 377/485 [01:57<00:40,  2.66it/s][A
     78%|███████▊  | 378/485 [01:57<00:38,  2.79it/s][A
     78%|███████▊  | 379/485 [01:57<00:38,  2.77it/s][A
     78%|███████▊  | 380/485 [01:58<00:36,  2.85it/s][A
     79%|███████▊  | 381/485 [01:58<00:42,  2.46it/s][A
     79%|███████▉  | 382/485 [01:59<00:45,  2.25it/s][A
     79%|███████▉  | 383/485 [01:59<00:43,  2.35it/s][A
     79%|███████▉  | 384/485 [02:00<00:40,  2.50it/s][A
     79%|███████▉  | 385/485 [02:00<00:37,  2.67it/s][A
     80%|███████▉  | 386/485 [02:00<00:37,  2.62it/s][A
     80%|███████▉  | 387/485 [02:01<00:35,  2.76it/s][A
     80%|████████  | 388/485 [02:01<00:35,  2.77it/s][A
     80%|████████  | 389/485 [02:01<00:34,  2.76it/s][A
     80%|████████  | 390/485 [02:02<00:35,  2.66it/s][A
     81%|████████  | 391/485 [02:02<00:35,  2.63it/s][A
     81%|████████  | 392/485 [02:02<00:36,  2.56it/s][A
     81%|████████  | 393/485 [02:03<00:35,  2.56it/s][A
     81%|████████  | 394/485 [02:03<00:34,  2.61it/s][A
     81%|████████▏ | 395/485 [02:04<00:33,  2.72it/s][A
     82%|████████▏ | 396/485 [02:04<00:33,  2.69it/s][A
     82%|████████▏ | 397/485 [02:04<00:31,  2.76it/s][A
     82%|████████▏ | 398/485 [02:05<00:31,  2.79it/s][A
     82%|████████▏ | 399/485 [02:05<00:30,  2.86it/s][A
     82%|████████▏ | 400/485 [02:05<00:30,  2.75it/s][A
     83%|████████▎ | 401/485 [02:06<00:34,  2.42it/s][A
     83%|████████▎ | 402/485 [02:06<00:35,  2.33it/s][A
     83%|████████▎ | 403/485 [02:07<00:32,  2.50it/s][A
     83%|████████▎ | 404/485 [02:07<00:31,  2.59it/s][A
     84%|████████▎ | 405/485 [02:07<00:30,  2.66it/s][A
     84%|████████▎ | 406/485 [02:08<00:28,  2.79it/s][A
     84%|████████▍ | 407/485 [02:08<00:27,  2.82it/s][A
     84%|████████▍ | 408/485 [02:08<00:26,  2.92it/s][A
     84%|████████▍ | 409/485 [02:09<00:25,  3.04it/s][A
     85%|████████▍ | 410/485 [02:09<00:26,  2.79it/s][A
     85%|████████▍ | 411/485 [02:10<00:31,  2.37it/s][A
     85%|████████▍ | 412/485 [02:10<00:29,  2.46it/s][A
     85%|████████▌ | 413/485 [02:10<00:28,  2.50it/s][A
     85%|████████▌ | 414/485 [02:11<00:27,  2.58it/s][A
     86%|████████▌ | 415/485 [02:11<00:25,  2.70it/s][A
     86%|████████▌ | 416/485 [02:11<00:24,  2.78it/s][A
     86%|████████▌ | 417/485 [02:12<00:24,  2.78it/s][A
     86%|████████▌ | 418/485 [02:12<00:23,  2.86it/s][A
     86%|████████▋ | 419/485 [02:12<00:23,  2.82it/s][A
     87%|████████▋ | 420/485 [02:13<00:24,  2.65it/s][A
     87%|████████▋ | 421/485 [02:13<00:27,  2.37it/s][A
     87%|████████▋ | 422/485 [02:14<00:25,  2.49it/s][A
     87%|████████▋ | 423/485 [02:14<00:23,  2.63it/s][A
     87%|████████▋ | 424/485 [02:14<00:22,  2.73it/s][A
     88%|████████▊ | 425/485 [02:15<00:22,  2.68it/s][A
     88%|████████▊ | 426/485 [02:15<00:21,  2.77it/s][A
     88%|████████▊ | 427/485 [02:16<00:20,  2.82it/s][A
     88%|████████▊ | 428/485 [02:16<00:19,  2.88it/s][A
     88%|████████▊ | 429/485 [02:16<00:19,  2.85it/s][A
     89%|████████▊ | 430/485 [02:17<00:21,  2.55it/s][A
     89%|████████▉ | 431/485 [02:17<00:21,  2.47it/s][A
     89%|████████▉ | 432/485 [02:18<00:21,  2.47it/s][A
     89%|████████▉ | 433/485 [02:18<00:20,  2.51it/s][A
     89%|████████▉ | 434/485 [02:18<00:19,  2.63it/s][A
     90%|████████▉ | 435/485 [02:19<00:18,  2.71it/s][A
     90%|████████▉ | 436/485 [02:19<00:17,  2.74it/s][A
     90%|█████████ | 437/485 [02:19<00:17,  2.79it/s][A
     90%|█████████ | 438/485 [02:20<00:17,  2.70it/s][A
     91%|█████████ | 439/485 [02:20<00:18,  2.52it/s][A
     91%|█████████ | 440/485 [02:21<00:20,  2.19it/s][A
     91%|█████████ | 441/485 [02:21<00:20,  2.11it/s][A
     91%|█████████ | 442/485 [02:22<00:20,  2.09it/s][A
     91%|█████████▏| 443/485 [02:22<00:19,  2.15it/s][A
     92%|█████████▏| 444/485 [02:23<00:18,  2.18it/s][A
     92%|█████████▏| 445/485 [02:23<00:17,  2.24it/s][A
     92%|█████████▏| 446/485 [02:23<00:16,  2.43it/s][A
     92%|█████████▏| 447/485 [02:24<00:14,  2.58it/s][A
     92%|█████████▏| 448/485 [02:24<00:14,  2.52it/s][A
     93%|█████████▎| 449/485 [02:25<00:14,  2.42it/s][A
     93%|█████████▎| 450/485 [02:25<00:14,  2.48it/s][A
     93%|█████████▎| 451/485 [02:25<00:13,  2.47it/s][A
     93%|█████████▎| 452/485 [02:26<00:13,  2.50it/s][A
     93%|█████████▎| 453/485 [02:26<00:12,  2.55it/s][A
     94%|█████████▎| 454/485 [02:26<00:11,  2.76it/s][A
     94%|█████████▍| 455/485 [02:27<00:11,  2.70it/s][A
     94%|█████████▍| 456/485 [02:27<00:10,  2.72it/s][A
     94%|█████████▍| 457/485 [02:28<00:10,  2.64it/s][A
     94%|█████████▍| 458/485 [02:28<00:11,  2.35it/s][A
     95%|█████████▍| 459/485 [02:29<00:11,  2.27it/s][A
     95%|█████████▍| 460/485 [02:29<00:11,  2.23it/s][A
     95%|█████████▌| 461/485 [02:29<00:10,  2.29it/s][A
     95%|█████████▌| 462/485 [02:30<00:09,  2.32it/s][A
     95%|█████████▌| 463/485 [02:30<00:09,  2.44it/s][A
     96%|█████████▌| 464/485 [02:31<00:08,  2.53it/s][A
     96%|█████████▌| 465/485 [02:31<00:07,  2.68it/s][A
     96%|█████████▌| 466/485 [02:31<00:06,  2.78it/s][A
     96%|█████████▋| 467/485 [02:32<00:06,  2.76it/s][A
     96%|█████████▋| 468/485 [02:32<00:06,  2.59it/s][A
     97%|█████████▋| 469/485 [02:32<00:06,  2.60it/s][A
     97%|█████████▋| 470/485 [02:33<00:05,  2.60it/s][A
     97%|█████████▋| 471/485 [02:33<00:05,  2.75it/s][A
     97%|█████████▋| 472/485 [02:34<00:04,  2.72it/s][A
     98%|█████████▊| 473/485 [02:34<00:04,  2.79it/s][A
     98%|█████████▊| 474/485 [02:34<00:03,  2.90it/s][A
     98%|█████████▊| 475/485 [02:35<00:03,  2.83it/s][A
     98%|█████████▊| 476/485 [02:35<00:03,  2.81it/s][A
     98%|█████████▊| 477/485 [02:35<00:03,  2.53it/s][A
     99%|█████████▊| 478/485 [02:36<00:02,  2.42it/s][A
     99%|█████████▉| 479/485 [02:36<00:02,  2.48it/s][A
     99%|█████████▉| 480/485 [02:37<00:02,  2.48it/s][A
     99%|█████████▉| 481/485 [02:37<00:01,  2.62it/s][A
     99%|█████████▉| 482/485 [02:37<00:01,  2.63it/s][A
    100%|█████████▉| 483/485 [02:38<00:00,  2.82it/s][A
    100%|█████████▉| 484/485 [02:38<00:00,  2.88it/s][A
    100%|██████████| 485/485 [02:38<00:00,  2.95it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: ../challenge_video_result_advanced.mp4 
    
    CPU times: user 8min 35s, sys: 1.58 s, total: 8min 37s
    Wall time: 2min 39s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output))
```





<video width="960" height="540" controls>
  <source src="../challenge_video_result_advanced.mp4">
</video>




## Afterthoughts

Many paramters have to be tuned patiently for the pipelines to give satisfactory results. The advanced pipeline works better by sanity checks, look ahead filter, and smoothing. However, it requires more computation power. 

Shadows and steep curves are the hardest to deal with because region masking is no longer sufficient to eliminate noises while preserving lanes. Moreover, shadows shortens true lanes and cast false positives to the binary image.

The sanity check can be added with intersection check and to sort out the wrong lane. 


```python

```
