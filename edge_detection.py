import cv2
import numpy as np
def binary_array(array,thresh,value=0):
    if value==0:
        binary=np.ones_like(array)
        binary[array<thresh]=0
    else:
        binary=np.zeros_like(array)
        binary[array>=thresh]=1
    return binary
def blur_gaussian(channel,ksize=3):
    return cv2.GaussianBlur(channel,(ksize,ksize),0)
def mag_thresh(image,sobel_kernel=3,thresh=(0,255)):
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx=np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely=np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    mag=np.sqrt(sobelx**2+sobely**2)
    return binary_array(mag,thresh)

def sobel(img_channel,orient='x',sobel_kernel=3):
    if orient=='x':
        sobel_output = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    elif orient=='y':
        sobel_output = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return np.abs(sobel_output)
def threshold(channel,thresh=(128,255),thresh_type=cv2.THRESH_BINARY):
   return cv2.threshold(channel,thresh[0],thresh[1],thresh_type)


