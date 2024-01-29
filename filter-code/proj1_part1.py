# Homework 1 Image Filtering - Tests on my_imfilter function
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import os
from skimage.transform import rescale
import numpy as np
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt
from helpers import load_image, save_image
from student import my_imfilter
import cv2

"""
This function loads an image, and then attempts to filter that image
using different kernels as a testing routine.
"""
def filter_test(img_path, type, quantity):

    test_image = load_image(img_path)
    test_image = rescale(test_image, 0.7, mode='reflect', multichannel=True)

    if type == "cartoon" or type == "both":

        # reference: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
        
        img = cv2.imread(img_path)
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)

        blur_filter = np.ones((5, 5), dtype=np.float32)
        blur_filter /= np.sum(blur_filter, dtype=np.float32)
        img = my_imfilter(img, blur_filter)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        cartoon = res.reshape((img.shape))

        cv2.imshow('cartoon',cartoon)

        if type == "cartoon":
            done = cv2.imwrite(img_path, cartoon) 
    
    if type == "edge" or type == "both":

        sobely_filter = np.asarray([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]],
                                dtype=np.float32)  # should respond to horizontal gradients
        sobely_image = my_imfilter(test_image, sobely_filter)


        sobelx_filter = np.asarray([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]],
                                dtype=np.float32)  # should respond to horizontal gradients
        sobelx_image = my_imfilter(sobely_image, sobelx_filter)

        img = np.sqrt((sobely_image * sobely_image) + (sobelx_image * sobelx_image))
        img = np.clip(img, 0.0, 1.0)

        #reference: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

        img = cv2.imread(img_path)

        blur_filter = np.ones((3, 3), dtype=np.float32)
        blur_filter /= np.sum(blur_filter, dtype=np.float32)
        blur_image = my_imfilter(img, blur_filter)

        blur_image = np.uint8(blur_image)
        edges = cv2.Canny(blur_image, threshold1=100, threshold2=200)

        edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.float32))

        edges = np.invert(edges)

        if type == "both":
            cartoon = cv2.bitwise_and(cartoon, cartoon, mask=edges)
            done = cv2.imwrite(img_path, cartoon) 
        
        else:
            done = cv2.imwrite(img_path, edges)