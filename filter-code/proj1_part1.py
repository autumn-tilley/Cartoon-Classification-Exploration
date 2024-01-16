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
def filter_test(img_path, type):
    resultsDir = '..' + os.sep + 'results'
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)

    test_image = load_image(img_path)
    test_image = rescale(test_image, 0.7, mode='reflect', multichannel=True)

    if type == "cartoon" or type == "both":

        # reference: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
        
        img = cv2.imread(img_path)
        Z = img.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 10
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        #center = np.uint8(center)
        res = center[label.flatten()]
        cartoon = res.reshape((img.shape))

        cv2.imshow('cartoon',cartoon)

        #done = save_image('../results/cartoon.jpg', res2)
        if type == "cartoon":
            done = cv2.imwrite(img_path, cartoon) 
    

    #edge or both
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

        # 0.5 added because the output image is centered around zero otherwise and mostly black
        #img = np.clip(sobelx_image+0.5, 0.0, 1.0)
        #plt.imshow(sobelx_image, cmap='gray')
        #plt.show()

        img = np.sqrt((sobely_image * sobely_image) + (sobelx_image * sobelx_image))
        img = np.clip(img, 0.0, 1.0)

        #https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

        img = cv2.imread(img_path)

        blur_filter = np.ones((5, 5), dtype=np.float32)
        # making the filter sum to 1
        blur_filter /= np.sum(blur_filter, dtype=np.float32)
        blur_image = my_imfilter(img, blur_filter)

        blur_image = np.uint8(blur_image)
        edges = cv2.Canny(blur_image, threshold1=100, threshold2=200)
        edges = np.invert(edges)

        if type == "both":
            cartoon = cv2.bitwise_and(cartoon, cartoon, mask=edges)
            done = cv2.imwrite(img_path, cartoon) 
        
        else:
            done = cv2.imwrite(img_path, edges) 

        #done = save_image(resultsDir + os.sep + 'edge.jpg', img)




    '''
    Small blur with a box filter
    This filter should remove some high frequencies.
    
    blur_filter = np.ones((3, 3), dtype=np.float32)
    # making the filter sum to 1
    blur_filter /= np.sum(blur_filter, dtype=np.float32)
    blur_image = my_imfilter(test_image, blur_filter)
    plt.imshow(blur_image,cmap='gray')
    plt.show()
    done = save_image(resultsDir + os.sep + 'blur_image.jpg', blur_image)

    Large blur
    This blur would be slow to do directly, so we instead use the fact that Gaussian blurs are separable and blur sequentially in each direction.
    
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = 10, 12
    large_1d_blur_filter = np.asarray(
        [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k, k+1)], dtype=np.float32)
    large_1d_blur_filter = large_1d_blur_filter.reshape(-1, 1)
    large_blur_image = my_imfilter(test_image, large_1d_blur_filter)
    # notice the T operator which transposes the filter
    large_blur_image = my_imfilter(large_blur_image, large_1d_blur_filter.T)
    plt.imshow(large_blur_image, cmap='gray')
    plt.show()
    done = save_image(resultsDir + os.sep +
                      'large_blur_image.jpg', large_blur_image)

    # Slow (naive) version of large blur
    # import time
    # large_blur_filter = np.dot(large_1d_blur_filter, large_1d_blur_filter.T)
    # t = time.time()
    # large_blur_image = my_imfilter(test_image, large_blur_filter);
    # t = time.time() - t
    # print('{:f} seconds'.format(t))
    ##

    Oriented filter (Sobel operator)
    
    sobel_filter = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=np.float32)  # should respond to horizontal gradients
    sobel_image = my_imfilter(test_image, sobel_filter)

    # 0.5 added because the output image is centered around zero otherwise and mostly black
    sobel_image = np.clip(sobel_image+0.5, 0.0, 1.0)
    plt.imshow(sobel_image, cmap='gray')
    plt.show()
    done = save_image(resultsDir + os.sep + 'sobel_image.jpg', sobel_image)


    High pass filter (discrete Laplacian)
    
    laplacian_filter = np.asarray(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian_image = my_imfilter(test_image, laplacian_filter)

    # added because the output image is centered around zero otherwise and mostly black
    laplacian_image = np.clip(laplacian_image+0.5, 0.0, 1.0)
    plt.figure()
    plt.imshow(laplacian_image, cmap='gray')
    plt.show()
    done = save_image(resultsDir + os.sep + 'laplacian_image.jpg', laplacian_image)

    # High pass "filter" alternative
    high_pass_image = test_image - blur_image
    high_pass_image = np.clip(high_pass_image+0.5, 0.0, 1.0)
    plt.figure()
    plt.imshow(high_pass_image, cmap='gray')
    plt.show()
    done = save_image(resultsDir + os.sep + 'high_pass_image.jpg', high_pass_image) '''
