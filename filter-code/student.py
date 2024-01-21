# Homework 1 Image Filtering Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage.transform import rescale

def my_imfilter(image, kernel):
    """
    Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    k, l = kernel.shape
    kernel = np.rot90(kernel, 2)

    if (k % 2) == 0 or (l % 2) == 0:
        raise Exception("does not work with an even-dimension filter")

    leftSideKernel = kernel[:,:l//2]
    rightSideKernel = kernel[:,(l//2 + 1):]
    topSideKernel = kernel[:k//2,:]
    bottomSideKernel = kernel[(k//2 + 1):,:]

    # when the kernel is an identity filter, there is no need to pad, just return original input image
    if (kernel == np.array([1])).all() or (kernel[k//2, l//2] == 1 and (leftSideKernel == np.zeros(leftSideKernel.shape)).all() and \
        (rightSideKernel == np.zeros(rightSideKernel.shape)).all() and \
            (topSideKernel == np.zeros(topSideKernel.shape)).all() and \
                (bottomSideKernel == np.zeros(bottomSideKernel.shape)).all()):
            filtered_image = image
    else:
        # when working with gray images without the "channel dimension"
        if image.ndim == 2:
            image = np.pad(image, ((k//2, k//2), (l//2, l//2)), 'symmetric')
            m, n = image.shape
            for i in range(m-k):
                for j in range(n-l):
                    neighborhood = image[i:i+k, j:j+k]
                    new_Value = np.sum(neighborhood * kernel)
                    filtered_image[i,j] = new_Value
        else: #when working with RGB images
            image = np.pad(image, ((k//2, k//2), (l//2, l//2), (0, 0)), 'symmetric')
            m, n, channel = image.shape
            for i in range(m-k):
                for j in range(n-l):
                    for c in range(channel):
                        neighborhood = image[i:i+k, j:j+l, c]
                        new_Value = np.sum(neighborhood * kernel)
                        filtered_image[i,j,c] = new_Value
    ##################

    return filtered_image