# Homework 1 Image Filtering Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the homework webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
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

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the homework webpage. Apply a filter (using kernel) to an image. Return the filtered image.
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
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here
    low_frequencies = my_imfilter(image1, kernel)

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = image2 - my_imfilter(image2, kernel)

    # (3) Combine the high frequencies and low frequencies, and make sure the hybrid image values are within the range 0.0 to 1.0
    # Your code here
    hybrid_image = np.clip((high_frequencies + low_frequencies), 0.0, 1.0)

    return low_frequencies, high_frequencies, hybrid_image