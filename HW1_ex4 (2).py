# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 08:26:28 2020

@author: Admin
"""

"""

Introduction to Signal and Image Processing
Homework 1

Created on Tue Mar 3, 2020
@author:
    
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def test_interp():
    # Tests the interp() function with a known input and output
    # Leads to error if test fails

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
    x_new = np.array((0.5, 2.3, 3, 5.45))
    y_new_solution = np.array([0.2, 0.46, 0.6, 0.69])
    y_new_result = interp(y, x, x_new)
    np.testing.assert_almost_equal(y_new_solution, y_new_result)

print(test_interp())

def test_interp_1D():
    # Test the interp_1D() function with a known input and output
    # Leads to error if test fails

    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
    y_rescaled_solution = np.array([
        0.20000000000000001, 0.29333333333333333, 0.38666666666666671,
        0.47999999999999998, 0.57333333333333336, 0.53333333333333333,
        0.44000000000000006, 0.45333333333333331, 0.54666666666666663,
        0.64000000000000001, 0.73333333333333339, 0.82666666666666677,
        0.91999999999999993, 1.0066666666666666, 1.0533333333333335,
        1.1000000000000001
    ])
    y_rescaled_result = interp_1D(y, 2)
    np.testing.assert_almost_equal(y_rescaled_solution, y_rescaled_result)


print(test_interp_1D())


def test_interp_2D():
    # Tests interp_2D() function with a known and unknown output
    # Leads to error if test fails

    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_scaled = np.array([[1., 1.4, 1.8, 2.2, 2.6, 3.],
                              [2., 2.4, 2.8, 3.2, 3.6, 4.],
                              [3., 3.4, 3.8, 4.2, 4.6, 5.],
                              [4., 4.4, 4.8, 5.2, 5.6, 6.]])

    result = interp_2D(matrix, 2)
    np.testing.assert_almost_equal(matrix_scaled, result)


#def interp(y_vals, x_vals, x_new):
#    # Computes interpolation at the given abscissas
#    #
#
#    
#    y_new = np.array([])
#
#    for x in range(0,len(x_new)):
#        for x_old in range(0,len(x_vals)):
#            if (x_vals[(x_old)-1] <= x_new[x]) and (x_vals[x_old] >= x_new[x]):
#                y_new_part1 = (y_vals[(x_old)-1])*(1 - ((x_new[x]-x_vals[(x_old)-1])/(x_vals[x_old]-x_vals[(x_old)-1])))
#                y_new_part2 = (y_vals[x_old])*((x_new[x]-x_vals[(x_old)-1])/(x_vals[x_old]-x_vals[(x_old)-1]))
#                y_new_whole = y_new_part1 + y_new_part2
#                y_new = np.append(y_new,y_new_whole)
# 
#                print(' new y val of x', x_new[x],' is  ', y_new_whole)
#                print(x_new[x], x_vals[x_old], x_vals[x_old -1])
#        continue
##            elif x_vals[x_old] == x_new[x]:
##                y_new = np.append(y_new, y_vals[x_old])
#    return y_new
    
def interp(y_vals, x_vals, x_new):
    # Computes interpolation at the given abscissas
    #

    
    y_new = np.array([])


    for x in range(0,len(x_new)):
#        counter_old = counter
#        counter +=1
        for x_old in range(0,len(x_vals)):
        

            if (x_vals[(x_old)-1] <= x_new[x] <= x_vals[(x_old)]):
                y_new_part1 = (y_vals[(x_old)-1])*(1 - ((x_new[x]-x_vals[(x_old)-1])/(x_vals[x_old]-x_vals[(x_old)-1])))
                y_new_part2 = (y_vals[x_old])*((x_new[x]-x_vals[(x_old)-1])/(x_vals[x_old]-x_vals[(x_old)-1]))
                y_new_whole = y_new_part1 + y_new_part2
                y_new = np.append(y_new,y_new_whole)
 
#                print(' new y val of x', x_new[x],' is  ', y_new_whole)
#                print(x_new[x], x_vals[x_old], x_vals[x_old -1])

                
                break
                
            if x_new[x] < x_vals[0]:
#                print(x_new[x], 'is smaller')
                y_new_part1 = (y_vals[0])*(1 - ((x_new[x]-0)/(x_vals[0]-0)))
                y_new_part2 = (y_vals[0])*((x_new[x]-0)/(x_vals[0]-0))
                y_new_whole = y_new_part1 + y_new_part2
                y_new = np.append(y_new,y_new_whole)

                
                break
    return y_new

#x_vals = np.array([1,2,3,4,5,6,10])
#y_vals = np.array([2,3,4,4,6,10,14])
#x_new = np.array([2.5, 5, 5, 7.5, 6.6, 2.2])

#x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#print(x)
#y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
#x_new = np.array((0.5, 2.3, 3, 5.45))
#
#print(interp(y, x, x_new))

import math

def interp_1D(signal, scale_factor):
    # Linearly interpolates one dimensional signal by a given saling fcator
    #
    # Inputs:
    #   signal: A one dimensional signal to be samples from, numpy array
    #   scale_factor: scaling factor, float
#    y = [math.sin(i) for i in range(20)]
#    y_signal = np.asarray([i*2 for i in range(11)])
#    print(len(y_signal), len(x))
#
#    x = np.asarray([i for i in range(len(y_signal))])
#    print((x))
##
#    plt.plot(x,y_signal)
#    

    
#    x_scale = np.linspace(x[0], len(x)*scale_factor,num=len(x)*scale_factor,axis =0)

    x_scale = np.asarray([i*scale_factor for i in range(len(signal))])
    y_scale = signal
    
#    x_new = np.asarray([i for i in range(len(y_scale)*scale_factor)])
    x_new = np.linspace(x_scale[0],x_scale[len(x)-1],num=(len(x_scale)*scale_factor),axis =0)
#    print(x_new)
    y_signal_interp = interp(y_scale,x_scale, x_new)
#    print((y_signal_interp))
#    print(len(x_new))
#    plt.plot(x_new,y_signal_interp)
    
    #
    # Outputs:
    #   signal_interp: Interpolated 1D signal, numpy array

    ################### PLEASE FILL IN THIS PART ###############################

    return y_signal_interp

#y_signal = np.asarray([i*2 for i in range(11)])
signal = np.array([0.3, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
scale_factor = 2

print(interp_1D(signal, scale_factor))


# =============================================================================
# Bilinear interpolation
# =============================================================================

def interp_2D(img, scale_factor):
    # Applies bilinear interpolation using 1D linear interpolation
    # It first interpolates in one dimension and passes to the next dimension
    #
    # Inputs:
    #   img: 2D signal/image (grayscale or RGB), numpy array
    #   scale_factor: Scaling factor, float
    #
    # Outputs:
    #   img_interp: interpolated image with the expected output shape, numpy array

    ################### PLEASE FILL IN THIS PART ###############################

    return img_interp


# set arguments
filename = 'bird.jpg'
# filename = 'butterfly.jpg'
scale_factor = 1.5  # Scaling factor

# Before trying to directly test the bilinear interpolation on an image, we
# test the intermediate functions to see if the functions that are coded run
# correctly and give the expected results.

print('...................................................')
print('Testing test_interp()...')
test_interp()
print('done.')

print('Testing interp_1D()....')
test_interp_1D()
print('done.')

print('Testing interp_2D()....')
test_interp_2D()
print('done.')

print('Testing bilinear interpolation of an image...')
# Read image as a matrix, get image shapes before and after interpolation
img = (plt.imread(filename)).astype('float')  # need to convert to float
in_shape = img.shape  # Input image shape

# Apply bilinear interpolation
img_int = interp_2D(img, scale_factor)
print('done.')

# Now, we save the interpolated image and show the results
print('Plotting and saving results...')
plt.figure()
plt.imshow(img_int.astype('uint8'))  # Get back to uint8 data type
filename, _ = os.path.splitext(filename)
plt.savefig('{}_rescaled.jpg'.format(filename))
plt.close()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img.astype('uint8'))
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(img_int.astype('uint8'))
plt.title('Rescaled by {:2f}'.format(scale_factor))
print('Do not forget to close the plot window --- it happens:) ')
plt.show()

print('done.')
