# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:17:53 2020

@author: Admin
"""

# =============================================================================
# svg to png
# =============================================================================
# have to install svglib

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

drawing = svg2rlg("270.svg")
renderPDF.drawToFile(drawing, "file.pdf")
renderPM.drawToFile(drawing, "file.png", fmt="PNG")


# =============================================================================
# Cut ou polygon
# =============================================================================
#https://gist.github.com/yosemitebandit/03bb3ae302582d9dc6be

"""Crop a polygonal selection from an image."""
import numpy as np
from PIL import Image
# have to install shapely
from shapely.geometry import Point
from shapely.geometry import Polygon


im = Image.open('270.jpg')
pixels = np.array(im)
im_copy = np.array(im)
print(pixels.shape)

coords =[(112.00, 170.00), (112.00, 230.00), (129.27, 231.50), (132.00, 230.00), (232.00, 230.00), (240.00, 238.00), (299.69, 148.25), ( 192.00, 157.00)]


# mal ohne komma stellen
region = Polygon(coords)
print(region)


counter = 0
for index, pixel in np.ndenumerate(pixels):
   # Unpack the index.
    row, col= index
    # # We only need to look at spatial pixel data for one of the four channels.
    # if channel != 0:
    #   continue
    point = Point(row, col)
    if not region.contains(point):
        im_copy[(row, col)] = 255
    # im_copy[(row, col, 1)] = 255  # donz need this because its grayscale
    # im_copy[(row, col, 2)] = 255
    # im_copy[(row, col, 3)] = 0
    # counter += 1
    # print(counter)

cut_image = Image.fromarray(im_copy)
cut_image.save('new.png')



# =============================================================================
# Extract path ?  - have not tried it yet, copied it from the internet
# =============================================================================


# Dont know if the following works - havent tried it yet,
# but it should somehow extract the paths / cordinated from the svg

# https://stackoverflow.com/questions/15857818/python-svg-parser


# The question was about extracting the path strings,
#  but in the end the line drawing commands were wanted. 
#  Based on the answer with minidom, I added the path parsing with svg.path 
#  to generate the line drawing coordinates:


# requires svg.path, install it like this: pip3 install svg.path

# converts a list of path elements of a SVG file to simple line drawing commands
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom

# read the SVG file
doc = minidom.parse('test.svg')
path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
doc.unlink()

# print the line draw commands
for path_string in path_strings:
    path = parse_path(path_string)
    for e in path:
        if isinstance(e, Line):
            x0 = e.start.real
            y0 = e.start.imag
            x1 = e.end.real
            y1 = e.end.imag
            print("(%.2f, %.2f) - (%.2f, %.2f)" % (x0, y0, x1, y1))


