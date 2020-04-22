# Keyword Spotting using Dynamic Time Warping (DTW)

import csv
import numpy as np
import matplotlib.pyplot as plt
from svglib.svglib import load_svg_file, svg2rlg
# Consider Sakoe-chiba-band, define it as percentage of the smaller of the two input strings.
# https://github.com/lunactic/PatRec17_KWS_Data
# Approach for loading files:
# - load first image and first svg
# - extract all words from first image to empty canvas of size maxwidth(grid) x maxheight(grid)
# - append list with arrays (have different dimensions; can't be array of arrays)
# - Repeat for valid-dataset

# Q: can we use the polygon svg-files for cropping?
# https://gis.stackexchange.com/questions/233279/clipping-a-raster-using-an-irregular-polygon-with-python

def load_train(mypath):
    train_path = mypath + '/task/train.txt'
    valid_path = mypath + '/task/valid.txt'
    return mypath


if __name__ == '__main__':
    MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/PatRec17_KWS_Data-master/PatRec17_KWS_Data-master'
    grid = svg2rlg(MYPATH + '/ground-truth/locations/270.svg')
    print(np.asarray(grid))