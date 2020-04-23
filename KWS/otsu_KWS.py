import matplotlib.pyplot as plt
from skimage import filters
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw
import numpy as np


MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/PatRec17_KWS_Data-master/PatRec17_KWS_Data-master/images/'
OUTPATH = MYPATH + '/binarized/'
files = [f for f in listdir(MYPATH)if isfile(join(MYPATH, f))]  # extracts all filenames from folder
for file in files:
    img = Image.open(MYPATH + file)
    val = filters.threshold_otsu(np.asarray(img))
    newIm = Image.fromarray(img > val)
    newIm.save(OUTPATH + 'otsu_' + file)

