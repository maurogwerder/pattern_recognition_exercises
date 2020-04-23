import matplotlib.pyplot as plt
from skimage import filters
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/PatRec17_KWS_Data-master/PatRec17_KWS_Data-master/images/'
OUTPATH = MYPATH + '/binarized/'
files = filenames = [f for f in listdir(MYPATH)if isfile(join(MYPATH, f))]  # extracts all filenames from folder
for file in files:
    img = plt.imread(MYPATH + file)
    val = filters.threshold_otsu(img)

    plt.figure(figsize=(24, 40))
    plt.imshow(img > val, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPATH + 'otsu_' + file)

