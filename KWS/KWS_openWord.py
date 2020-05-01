import numpy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/PatRec17_KWS_Data-master/PatRec17_KWS_Data-master'
    OUTPATH = MYPATH + '/images/single_words_270/'
    GRIDPATH = MYPATH + '/ground-truth/locations/270.svg'
    pattern = 'L\s(\d+\.\d+)\s(\d+\.\d+)+'

    im = Image.open(MYPATH + "/images/binarized/otsu_270.jpg").convert("RGBA")
    imArray = numpy.asarray(im)

    with open(GRIDPATH, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        count = 1
        for line in data[2:-1]:
            coords = re.findall(pattern, str(line))
            for coord in range(len(coords)):
                coords[coord] = (float(coords[coord][0]), float(coords[coord][1]))

            maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
            ImageDraw.Draw(maskIm).polygon(coords, outline=1, fill=1)
            mask = numpy.array(maskIm)

            # assemble new image (uint8: 0-255)
            newImArray = numpy.empty(imArray.shape, dtype='uint8')

            # colors (three first columns, RGB)
            newImArray[:, :, :3] = imArray[:, :, :3]

            # transparency (4th column)
            newImArray[:, :, 3] = mask * 255
            # back to Image from numpy
            newIm = Image.fromarray(newImArray, "RGBA")

            #if len(newImArray.shape) == 3:
                #newImArray = newImArray[:,:,0]

            #cut_img = np.full(newImArray[:,:,0].shape, 0)
            #cut_img[:, :] = newImArray[:, :] * mask

            indexes = np.where(mask)

            if len(indexes[0]) == 0:
                pass
            else:
                row_max = np.max(indexes[0])
                row_min = np.min(indexes[0])

                col_max = np.max(indexes[1])
                col_min = np.min(indexes[1])

                cut_img = newIm.crop((col_min, row_min, col_max, row_max))
                # cut_img.show()

                cut_img = cut_img.convert("L")
            newIm.save(OUTPATH + f'270_word{count}.png', 'png')  # save word image to current directory
            count += 1


main()