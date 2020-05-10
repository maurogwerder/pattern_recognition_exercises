import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from scipy.misc import imread
from os.path import isfile, join
from os import listdir, walk
import re
import os
import time
# =============================================================================
# extract the words
# =============================================================================


import numpy
from PIL import Image, ImageDraw

import re


def get_reference_words(word):
    reference_word = f"^{word}"

    pattern = re.compile(reference_word.lower())

    reference_pictures = [item for item in training_list if re.findall(pattern, item[1].lower()) != []]
    return reference_pictures


# =============================================================================
# some feature functions
# =============================================================================
# black = 0
# white = 1

# test vector
# vector = np.array([1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1])
# print(vector)

# -----Black white transitions-------------------

def bw_transitions(vector):
    transitions = 0

    # print(len(vector))

    for i in range(len(vector) - 1):
        # print(vector[i])
        if vector[i] != vector[i + 1]:
            transitions = transitions + 1

    bw_transitions = transitions / 2

    return bw_transitions


def single_lc(col):
    for row in reversed(range(0, len(col))):
        if col[row] == 0:
            fraction = row / len(col)
            # print(fraction)
            return 1 - fraction, row
        elif row == 0:
            return 0, 0


def single_uc(col):
    for row in range(0, len(col)):
        if col[row] == 0:
            fraction = row / len(col)

            return 1 - fraction, row
            break
        elif row == (len(col) - 1):
            return 0, 0


def inbound_ratio(col, uc, lc):
    col = col[uc:lc]
    if len(col) == 0:
        return 0
    ratio = sum(col == 0) / len(col)
    return ratio


# bla = bw_transitions(vector)
# print(bla)

# print(bw_transitions)

# ----fraction of black pixels---------------------

def fract_black(vector):
    black_ind = np.where(vector == 0)
    black = len(black_ind[0])
    fr_black = black / len(vector)

    return fr_black


# print(len(vector))
# print(fract_black(vector))


# ----upper gradient---------------------
def feature_gradient_upper(vector, vector_2):
    vect_1 = vector
    vect_2 = vector_2

    vect_1_upper = vect_1[:int((len(vect_1) + 1) / 2)]
    vect_2_upper = vect_2[:int((len(vect_2) + 1) / 2)]

    feature_count_1 = len(vect_1_upper[vect_1_upper > 128])
    feature_count_2 = len(vect_2_upper[vect_2_upper > 128])

    feature = feature_count_2 - feature_count_1

    return feature


# ----lower gradient---------------------

def feature_gradient_lower(vector, vector_2):
    vect_1 = vector
    vect_2 = vector_2

    vect_1_upper = vect_1[int((len(vect_1) + 1) / 2):]
    vect_2_upper = vect_2[int((len(vect_2) + 1) / 2):]

    feature_count_1 = len(vect_1_upper[vect_1_upper > 128])
    feature_count_2 = len(vect_2_upper[vect_2_upper > 128])

    feature = feature_count_2 - feature_count_1

    return feature


def binarize(img):
    row = int(img.shape[0])
    col = int(img.shape[1])

    for i in range(row):
        for j in range(col):

            if img[i][j] < 0.5:
                img[i][j] = 0

            else:
                img[i][j] = 1

    return img


def window_trimmer(window):
    i = 0

    while i < len(window):
        if window[i] != 0:
            window = window[i:]
            i = len(window)
        i += 1

    j = len(window) - 1

    while j > 0:
        if window[j] != 0:
            window = window[0: j + 1]
            j = 0
        j = j - 1

    return (window)


def feature_extract(word):
    cols = int(word.shape[1])
    # word = binarize(word)
    trans = list()  # vector with number of black white transitions per window
    black_frac = list()  # vector with fraction of black pixels per window
    lc_frac_l = list()  # vector with the relative position of the lower contour per window
    uc_frac_l = list()  # vector with the relative position of the upper contour per window
    bound_frac = list()  # vector with fraction of black pixels within the contours per window
    grad_upper = list()  # vector with upper gradient of black pixel
    grad_lower = list()  # vector with lower gradient of black pixel
    for i in range(cols):
        window = word[:, i]
        vector = window_trimmer(window)

        bw_tran = bw_transitions(vector)
        trans.append(bw_tran)

        black = fract_black(vector)
        black_frac.append(black)

        lc_frac, lc = single_lc(vector)
        lc_frac_l.append(lc_frac)

        uc_frac, uc = single_uc(vector)
        uc_frac_l.append(uc_frac)

        inbound = inbound_ratio(vector, uc, lc)
        bound_frac.append(inbound)

        if i < cols - 1:
            window_2 = word[:, i + 1]
            vector_2 = window_trimmer(window_2)

            grad_u = feature_gradient_upper(vector, vector_2)
            grad_upper.append(grad_u)

            grad_l = feature_gradient_lower(vector, vector_2)
            grad_lower.append(grad_l)

    return [trans, black_frac, lc_frac_l, uc_frac_l, bound_frac, grad_upper, grad_lower]


if __name__ == '__main__':
    MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/dtw_sorted/dtw_sorted.py'
    ########CONVERT SVG TO TEXT FILE AND INSERT HERE
    # MYPATH = 'C:/Users/mg_gw/OneDrive/Dokumente/UNIBE 8. Semester/3_Pattern_Recognition/32_exercises/03_KWS/dtw_sorted'
    my_path = os.getcwd() + "/PatRec17_KWS_Data-master/ground-truth/transcription.txt"

    file = open(my_path)  # open text file

    lines = file.readlines()  # each line becomes element in list lines

    file.close

    n = 35  # number of pages
    transcription_list = []

    page_number = 270
    # page_number = '270'
    a = 0

    for i in range(len(lines)):  # iterate over all lines

        pattern = r'^(\d{3})-\d{2}-\d{2}\s(.+)\n'
        line = lines[i]
        # print(line)

        pattern1 = '^(\d{3})'
        page_numb = re.findall(pattern1, line)  # find pattern
        page_numb = str(page_numb)
        page_numb = page_numb.strip(' \'[]')
        # print('g',page_numb)
        page_numb = int(page_numb)

        index = page_numb - 270

        # print(index)

        index = page_numb - 270

        # print(index)

        out1 = re.search(pattern, line).group(1)  # find pattern
        out1 = str(out1)

        out1 = out1.strip(' \'[]')

        out1 = int(out1)

        out2 = re.search(pattern, line).group(2)  # find pattern

        out2 = str(out2)

        out2 = out2.strip(' \'[]')

        out = tuple((out1, out2))

        # out = float(out)

        transcription_list.append(out)

    # all words = 3726
    # print(len((transcription_list)))
    # print((transcription_list))
    # print(transcription_list[1])

    # print(transcription_list[4][1])

    page_270 = []
    for i in range(len(transcription_list)):
        if transcription_list[i][0] == 270:
            # print(transcription_list[i][1])
            page_270.append((270, transcription_list[i][1]))
    #print(transcription_list)
    # print(len(page_270))
    # print(page_270)

    train_set_path = os.getcwd() + r"/img_train"

    training_list = []  # (Page, word, picture)
    index = 0
    for i in os.listdir(train_set_path):
        for j in os.listdir(train_set_path + "/" + i):
            training_list.append((transcription_list[index][0], transcription_list[index][1],
                                  np.asarray(Image.open(train_set_path + "/" + i + "/" + j))))
            index += 1
    validate_set_path = os.getcwd() + r"/img_validate"

    validate_list = []  # (Page, word, picture)
    index_ovset = index
    index = 0
    for i in os.listdir(validate_set_path):
        for j in os.listdir(validate_set_path + "/" + i):
            validate_list.append((transcription_list[index_ovset][0], transcription_list[index_ovset][1],
                                  np.asarray(Image.open(validate_set_path + "/" + i + "/" + j))))
            index += 1
            index_ovset += 1


    # Calculate and write down all feature-vectors in a file for each word in the validate set
    for index, val_word in enumerate(validate_list):
        features = feature_extract(val_word[2])
        with open(f'{val_word[0]}_{index}_{val_word[1]}.txt', 'w') as feature_file:
            for feat in features:
                feature_file.write(','.join(map(str, feat)) + '\n')

    # Do the same for the training set:
    for index, train_word in enumerate(training_list):
        features = feature_extract(train_word[2])
        with open(f'{train_word[0]}_{index}_{train_word[1]}.txt', 'w') as feature_file:
            for feat in features:
                feature_file.write(','.join(map(str, feat)) + '\n')
