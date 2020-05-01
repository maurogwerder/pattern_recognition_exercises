import numpy as np


def get_uc(img):
    # convert the floats to integers or else = is not really 0 but 0.0000....something
    img.astype(int)

    # if there are different heights would have to extract the height
    # in the first for loop

    height, length = img.shape  # height = number of rows, length = number of cols

    uc_list = []

    counter = 0

    for col in range(0, length):

        height = len(img[:, col])
        for row in range(0, height):

            if img[row, col] == 0:
                fraction = row / height

                uc_list.append(1 - fraction)
                break
            elif row == (height - 1):

                # print('None')

                # appends None to the list if there is no black pixels
                # could change None to anything you want to have
                # if there is no black pixel in that column
                uc_list.append(None)

    ### change output format here if needed ###

    return uc_list


def get_lc(img):
    # convert the floats to integers or else = is not really 0 but 0.0000....something
    img.astype(int)

    # if there are different heights would have to extract the height
    # in the first for loop

    height, length = img.shape  # height = number of rows, length = number of cols

    lc_list = []

    counter = 0

    for col in range(0, length):

        height = len(img[:, col])
        for row in reversed(range(0, height)):

            if img[row, col] == 0:
                fraction = row / height
                # print(fraction)
                lc_list.append(1 - fraction)
                break
            elif row == (0):

                # appends None to the list if there is no black pixels
                # could change None to anything you want to have
                # if there is no black pixel in that column
                lc_list.append(None)

    return lc_list


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
    return(window)


def single_lc(col):
    for row in reversed(range(0, len(col))):
        if col[row] == 0:
            fraction = row / len(col)
            # print(fraction)
            return 1 - fraction, row
        elif row == 0:
            return None, None


def single_uc(col):
    for row in range(0, len(col)):
        if col[row] == 0:
            fraction = row / len(col)

            return 1 - fraction, row
            break
        elif row == (len(col) - 1):
            return None, None


def inbound_ratio(col, uc, lc):
    uc = uc * len(col)  # as input is a fraction, we again get it to a real pixel value
    lc = lc * len(col)
    col = col[uc:lc]
    ratio = sum(col == 0) / len(col)
    return ratio
"""
def rolling_window(arr):
    cols = len(arr[0])
    rows = len(arr)
    lc_list = []
    uc_list = []
    inbound_rat_list = []
    for c in range(cols):
        window = window_trimmer(arr[:, c])
"""

