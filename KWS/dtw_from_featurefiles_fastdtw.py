import csv
import numpy as np
from os.path import isfile, join
from os import listdir, walk
import re
import os
import itertools
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw(s, t):
    """n, m = len(s), len(t)
    dtw_matrix = np.zeros((n + 1, m + 1))
    s = np.asarray(s)
    t = np.asarray(t)

    dtw_matrix = np.full((n+1,m+1), np.inf)
    dtw_matrix[0, 0] = 0

    ratio = n/m

    for (i, j) in itertools.product(range(1, n + 1), range(1, m + 1)):
        if (i - j * ratio) <= 5:
            try:
                cost = np.sqrt((s[i - 1] - t[j - 1])**2+(i-j)**2)
                # take last min from a square box
                last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
                dtw_matrix[i, j] = cost + last_min

            except TypeError:

                if s[i - 1] is None:
                    s[i - 1] = 0
                if t[j - 1] is None:
                    t[j - 1] = 0
                cost = np.sqrt((s[i - 1] - t[j - 1])**2+(i-j)**2)
                # take last min from a square box
                last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
                dtw_matrix[i, j] = cost + last_min
    """
    
    distance, path = fastdtw(s, t ,dist=euclidean)
    
    return distance


def feature_all_words(ref_words, valid_set):

    c = 0
    #

    all_trans = []
    all_blac_frac = []
    all_lc_frac_l = []
    all_uc_frac_l = []
    all_bound_frac = []
    all_feature_grad_upper = []
    all_feature_grad_lower = []

    for i in ref_words:

        trans = []
        blac_frac = []
        lc_frac_l = []
        uc_frac_l = []
        bound_frac = []
        feature_grad_upper = []
        feature_grad_lower = []

        # Image.fromarray(i[2]).show()
        compare_features = np.asarray(i[3])

        print(len(ref_words))

        for j in valid_set:

            features = j[3]
            print(j[2])
            score_list = list()

            for e in range(len(features)):
                dtw_matrix = dtw(features[e], compare_features[e])
                score_list.append(dtw_matrix)

            trans.append((j[1], j[2], score_list[0]))
            blac_frac.append((j[1], j[2], score_list[1]))
            lc_frac_l.append((j[1], j[2], score_list[2]))
            uc_frac_l.append((j[1], j[2], score_list[3]))
            bound_frac.append((j[1], j[2], score_list[4]))
            feature_grad_upper.append((j[1], j[2], score_list[5]))
            feature_grad_lower.append((j[1], j[2], score_list[6]))

            c = c + 1
            print(c, " of ", len(valid_set)*len(ref_words))
            #print(sorted(trans, key=lambda tup: tup[2]))
        all_trans.append(trans)
        all_blac_frac.append(blac_frac)
        all_lc_frac_l.append(lc_frac_l)
        all_uc_frac_l.append(uc_frac_l)
        all_bound_frac.append(bound_frac)
        all_feature_grad_upper.append(feature_grad_upper)
        all_feature_grad_lower.append(feature_grad_lower)

        # print(trans)
        #trans.sort(key = operator.itemgetter(0))
        

    return all_trans, all_blac_frac, all_lc_frac_l, all_uc_frac_l, all_bound_frac, all_feature_grad_upper, all_feature_grad_lower


def get_reference_words(word):
    reference_word = f"^{word}"

    pattern = re.compile(reference_word.lower())

    reference_pictures = [item for item in training_list if re.findall(pattern, item[2].lower()) != []]
    return reference_pictures


def majority_vote(votes):
    dictio = {}

    for i in votes:
        if (i[1] + '_' + str(i[0])) in dictio.keys():
            dictio[i[1] + '_' + str(i[0])] += 1
        else:
            dictio[i[1] + '_' + str(i[0])] = 1
    print(dictio)
    r = list()
    for i in dictio.keys():
        r.append((dictio[i], i))
    r.sort(reverse=True)
    print(r)
    print(r[0])
    print("------------------------------")
    return r[0]

if __name__ == "__main__":
    
    keyword_list = list()
    
    with open("keywords.txt","r") as keys:
        for i in keys:
            keyword_list.append(i.strip())
    
    
    
    for Key_word in keyword_list:
     
    
        train_set_path = os.getcwd() + r"/feat_train"
    
        training_list = []  # (Page, word, picture)
        pattern = '(\d{3})_(\d+)_(.+).txt'
        for j in os.listdir(train_set_path):
            page = re.search(pattern, j).group(1)
            word_name = re.search(pattern, j).group(3)
            ind = re.search(pattern, j).group(2)
            with open(train_set_path + '/' + j, 'r') as feature_file:
                feature_reader = csv.reader(feature_file)
                feature_list = list(feature_reader)
                for feat in range(len(feature_list)):
                    for i in range(len(feature_list[feat])):
                        if feature_list[feat][i] == 'None':
                            feature_list[feat][i] = None
                        else:
                            feature_list[feat][i] = float(feature_list[feat][i])
    
            training_list.append((int(page), int(ind), word_name,
                                  feature_list))
    
    
        validate_set_path = os.getcwd() + r"/feat_validate"
    
        validate_list = []  # (Page, word, picture)
    
        for j in os.listdir(validate_set_path):
            page = re.search(pattern, j).group(1)
            ind = re.search(pattern, j).group(2)
            word_name = re.search(pattern, j).group(3)
            with open(validate_set_path + '/' + j, 'r') as feature_file:
                feature_reader = csv.reader(feature_file)
                feature_list = list(feature_reader)
                for feat in range(len(feature_list)):
                    for i in range(len(feature_list[feat])):
                        if feature_list[feat][i] == 'None':
                            feature_list[feat][i] = None
                        else:
                            feature_list[feat][i] = float(feature_list[feat][i])
            validate_list.append((int(page), int(ind), word_name,
                                  feature_list))
    
        ref_words = get_reference_words(Key_word)
        ref_words = ref_words[0:5]
        # print(ref_words)
        print("ref_word: ",ref_words[0][2])     
        # 'for' is the refernce image in this case
        print("START___________________________________________")
        dtw_feat_values = feature_all_words(ref_words, validate_list)
        print("STOP___________________________________________")
    
        #print(dtw_feat_values)
    
        trans = dtw_feat_values[0]
        blac_frac = dtw_feat_values[1]
        lc_frac_l = dtw_feat_values[2]
        uc_frac_l = dtw_feat_values[3]
        bound_frac = dtw_feat_values[4]
        grad_upper = dtw_feat_values[5]
        grad_lower = dtw_feat_values[6]
    
        for i in range(0, len(trans)):
            trans[i] = sorted(trans[i], key=lambda tup: tup[2])
            blac_frac[i] = sorted(blac_frac[i], key=lambda tup: tup[2])
            lc_frac_l[i] = sorted(lc_frac_l[i], key=lambda tup: tup[2])
            uc_frac_l[i] = sorted(uc_frac_l[i], key=lambda tup: tup[2])
            bound_frac[i] = sorted(bound_frac[i], key=lambda tup: tup[2])
            grad_upper[i] = sorted(grad_upper[i], key=lambda tup: tup[2])
            grad_lower[i] = sorted(grad_lower[i], key=lambda tup: tup[2])
    
        Result = list()
        
        
    
        
        m = len(trans[0])
        for i in range(m):
            vote = list()
    
            le = len(ref_words)
            for j in range(le):
                print("trans:     ",trans[j])
                vote.append(trans[j][i])
                vote.append(blac_frac[j][i])
                vote.append(lc_frac_l[j][i])
                vote.append(uc_frac_l[j][i])
                vote.append(bound_frac[j][i])
                vote.append(grad_upper[j][i])
                vote.append(grad_lower[j][i])
            print("vote:     ",vote)
    
            res = majority_vote(vote)
            Result.append(res)
         
      
            
        with open("res/"+Key_word+"_Result.txt", "w") as result:
            result.write(str(Result) + "\n")
        with open("res/a/"+Key_word+"_trans.txt", "w") as result:
            result.write(str(trans) + "\n")
        with open("res/a/"+Key_word+"_blac_frac.txt", "w") as result:
            result.write(str(blac_frac) + "\n")
        with open("res/a/"+Key_word+"_lc_frac_l.txt", "w") as result:
            result.write(str(lc_frac_l) + "\n")
        with open("res/a/"+Key_word+"_uc_frac_l.txt", "w") as result:
            result.write(str(uc_frac_l) + "\n")
        with open("res/a/"+Key_word+"_bound_frac.txt", "w") as result:
            result.write(str(bound_frac) + "\n")
        with open("res/a/"+Key_word+"_grad_upper.txt", "w") as result:
            result.write(str(grad_upper) + "\n")
        with open("res/a/"+Key_word+"_grad_lower.txt", "w") as result:
            result.write(str(grad_lower) + "\n")        
            
          
        