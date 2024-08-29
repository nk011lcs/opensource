
from numpy.core.numerictypes import maximum_sctype
import psycopg2
import sys
import os
import re
import numpy as np
import pickle as pkl
from tqdm import tqdm
import random as rd

def lcs_mat(list1, list2):
    m = len(list1)
    n = len(list2)
    mat = [[0] * (n + 1) for row in range(m + 1)]
    for row in range(1, m + 1):
        for col in range(1, n + 1):
            if list1[row - 1]== list2[col - 1]:
            # if match(list1[row - 1], list2[col - 1]):
                mat[row][col] = mat[row - 1][col - 1] + 1
            else:
                mat[row][col] = max(mat[row][col - 1], mat[row - 1][col])
    return mat


def all_lcs(lcs_dict, mat, list1, list2, index1, index2):
    if ((index1, index2) in lcs_dict):
        return lcs_dict[(index1, index2)]
    if (index1 == 0) or (index2 == 0):
        return [[]]
    elif list1[index1 - 1]== list2[index2 - 1]:
    # elif match(list1[index1 - 1], list2[index2 - 1]):
        lcs_dict[(index1, index2)] = [prevs + [list1[index1 - 1]] for prevs in
                                      all_lcs(lcs_dict, mat, list1, list2, index1 - 1, index2 - 1)]
        return lcs_dict[(index1, index2)]
    else:
        lcs_list = []  
        if mat[index1][index2 - 1] >= mat[index1 - 1][index2]:
            before = all_lcs(lcs_dict, mat, list1, list2, index1, index2 - 1)
            for series in before: 
                if series not in lcs_list:
                    lcs_list.append(
                        series) 
        if mat[index1 - 1][index2] >= mat[index1][index2 - 1]:
            before = all_lcs(lcs_dict, mat, list1, list2, index1 - 1, index2)
            for series in before:
                if series not in lcs_list:
                    lcs_list.append(series)
        lcs_dict[(index1, index2)] = lcs_list
        return lcs_list


def lcs(list1, list2):
    mapping = dict()
    m = lcs_mat(list1, list2)
    return all_lcs(mapping, m, list1, list2, len(list1), len(list2))


if __name__ == '__main__':
    rawdata = pkl.load(
        open("train/raw-filtered.pkl", 'rb'))
    num = (0, 24)
    keys = list(rawdata.keys())
    dist = np.ndarray(shape=(len(keys), len(keys)), dtype=float)
    dist = np.zeros_like(dist)
    for i in tqdm(range(len(keys))):
        for j in range(i+1, len(keys)):
            l1 = [ _[4] for _ in rawdata[keys[i]]]
            l2 = [ _[4] for _ in rawdata[keys[j]]]
            if len(set(l1).intersection(l2)) == 0:
                dist[i][j] = 1
                dist[j][i] = 1
                continue
            d = lcs(l1, l2)
            d = 1 - float(len(d[0])) / max(len(l1), len(l2))
            dist[i][j] = d
            dist[j][i] = d
    
    pkl.dump([keys, dist], open(
        'train/dist-filtered.pkl', 'wb'))
