import pickle as pkl
from math import ceil
import numpy as np
import pickle as pkl
import tqdm
import random as rd
from utils import *
import copy
from numpy.linalg import norm
sim_thre = 0.95
import os
os.environ['PYTHONHASHSEED'] = '0'


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T) 
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    return res


if __name__ == '__main__':
    rawdata = pkl.load(open("train/raw-filtered.pkl", 'rb'))
    [keys1, dists] = pkl.load(open("train/dist-filtered.pkl", 'rb'))
    [keys2, fts] = pkl.load(open("train/feat-filtered.pkl", 'rb'))
    keys = list(rawdata.keys())
    rawdata = [rawdata[keys[i]] for i in range(len(keys))]
    assert keys == keys1
    assert keys == keys2

    idxs = []
    for anchoridx in tqdm.tqdm(range(len(keys))):
        for simidx in range(anchoridx, len(keys)):
            if anchoridx == simidx:
                continue
            anchorfunc = [f[39:89] for f in fts[anchoridx] if max(f[39:89]) > 0]
            simfunc = [f[39:89] for f in fts[simidx] if max(f[39:89]) > 0]
            
            if len(anchorfunc) > 0 and len(simfunc) > 0:
                cospn = np.array(get_cos_similar_matrix(np.matrix(anchorfunc), np.matrix(simfunc)))
                if np.any((cospn < 0.999) & (cospn > sim_thre)):
                    idxs.append((anchoridx, simidx, dists[anchoridx][simidx]))
    pkl.dump([keys, idxs], open('train/idx-2.pkl', 'wb'))
    
