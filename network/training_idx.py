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
    for i in tqdm.tqdm(range(1000000)):
        while(1):
            anchoridx = rd.choice(list(range(len(keys))))
            posidx = rd.choice(list(range(len(keys))))
            negidx = rd.choice(list(range(len(keys))))
            ranchor = rawdata[anchoridx]
            rpos = rawdata[posidx]
            rneg = rawdata[negidx]
            d_ap = dists[anchoridx][posidx]
            d_an = dists[anchoridx][negidx]
            if d_ap == d_an:
                continue

            if d_ap > d_an:
                posidx, negidx = negidx, posidx
                d_ap , d_an =  d_an, d_ap
            
            anchorfunc = [f[39:89] for f in fts[anchoridx] if max(f[39:89]) > 0]
            posfunc = [f[39:89] for f in fts[posidx] if max(f[39:89]) > 0]
            negfunc = [f[39:89] for f in fts[negidx] if max(f[39:89]) > 0]
            if len(anchorfunc) > 0 and len(posfunc) > 0:
                cosap = np.array(get_cos_similar_matrix(np.matrix(anchorfunc), np.matrix(posfunc)))
                if np.any((cosap < 0.999) & (cosap > sim_thre)):
                    continue
            if len(anchorfunc) > 0 and len(negfunc) > 0:
                cosan = np.array(get_cos_similar_matrix(np.matrix(anchorfunc), np.matrix(negfunc)))
                if np.any((cosan < 0.999) & (cosan > sim_thre)):
                    continue
            if len(posfunc) > 0 and len(negfunc) > 0:
                cospn = np.array(get_cos_similar_matrix(np.matrix(posfunc), np.matrix(negfunc)))
                if np.any((cospn < 0.999) & (cospn > sim_thre)):
                    continue
            
            idxs.append((anchoridx, posidx, negidx))
            break
    pkl.dump([keys, idxs], open('train/idx-1.pkl', 'wb'))
    
