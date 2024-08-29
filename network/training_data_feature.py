
from itertools import count
from numpy.core.numerictypes import maximum_sctype
import psycopg2
import sys
import os
import re
import numpy as np
import pickle as pkl
from tqdm import tqdm
import random as rd

oprvocab = ["++", "-=", "--", ">>>=", "<<", "%", "!", "==", "~", "|", ">>>", "^", "*=", ">>", "||", "&&", "=", "&=", "<=", "!=", "|=", "+", "/=", "*", "-", "+=", "&", ">", "^=", ">=", "<", "/"]

if __name__ == '__main__':
    rawdata = pkl.load(
        open("train/raw-filtered.pkl", 'rb'))
    wvdict = pkl.load(
        open('wv.pkl', 'rb'))
    num = (0, 24)
    keys = list(rawdata.keys())
    fts = []
    for i in tqdm(keys):
        raw = rawdata[i]
        lineft = []
        for rawline in raw:
            [s, M, O, V, h, vcount, mcount] = rawline
            ft = [0 for _ in range(39)]
            ft[s] = 1
            for o in O:
                ft[7 + oprvocab.index(o)] += 1
            ft = np.array(ft)
            # mcount = 0
            Mfeature = np.arange(50, dtype=float)
            Mfeature = np.zeros_like(Mfeature)
            for m in M:
                try:
                    Mfeature += np.array(wvdict[m])
                    # mcount += 1
                except Exception:
                    pass
            if mcount > 1:
                Mfeature /= mcount
            # vcount = 0
            Vfeature = np.arange(50, dtype=float)
            Vfeature = np.zeros_like(Vfeature)
            for v in V:
                try:
                    Vfeature += np.array(wvdict[v])
                    # vcount += 1
                except Exception:
                    pass
            if vcount > 1:
                Vfeature /= vcount
            lineft.append(np.concatenate((ft, Mfeature, Vfeature), axis=0))
        fts.append(lineft)
    pkl.dump([keys,fts], open(
        'train/feat-filtered.pkl', 'wb'))
