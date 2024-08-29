
import psycopg2
import sys
import os
import re
import numpy as np
import pickle as pkl
from tqdm import tqdm
import random as rd
from numpy.core.numerictypes import maximum_sctype
import psycopg2
import sys
import os
import re
import numpy as np
import pickle as pkl
from tqdm import tqdm
import random as rd
MAXLENGTH = 80
MINLENGTH = 6

os.environ["PYTHONHASHSEED"] = "0"

featurelength = 139
stmvocab = ['IfStatement', 'ReturnStatement', 'ForStatement',
            'SwitchStatement', 'ExpressionStatement', 'WhileStatement', 'VariableDeclarationStatement']
oprvocab = ["++", "-=", "--", ">>>=", "<<", "%", "!", "==", "~", "|", ">>>", "^", "*=", ">>", "||", "&&", "=", "&=", "<=", "!=", "|=", "+", "/=", "*", "-", "+=", "&", ">", "^=", ">=", "<", "/"]
wvdict = pkl.load(open('wv.pkl', 'rb'))

sim_thre = 0.95


def splitcamel(l):
    l = re.sub(r'[^A-Za-z ]+', '', l)
    return re.sub('(?!^)([A-Z][a-z]+)', r' \1', l).split()

def stm(line):
    d = []
    line = line.replace(' ','')
    s = line.split(':')[0]
    s = stmvocab.index(s)
    feat = ':'.join(line.split(':')[1:]).split(';')
    M = []
    V = []
    O = []
    countv = 0
    countm = 0
    for f in feat:
        if len(f)>0:
            f = f.split(':')
            if f[0] == 'VariableType':
                V += splitcamel(f[1])
                countv += 1
            elif f[0] == 'MethodInvocation':
                M += splitcamel(f[1])
                countm += 1
            else:
                O.append(f[1])
    M = list(set(M))
    V = list(set(V))
    M = [i.lower() for i in M]
    M = [i for i in M if i in wvdict]
    V = [i.lower() for i in V]
    V = [i for i in V if i in wvdict]
    O=sorted(O)
    M=sorted(M)
    V=sorted(V)
    h = hash(str([s, M, O, V]))
    d = [s, M, O, V, h, countv, countm]
    return d

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
    
def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T) 
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    return res

if __name__ == '__main__':
    rootpath = ""

    with open('test/testlist-min6.txt', 'r') as fp:
        fileset = eval(fp.read())

    datadict = {}
    for i in tqdm(fileset):
        stms = []
        with open(rootpath+str(i)+'.java.out','r') as fp:
            lines = fp.readlines()
            for j in lines:
                try:
                    stms.append(stm(j.strip()))
                except Exception:
                    pass
        if len(stms)>=MINLENGTH:
        # if len(stms) <= MAXLENGTH and len(stms)>=MINLENGTH:
            datadict[i] = stms
    rawdata = datadict
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
                try:
                    ft[7 + oprvocab.index(o)] += 1
                except ValueError:
                    continue
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
        fts.append(lineft[:80])
    pkl.dump([keys,fts], open('test/feat-test-min6.pkl', 'wb'))



