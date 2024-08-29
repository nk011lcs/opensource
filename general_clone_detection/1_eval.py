import pickle as pkl
from tqdm import tqdm

import numpy as np
import pickle as pkl

from numpy.core.numeric import zeros_like, ones_like
import time

def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[np.nonzero(l2 < 0)] = 0.0
    return np.sqrt(l2)

if __name__ == '__main__':
    tmax = 0.47
    tmin = 0.0
    # print(time.time())
    [keys, emb] = pkl.load(open(
                "emb-CNN1-min6.pkl", 'rb'))
    emb = np.asarray(emb)
    dist_mask_0 = zeros_like(np.ndarray((len(keys), len(keys)), dtype=int))
    dist_mask_1 = ones_like(np.ndarray((len(keys), len(keys)), dtype=int))

    dist1 = np.arange(len(keys)).repeat(len(keys)).reshape((len(keys), len(keys)))
    dist2 = dist1.transpose()
    dist_mask = np.where(dist1 < dist2, dist_mask_1, dist_mask_0)

    dist = l2_dist(emb, emb)
    # print(time.time())
    clonepairs = np.nonzero(np.where(((dist <= tmax) &(dist >= tmin) & dist_mask == 1), dist_mask_1, dist_mask_0))
    clonepairs = list(zip(clonepairs[0], clonepairs[1]))
    clonepairs = [(min(int(keys[i[0]]), int(keys[i[1]])),max(int(keys[i[0]]), int(keys[i[1]])))for i in clonepairs]
    
    [testedfunc,clonedict,GT12,GVST3,GST3,GMT3] = pkl.load(
        open('groundtruth-t3.pkl', 'rb'))
    embresult = []
    for c in clonepairs:
        idx1 = c[0]
        idx2 = c[1]
        if idx1 in testedfunc and idx2 in testedfunc:
            idxmin = min(idx1,idx2)
            idxmax = max(idx1,idx2)
            embresult.append(str((idxmin,idxmax)))

    detectionresult = embresult
    keylist = {1:'T12', 2:'VST3',3:'ST3',4:'MT3',0:'IGNORE'}
    hit = {'T12':0, 'VST3':0,'ST3':0,'MT3':0, 'IGNORE':0}
    pairs =  {'T12':[],'VST3':[],'ST3':[],'MT3':[],'IGNORE':[]}
    FP = 0
    for c in detectionresult:
        try:
            clonetype=clonedict[c]
            hit[keylist[clonetype] ]+= 1
            pairs[keylist[clonetype]].append(c)
        except KeyError:
            FP += 1

    hit_report = hit['T12']+hit['VST3']+hit['ST3']+hit['MT3']
    ignore_report = hit['IGNORE']
    false_report = FP

    print('precision')
    P = (hit_report+ignore_report)/(hit_report+ignore_report+false_report)
    print(P)

    print('recall')
    print(hit['T12']/GT12, hit['VST3']/GVST3, hit['ST3']/GST3, hit['MT3']/GMT3, hit['IGNORE']/683747)
    
    print('recall-13')
    R = hit_report/(GT12+GVST3+GST3+GMT3)
    print(R)

    print('recall-all')
    R = (hit_report+ignore_report)/(GT12+GVST3+GST3+GMT3+683747)
    print(R)

    print('f1-all')
    print(2*P*R/(P+R))

