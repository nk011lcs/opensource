import pickle as pkl
import numpy as np
from numpy.core.numeric import zeros_like, ones_like
from tqdm import tqdm
if __name__ == '__main__':
    [testedfunc,clonedict,GT12,GVST3,GST3,GMT3] = pkl.load(
        open('groundtruth-t3.pkl', 'rb'))

    blockdict = {}
    clones = []

    with open('sourcererCC/files-stats-0.stats','r')as fp:
        lines = fp.readlines()
    fnamedict = {}
    for l in lines:
        idx = l.split(',')[1]
        fname = l.split(',')[2].split('/')[-1][:-6]
        fnamedict[idx] = fname


    # print('f')
    sourceresult = []

    with open('sourcererCC/result.txt','r')as fp:
        lines = fp.readlines()
    for l in lines:
        c = l.strip().split(',')
        idx1 = int(fnamedict[c[1]])
        idx2 = int(fnamedict[c[3]])
        idxmin = min(idx1,idx2)
        idxmax = max(idx1,idx2)
        sourceresult.append(str((idxmin,idxmax)))
    

    detectionresult = sourceresult
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

