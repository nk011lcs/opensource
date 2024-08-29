import pickle as pkl
from tqdm import tqdm


if __name__ == '__main__':
    [testedfunc,clonedict,GT12,GVST3,GST3,GMT3] = pkl.load(
    open('groundtruth-t3.pkl', 'rb'))

    blockpath = 'ccaligner-clones'
    with open(blockpath, 'r')as fp:
        blines = fp.readlines()
    
    clones = []
    for b in tqdm(blines):
        if not (b.strip().split(',')[1] == '1' and  b.strip().split(',')[4] == '1'):
            continue
        c1 = int(b.strip().split(',')[0].split('/')[-1][:-5])
        c2 = int(b.strip().split(',')[3].split('/')[-1][:-5])
        
        clones.append((c1,c2))
    clones = list(set(clones))
    
    ccresult = []
    for c in tqdm(clones):
        idx1 = c[0]
        idx2 = c[1]
        if idx1 in testedfunc and idx2 in testedfunc:
            idxmin = min(idx1,idx2)
            idxmax = max(idx1,idx2)
            ccresult.append(str((idxmin,idxmax)))
    ccresult = list(set(ccresult))

    detectionresult = ccresult
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

