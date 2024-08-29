import pickle as pkl
from tqdm import tqdm


if __name__ == '__main__':
    [testedfunc,clonedict,GT12,GVST3,GST3,GMT3] = pkl.load(
        open('groundtruth-t3.pkl', 'rb'))

    blockpath = 'nicad-test-min6_functions-consistent-clones-0.30.xml'
    with open(blockpath, 'r')as fp:
        blines = fp.readlines()
    
    funclist = []
    for b in tqdm(blines):
        if '<source file="/home/sise/sda/gyj/EDE/NiCad-6.2/test-min6/' in b:
            c = int(b.strip().split(' ')[1][len("file=\"/home/sise/sda/gyj/EDE/NiCad-6.2/test-min6/"):-6])
            funclist.append(c)
    clones = []
    for i in range(0,len(funclist),2):
        clones.append((funclist[i],funclist[i+1]))
    clones = list(set(clones))
    
    nicadresult = []
    for c in clones:
        idx1 = c[0]
        idx2 = c[1]
        if idx1 in testedfunc and idx2 in testedfunc:
            idxmin = min(idx1,idx2)
            idxmax = max(idx1,idx2)
            nicadresult.append(str((idxmin,idxmax)))
    
    
    detectionresult = nicadresult
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
