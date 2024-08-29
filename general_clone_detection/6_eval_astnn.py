import pickle as pkl
from tqdm import tqdm

if __name__ == '__main__':
    with open('astnn-result.pkl','rb') as fp:
        results = pkl.load(fp)
    
    [testedfunc,clonedict,GT12,GVST3,GST3,GMT3] = pkl.load(
        open('groundtruth-t3.pkl', 'rb'))


    keylist = {1:'T12', 2:'VST3',3:'ST3',4:'MT3',0:'IGNORE'}
    hit = {'T12':0, 'VST3':0,'ST3':0,'MT3':0, 'IGNORE':0}
    pairs =  {'T12':[],'VST3':[],'ST3':[],'MT3':[],'IGNORE':[]}
    FP = 0
    for c in tqdm(results):
        if results[c] == 1:
            idx1 = c.split(',')[0].strip()
            idx2 = c.split(',')[1].strip()
            thekey = '(' + idx1 + ', ' + idx2 + ')'
            try:
                clonetype=clonedict[thekey]
                hit[keylist[clonetype] ]+= 1
                pairs[keylist[clonetype]].append(thekey)
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
