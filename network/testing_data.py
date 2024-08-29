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

conn = psycopg2.connect(user="yuanjung", password="123",
                        database="bigclonebench")
cur = conn.cursor()

stmvocab = ['IfStatement', 'ReturnStatement', 'ForStatement',
            'SwitchStatement', 'ExpressionStatement', 'WhileStatement', 'VariableDeclarationStatement']
wvdict = pkl.load(open('wv.pkl', 'rb'))


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
                O += f[1]
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

if __name__ == '__main__':
    rootpath = ""
    num = (25, 45)

    cur.execute('select * from clones where functionality_id >=' +
                str(num[0])+'and functionality_id <='+str(num[1])+';')
    rs = cur.fetchall()

    fileset = []
    for r in tqdm(rs):
        fileset.append(r[0])
        fileset.append(r[1])
    fileset = list(set(fileset))
    rd.shuffle(fileset)

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
        if len(stms) <= MAXLENGTH and len(stms)>=MINLENGTH:
        # if len(stms)>=MINLENGTH:
            os.system('cp '+str(i)+'.java '+ ' /test/'+str(i)+'.java')
