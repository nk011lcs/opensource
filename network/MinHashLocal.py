from my_minhash import MinHash
from datasketch.hashfunc import sha1_hash32
hashsize = 12
def myhash(d):
    h = sha1_hash32(d)%((1 << hashsize) - 1)
    print(h)
    return h

set1 = set([''])
set2 = set(['get','Instance'])
set3 = set([''])
set4 = set(['add'])
set5 = set(['print','get','Instance','Stack'])
set6 = set(['print','get','Instance','Stack','Trace'])

m1 = MinHash(num_perm=5, hashfunc=myhash, size=hashsize)
m2 = MinHash(num_perm=5, hashfunc=myhash, size=hashsize)
m3 = MinHash(num_perm=5, hashfunc=myhash, size=hashsize)
m4 = MinHash(num_perm=5, hashfunc=myhash, size=hashsize)
m5 = MinHash(num_perm=5, hashfunc=myhash, size=hashsize)
m6 = MinHash(num_perm=5, hashfunc=myhash, size=hashsize)
for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))
for d in set4:
    m4.update(d.encode('utf8'))
for d in set5:
    m5.update(d.encode('utf8'))
for d in set6:
    m6.update(d.encode('utf8'))
print(str(m1.hashvalues))
print(m2.hashvalues)
print(m3.hashvalues)
print(m4.hashvalues)
print(m5.hashvalues)
print(m6.hashvalues)
print(m5.jaccard(m6))