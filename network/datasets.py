import torch
import numpy as np
from random import randint
from torch.utils.data import Dataset


class StringDataset(Dataset):

    def __init__(self, h):
        self.feature = h.feature
        self.N, self.C, self.M = h.nb, h.C, h.M

    def __getitem__(self, index):
        return torch.from_numpy(self.feature[index]).unsqueeze(0)
         
    def __len__(self):
        return len(self.feature)


class TripletString(Dataset):

    def __init__(self, h):

        self.dist = h.dist
        self.feature = h.feature
        self.N, self.C, self.M = h.nt, h.C, h.M
        self.K = h.nt
        self.index = np.arange(self.N)
        self.avg_dist = np.mean(self.dist)

    def __getitem__(self, idx):
        anchor = idx
        while(1):
            p_idx = randint(0, self.N-1)
            n_idx = randint(0, self.N-1)
            if p_idx == anchor or n_idx == anchor:
                continue
            pos_dist = self.dist[anchor, p_idx]
            neg_dist = self.dist[anchor, n_idx]
            if pos_dist == neg_dist:
                continue
            if pos_dist > neg_dist:
                p_idx, n_idx = n_idx, p_idx
                pos_dist, neg_dist = neg_dist, pos_dist
            pos_neg_dist = self.dist[p_idx, n_idx]

            anchor = self.feature[anchor]
            positive = self.feature[p_idx]
            negative = self.feature[n_idx]

            return (
                anchor,
                positive,
                negative,
                pos_dist,
                neg_dist,
                pos_neg_dist,
            )

    def __len__(self):
        return self.N

class IdxFeat(Dataset):
    def __init__(self, h):
        self.idxs = h.idxs
        self.feature = h.feature
        self.dist = h.dist
        self.N = len(h.idxs)
        self.C, self.M = h.C, h.M
    def __getitem__(self, idx):
        info = self.idxs[idx]
        return (self.feature[info[0]], self.feature[info[1]], self.feature[info[2]], self.dist[info[0],info[1]], self.dist[info[0],info[2]],self.dist[info[1],info[2]])
    def __len__(self):
        return self.N

class ZeroFeat(Dataset):
    def __init__(self, h):
        self.idxs = h.idxs
        self.feature = h.feature
        self.dist = h.dist
        self.sim = h.sim
        self.N = len(h.idxs)
        self.C, self.M = h.C, h.M
    def __getitem__(self, idx):
        s1 = self.feature[self.sim[idx][0]]
        s2 = self.feature[self.sim[idx][1]]
        s3 = self.sim[idx][2]
        return (s1, s2, s3)
    def __len__(self):
        return len(self.sim)