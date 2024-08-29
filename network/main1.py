import os
import time
import numpy as np
import pickle as pkl
import torch
from tqdm import tqdm
from config import args
from trainer1 import train_epoch
from datasets import StringDataset, IdxFeat, ZeroFeat
from tester import test_mse, testsim_precision, threshold_recall, clone_detection, test_recall
import random as rd

class DataHandler:
    def __init__(self):
        self.args = args
        self.dataset = args.dataset
        self.C = args.in_dim
        self.M = args.maxl
        if args.traintest == 'train':
            [keys, idxs] = pkl.load(open("train/idx-1.pkl", 'rb'))
            [keys1, feat] = pkl.load(open("train/feat-filtered.pkl", 'rb'))
            [keys2, dist] = pkl.load(open("train/dist-filtered.pkl", 'rb'))
            [keys3, sim] = pkl.load(open("train/idx-2.pkl", 'rb'))
            sim = sim + sim
            rd.shuffle(sim)
            assert(keys == keys1)
            assert(keys == keys2)
            assert(keys == keys3)
            trainfeature = []
            for t in feat:
                tt = np.pad(t, ((0, self.M), (0, 0)),
                            'constant', constant_values=0.0)[:self.M, :].astype(np.float32)
                trainfeature.append(tt)
            self.feature = trainfeature
            self.dist = dist
            self.key = keys1
            self.idxs = idxs[:1000000]
            self.sim = sim[:2000000]
        elif args.traintest == 'test':
            [keys, bfeature] = pkl.load(open("test/feat-test-min6.pkl", 'rb'))
            basefeature = []
            for b in bfeature:
                bb = np.pad(b, ((0, self.M), (0, 0)),
                            'constant', constant_values=0.0)[:self.M, :].astype(np.float32)
                basefeature.append(bb)
            self.feature = basefeature
            self.key = keys
            self.nb = len(self.key)
            self.dist = []
            self.diffidx = []
            self.simidx = []


if __name__ == "__main__":
    device = torch.device("cuda:1")
    testmodel = "model/model1-256-CNN4.torch"
    model_file = "model/model1-{}-{}.torch".format(args.embed_dim, args.nettype)
    if args.traintest == "train":
        h = DataHandler()
        train_loader = IdxFeat(h)
        zero_loader = ZeroFeat(h)
        start_time = time.time()
        model = train_epoch(args, train_loader, zero_loader, device)
        if args.save_model:
            torch.save(model, model_file)
        train_time = time.time() - start_time
        print("# Training time: " + str(train_time))
    elif args.traintest == "test":
        h = DataHandler()
        test_dataset = StringDataset(h)
        model = torch.load(testmodel)
        model.eval()
        net = model.embedding_net
        net.eval()
        embedding = []
        for i, x in enumerate(tqdm(test_dataset)):
            embedding.append(net(x.to(device)).squeeze().cpu().data.numpy())
        pkl.dump([h.key,embedding], open("test/emb-{}-min6.pkl".format(args.nettype), 'wb'))


    