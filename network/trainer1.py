import time

from tqdm import tqdm
from networks import *


def train_epoch(args, train_set, zero_set, device):
    # N = train_set.N
    C, M = train_set.C, train_set.M
    torch.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    zero_loader = torch.utils.data.DataLoader(
        zero_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    EmbeddingNet = eval(args.nettype)
    net = EmbeddingNet(C, M).to(device)
    model = TripletNet(net).to(device)
    losser = TripletLoss(args)
    zerolosser = ThresholdLoss(args)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    start_time = time.time()
    with tqdm(total=args.epochs, desc="# training") as p_bar:
        for epoch in range(args.epochs):
            agg = 0.0
            agg_r = 0.0
            agg_m = 0.0
            agg_zero = 0.0
            for ind, batch in enumerate(train_loader):
                (anchor, pos, neg, pos_dist, neg_dist, pn_dist) = (
                    i.to(device) for i in batch)
                optimizer.zero_grad()
                output = model((anchor, pos, neg))
                r, m, loss = losser(output, pos_dist, neg_dist, pn_dist, epoch)

                loss.backward()
                optimizer.step()
                agg += loss.item()
                agg_r += r.item()
                agg_m += m.item()
                p_bar.set_description(
                    "# Epoch: %3d; Batch: %4d; Time: %.1fs; L_rank: %.6f; L_mse: %.6f; L_zero: %.6f"
                    % (
                        epoch,
                        ind,
                        time.time() - start_time,
                        agg_r / (ind + 1),
                        agg_m / (ind + 1),
                        agg_zero
                    )
                )
            if not epoch+2 >= args.epochs:
                for ind, batch in enumerate(zero_loader):
                    (s1, s2, d) = (i.to(device) for i in batch)
                    optimizer.zero_grad()
                    emb1 = net(s1)
                    emb2 = net(s2)
                    loss = zerolosser(emb1, emb2, d)
                    loss.backward()
                    optimizer.step()
                    agg_zero += loss.item()
                    p_bar.set_description(
                        "# Epoch: %3d; Batch: %4d; Time: %.1fs; L_rank: %.6f; L_mse: %.6f; L_zero: %.6f"
                        % (
                            epoch,
                            ind,
                            time.time() - start_time,
                            agg_r / (len(train_loader) + 1),
                            agg_m / (len(train_loader) + 1),
                            agg_zero /(ind +1)
                        )
                    )
            
            p_bar.update(1)
            
    return model


class TripletLoss(nn.Module):
    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.l, self.r = 0, 0
        step = args.epochs // 5
        self.Ls = {
            step * 0: (1, 0.01),
            step * 1: (1, 0.1),
            step * 2: (1, 1),
            step * 3: (0.1, 1),
            step * 4: (0.01, 1)
        }

    def dist(self, ins, pos):
        return torch.norm(ins - pos, dim=1)

    def forward(self, x, pos_dist, neg_dist, pn_dist, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x

        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = (neg_dist - pos_dist)
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)
        mse_loss = (pos_embed_dist - pos_dist) ** 2 + (neg_embed_dist -
                                                       neg_dist) ** 2 + (pos_neg_embed_dist - pn_dist) ** 2

        return torch.mean(rank_loss), \
            torch.mean(mse_loss), \
            torch.mean(self.l * rank_loss +
                       self.r * mse_loss)


class ThresholdLoss(nn.Module):
    def __init__(self, args):
        super(ThresholdLoss, self).__init__()
        self.threshold = args.threshold

    def dist(self, ins, pos):
        return torch.norm(ins - pos, dim=1)

    def forward(self, s1, s2, d):
        embed_dist = self.dist(s1, s2)
        loss = F.relu(embed_dist - d + 0.01)
        return torch.mean(loss)
