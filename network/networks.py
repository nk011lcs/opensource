import torch
import torch.nn as nn
from config import args
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2, x3 = x
        return self.embedding_net(x1), self.embedding_net(x2), self.embedding_net(x3)


class CNN(nn.Module):
    def __init__(self, C, M):
        super(CNN, self).__init__()
        self.C = C
        self.M = M
        self.out_size = args.embed_dim
        self.fco = nn.Linear(32, 10)
        self.fcm = nn.Linear(50, 10)
        self.fcv = nn.Linear(50, 10)
        self.conv = nn.Sequential()

        conv1 = nn.Conv2d(
            in_channels=7, out_channels=30, kernel_size=3, padding=1, stride=1)
        conv2 = nn.Conv2d(
            in_channels=30, out_channels=30, kernel_size=3, padding=1, stride=1)
        conv3 = nn.Conv2d(
            in_channels=30, out_channels=30, kernel_size=3, padding=1, stride=1)

        self.conv.add_module('conv1', conv1)
        self.conv.add_module('bn1', nn.BatchNorm2d(30))
        self.conv.add_module('pool1', nn.AvgPool2d(2))
        self.conv.add_module('relu1', nn.ReLU())

        self.conv.add_module('conv2', conv2)
        self.conv.add_module('bn2', nn.BatchNorm2d(30))
        self.conv.add_module('pool2', nn.AvgPool2d(2))
        self.conv.add_module('relu2', nn.ReLU())

        self.conv.add_module('conv3', conv3)
        self.conv.add_module('bn3', nn.BatchNorm2d(30))
        self.conv.add_module('pool3', nn.AvgPool2d(2))
        self.conv.add_module('relu3', nn.ReLU())

        self.flat_size = 900
        self.fc3 = nn.Linear(self.flat_size, self.out_size)

    def forward(self, x: torch.Tensor):
        N = len(x)
        x1 = x[:, :, :7]
        xo = x[:, :, 7:39]
        xm = x[:, :, 39:89]
        xv = x[:, :, 89:]
        xfeature = torch.cat([self.fco(xo), self.fcm(xm), self.fcv(xv)], dim=2)
        x1 = torch.mul(x[:, :, :1], xfeature)
        x2 = torch.mul(x[:, :, 1:2], xfeature)
        x3 = torch.mul(x[:, :, 2:3], xfeature)
        x4 = torch.mul(x[:, :, 3:4], xfeature)
        x5 = torch.mul(x[:, :, 4:5], xfeature)
        x6 = torch.mul(x[:, :, 5:6], xfeature)
        x7 = torch.mul(x[:, :, 6:7], xfeature)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2).view(
            x1.shape[0], x1.shape[1], 7, 30)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.view(N, self.flat_size)
        x = self.fc3(x)
        return x


class CNN1(nn.ModuleList):

    def __init__(self, C, M):
        super(CNN1, self).__init__()

        self.seq_len = M
        self.dropout = nn.Dropout(0.25)

        self.fco = nn.Linear(32, 10)
        self.fcm = nn.Linear(50, 10)
        self.fcv = nn.Linear(50, 10)

        self.kernel_1 = 3
        self.kernel_2 = 4
        self.kernel_3 = 5
        self.kernel_4 = 6

        self.in_size = C
        self.out_size = 5
        self.out = args.embed_dim
        self.stride = 1

        self.conv1_1 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_1, self.stride)
        self.conv1_2 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_2, self.stride)
        self.conv1_3 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_3, self.stride)
        self.conv1_4 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.conv2_1 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_1, self.stride)
        self.conv2_2 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_2, self.stride)
        self.conv2_3 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_3, self.stride)
        self.conv2_4 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_4, self.stride)

        self.flat_size = 3920
        self.fc1 = nn.Linear(self.flat_size, self.flat_size)
        self.fc2 = nn.Linear(self.flat_size, self.out)


    def forward(self, x):
        
        xo = x[:, :, 7:39]
        xm = x[:, :, 39:89]
        xv = x[:, :, 89:]
        xfeature = torch.cat([self.fco(xo), self.fcm(xm), self.fcv(xv)], dim=2)
        x1 = torch.mul(x[:, :, 0:1], xfeature)
        x2 = torch.mul(x[:, :, 1:2], xfeature)
        x3 = torch.mul(x[:, :, 2:3], xfeature)
        x4 = torch.mul(x[:, :, 3:4], xfeature)
        x5 = torch.mul(x[:, :, 4:5], xfeature)
        x6 = torch.mul(x[:, :, 5:6], xfeature)
        x7 = torch.mul(x[:, :, 6:7], xfeature)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2)


        x1 = self.conv1_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        x1 = self.conv2_1(x1)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        
        x2 = self.conv1_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)
        x2 = self.conv2_2(x2)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        x3 = self.conv1_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        x3 = self.conv2_3(x3)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        x4 = self.conv1_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        x4 = self.conv2_4(x4)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        union = self.fc1(union)
        union = torch.relu(union)
        out = self.fc2(union)
        return out 


class CNN2(nn.ModuleList):

    def __init__(self, C, M):
        super(CNN2, self).__init__()

        self.seq_len = M
        # self.dropout = nn.Dropout(0.25)

        self.fco = nn.Linear(32, 20)
        self.fcm = nn.Linear(50, 20)
        self.fcv = nn.Linear(50, 20)

        self.kernel_1 = 3
        self.kernel_2 = 4
        self.kernel_3 = 5
        self.kernel_4 = 6

        self.in_size = C
        self.out_size = 10
        self.out = args.embed_dim
        self.stride = 1

        self.conv1_1 = nn.Conv1d(self.seq_len, self.out_size,
                                self.kernel_1, self.stride)
        self.conv1_2 = nn.Conv1d(self.seq_len, self.out_size,
                                self.kernel_2, self.stride)
        self.conv1_3 = nn.Conv1d(self.seq_len, self.out_size,
                                self.kernel_3, self.stride)
        self.conv1_4 = nn.Conv1d(self.seq_len, self.out_size,
                                self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # self.conv2_1 = nn.Conv1d(self.out_size * 4, self.out_size,
        #                         self.kernel_1, self.stride)
        # self.conv2_2 = nn.Conv1d(self.out_size * 4, self.out_size,
        #                         self.kernel_2, self.stride)
        # self.conv2_3 = nn.Conv1d(self.out_size * 4, self.out_size,
        #                         self.kernel_3, self.stride)
        # self.conv2_4 = nn.Conv1d(self.out_size * 4, self.out_size,
        #                         self.kernel_4, self.stride)

        self.flat_size = 5280
        self.fc1 = nn.Linear(self.flat_size, self.flat_size)
        self.fc2 = nn.Linear(self.flat_size, self.out)


    def forward(self, x):
        
        # xo = x[:, :, 7:39]
        # xm = x[:, :, 39:89]
        # xv = x[:, :, 89:]
        # xfeature = torch.cat([self.fco(xo), self.fcm(xm), self.fcv(xv)], dim=2)
        # x1 = torch.mul(x[:, :, 0:1], xfeature)
        # x2 = torch.mul(x[:, :, 1:2], xfeature)
        # x3 = torch.mul(x[:, :, 2:3], xfeature)
        # x4 = torch.mul(x[:, :, 3:4], xfeature)
        # x5 = torch.mul(x[:, :, 4:5], xfeature)
        # x6 = torch.mul(x[:, :, 5:6], xfeature)
        # x7 = torch.mul(x[:, :, 6:7], xfeature)

        # x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2)


        x1 = self.conv1_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        # x1 = self.conv2_1(x1)
        # x1 = torch.relu(x1)
        # x1 = self.pool_1(x1)
        
        x2 = self.conv1_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)
        # x2 = self.conv2_2(x2)
        # x2 = torch.relu((x2))
        # x2 = self.pool_2(x2)

        x3 = self.conv1_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        # x3 = self.conv2_3(x3)
        # x3 = torch.relu(x3)
        # x3 = self.pool_3(x3)

        x4 = self.conv1_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        # x4 = self.conv2_4(x4)
        # x4 = torch.relu(x4)
        # x4 = self.pool_4(x4)

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        union = self.fc1(union)
        union = torch.relu(union)
        out = self.fc2(union)
        return out 



class CNN5(nn.ModuleList): #cnn1 in row

    def __init__(self, C, M):
        super(CNN5, self).__init__()

        self.seq_len = M
        self.insize = 210
        self.dropout = nn.Dropout(0.25)

        self.fco = nn.Linear(32, 10)
        self.fcm = nn.Linear(50, 10)
        self.fcv = nn.Linear(50, 10)

        self.kernel_1 = 3
        self.kernel_2 = 4
        self.kernel_3 = 5
        self.kernel_4 = 6

        self.in_size = C
        self.out_size = 5
        self.out = args.embed_dim
        self.stride = 1

        self.conv1_1 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_1, self.stride)
        self.conv1_2 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_2, self.stride)
        self.conv1_3 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_3, self.stride)
        self.conv1_4 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.conv2_1 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_1, self.stride)
        self.conv2_2 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_2, self.stride)
        self.conv2_3 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_3, self.stride)
        self.conv2_4 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_4, self.stride)

        self.flat_size = 1320
        self.fc1 = nn.Linear(self.flat_size, self.flat_size)
        self.fc2 = nn.Linear(self.flat_size, self.out)


    def forward(self, x):
        
        xo = x[:, :, 7:39]
        xm = x[:, :, 39:89]
        xv = x[:, :, 89:]
        xfeature = torch.cat([self.fco(xo), self.fcm(xm), self.fcv(xv)], dim=2)
        x1 = torch.mul(x[:, :, 0:1], xfeature)
        x2 = torch.mul(x[:, :, 1:2], xfeature)
        x3 = torch.mul(x[:, :, 2:3], xfeature)
        x4 = torch.mul(x[:, :, 3:4], xfeature)
        x5 = torch.mul(x[:, :, 4:5], xfeature)
        x6 = torch.mul(x[:, :, 5:6], xfeature)
        x7 = torch.mul(x[:, :, 6:7], xfeature)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=2)

        x = x.transpose(1, 2)

        x1 = self.conv1_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        x1 = self.conv2_1(x1)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        
        x2 = self.conv1_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)
        x2 = self.conv2_2(x2)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        x3 = self.conv1_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        x3 = self.conv2_3(x3)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        x4 = self.conv1_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        x4 = self.conv2_4(x4)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        union = self.fc1(union)
        union = torch.relu(union)
        out = self.fc2(union)
        return out 


class CNN3(nn.ModuleList): #cnn1 in row origin feature

    def __init__(self, C, M):
        super(CNN3, self).__init__()

        self.seq_len = M
        self.insize = 139
        self.dropout = nn.Dropout(0.25)

        self.fco = nn.Linear(32, 10)
        self.fcm = nn.Linear(50, 10)
        self.fcv = nn.Linear(50, 10)

        self.kernel_1 = 3
        self.kernel_2 = 4
        self.kernel_3 = 5
        self.kernel_4 = 6

        self.in_size = C
        self.out_size = 5
        self.out = args.embed_dim
        self.stride = 1

        self.conv1_1 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_1, self.stride)
        self.conv1_2 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_2, self.stride)
        self.conv1_3 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_3, self.stride)
        self.conv1_4 = nn.Conv1d(self.insize, self.out_size * 4,
                                self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.conv2_1 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_1, self.stride)
        self.conv2_2 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_2, self.stride)
        self.conv2_3 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_3, self.stride)
        self.conv2_4 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_4, self.stride)

        self.flat_size = 1320
        self.fc1 = nn.Linear(self.flat_size, self.flat_size)
        self.fc2 = nn.Linear(self.flat_size, self.out)


    def forward(self, x):
        x = x.transpose(1, 2)

        x1 = self.conv1_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        x1 = self.conv2_1(x1)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        
        x2 = self.conv1_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)
        x2 = self.conv2_2(x2)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        x3 = self.conv1_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        x3 = self.conv2_3(x3)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        x4 = self.conv1_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        x4 = self.conv2_4(x4)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        union = self.fc1(union)
        union = torch.relu(union)
        out = self.fc2(union)
        return out 


class CNN4(nn.ModuleList): #cnn1 in column origin feature

    def __init__(self, C, M):
        super(CNN4, self).__init__()

        self.seq_len = M
        # self.insize = 210
        self.dropout = nn.Dropout(0.25)

        self.fco = nn.Linear(32, 10)
        self.fcm = nn.Linear(50, 10)
        self.fcv = nn.Linear(50, 10)

        self.kernel_1 = 3
        self.kernel_2 = 4
        self.kernel_3 = 5
        self.kernel_4 = 6

        self.in_size = C
        self.out_size = 5
        self.out = args.embed_dim
        self.stride = 1

        self.conv1_1 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_1, self.stride)
        self.conv1_2 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_2, self.stride)
        self.conv1_3 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_3, self.stride)
        self.conv1_4 = nn.Conv1d(self.seq_len, self.out_size * 4,
                                self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.conv2_1 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_1, self.stride)
        self.conv2_2 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_2, self.stride)
        self.conv2_3 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_3, self.stride)
        self.conv2_4 = nn.Conv1d(self.out_size * 4, self.out_size,
                                self.kernel_4, self.stride)

        self.flat_size = 2500
        self.fc1 = nn.Linear(self.flat_size, self.flat_size)
        self.fc2 = nn.Linear(self.flat_size, self.out)


    def forward(self, x):
        
        x1 = self.conv1_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        x1 = self.conv2_1(x1)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        
        x2 = self.conv1_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)
        x2 = self.conv2_2(x2)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        x3 = self.conv1_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        x3 = self.conv2_3(x3)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        x4 = self.conv1_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        x4 = self.conv2_4(x4)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)

        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        union = self.fc1(union)
        union = torch.relu(union)
        out = self.fc2(union)
        return out 
