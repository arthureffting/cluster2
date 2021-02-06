import os
import torch
from torch import nn
from torch.nn import Dropout

from scripts.models.lol.lstm import ConvLSTM


class TemporalAttension(nn.Module):

    def __init__(self, channels):
        super(TemporalAttension, self).__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

        init = torch.zeros((3, 3))
        init[1, 1] = 1

        self.conv.weight.data.copy_(init)

        self.conv.bias.data.copy_(torch.zeros(channels))

    def forward(self, x):
        x1 = x[:, :-1]
        x2 = x[:, 1:]
        o = x2 - x1
        o = torch.cat((torch.zeros((x.size(0), 1, x.size(2), x.size(3), x.size(4)), device=x.device), o), 1)
        o = o.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.conv(o).view(x.size()) * x + x

        return x


class PreLstmConvolutions(nn.Module):

    def __init__(self):
        super(PreLstmConvolutions, self).__init__()

        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2, 2)

        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(2, 2)

        self.c3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(64)
        self.r3 = nn.ReLU()
        self.p3 = nn.MaxPool2d(2, 2)

    def forward(self, y):
        y = y.cuda()

        y = self.c1(y)
        y = self.b1(y)
        y = self.r1(y)
        y = self.p1(y)

        y = self.c2(y)
        y = self.b2(y)
        y = self.r2(y)
        y = self.p2(y)

        y = self.c3(y)
        y = self.b3(y)
        y = self.r3(y)
        y = self.p3(y)

        return y


class MemoryLayer(nn.Module):

    def __init__(self):
        super(MemoryLayer, self).__init__()
        ks = [(3, 3), (3, 3)]
        nm = [64, 64]
        self.convlstm = ConvLSTM(input_dim=64,
                                 hidden_dim=nm,
                                 kernel_size=ks,
                                 num_layers=len(nm),
                                 batch_first=True,
                                 bias=False,
                                 return_all_layers=False)

    def forward(self, y):
        y, hidden = self.convlstm(y)
        y = y[0]
        y = y[0, -1, :, :, :]  # (32,64,64)
        return y


class PostLstmConvolutions(nn.Module):

    def __init__(self):
        super(PostLstmConvolutions, self).__init__()

        self.c1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(128)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2, 2)

        self.c2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(128)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(2, 2)

        # self.c3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.b3 = nn.BatchNorm2d(256)
        # self.r3 = nn.ReLU()
        # self.p3 = nn.MaxPool2d(2, 2)

        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, y):
        y = self.c1(y)
        y = self.b1(y)
        y = self.r1(y)
        y = self.p1(y)

        y = self.c2(y)
        y = self.b2(y)
        y = self.r2(y)
        y = self.p2(y)

        # y = self.c3(y)
        # y = self.b3(y)
        # y = self.r3(y)
        # y = self.p3(y)

        # y = self.dropout(y)
        return y


class FullyConnectedLayer(nn.Module):

    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        # Linear layer at the end to get angle and sizing
        # self.l2 = nn.Linear(512, 128)
        self.l1 = nn.Linear(128, 8)
        self.l1.bias.data[0] = 0  # upper x
        self.l1.bias.data[1] = 0  # upper y
        self.l1.bias.data[2] = 0  # base x
        self.l1.bias.data[3] = 0  # base y
        self.l1.bias.data[4] = 0  # lower x
        self.l1.bias.data[5] = 0  # lower y
        self.l1.bias.data[6] = 0  # angle
        self.l1.bias.data[7] = 0  # stop confidence

    def forward(self, y):
        # y = self.l2(y)
        y = self.l1(y)
        return y


class LineOutlinerTsaDropout(nn.Module):

    def __init__(self, path=None, patch_ratio=5):
        super(LineOutlinerTsaDropout, self).__init__()
        self.path = path
        self.tsa = TemporalAttension(3)
        self.patch_ratio = patch_ratio
        self.initial_convolutions = PreLstmConvolutions().cuda()

        # summary(self.initial_convolutions, (3, 32, 32), batch_size=1)
        self.memory_layer = MemoryLayer().cuda()

        self.final_convolutions = PostLstmConvolutions().cuda()
        # summary(self.final_convolutions, (64, 4, 4))

        self.fully_connected = FullyConnectedLayer().cuda()
        # summary(self.fully_connected, (1, 128))

        self.dropout = Dropout(0.5)

        if path is not None and os.path.exists(path):
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state)

    def forward(self, inputs, copy=False):

        size = inputs[0, -1, :, :, :].shape[1] / self.patch_ratio
        upper_prior_x = size
        upper_prior_y = - size
        base_prior_x = size
        base_prior_y = 0
        lower_prior_x = size
        lower_prior_y = 0

        y = inputs.cuda()
        y = self.tsa(y)
        after_tsa_copy = y.detach().cpu().clone() if copy else None
        y = y.squeeze(0)
        y = self.initial_convolutions(y)
        y = y.unsqueeze(0)
        y = self.memory_layer(y)
        y = y.unsqueeze(0)
        y = self.final_convolutions(y)
        y = y.unsqueeze(0)
        y = torch.flatten(y, 1)
        y = torch.flatten(y, 0)
        y = self.dropout(y)
        y = self.fully_connected(y)

        y[0] = torch.add(y[0], upper_prior_x)
        y[1] = torch.add(y[1], upper_prior_y)
        y[2] = torch.add(y[2], base_prior_x)
        y[3] = torch.add(y[3], base_prior_y)
        y[4] = torch.add(y[4], lower_prior_x)
        y[5] = torch.add(y[5], lower_prior_y)
        # y[6] = y[6] + 0
        y[7] = torch.sigmoid(y[7])

        if copy:
            return y, after_tsa_copy
        else:
            return y
