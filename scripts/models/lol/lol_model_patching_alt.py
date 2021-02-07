import os
import random

import torch
from kornia import translate, rotate, scale
from torch import nn
from torch.nn.functional import interpolate

from scripts.models.lol.lstm import ConvLSTM
from scripts.new.patching.extract_tensor_patch import extract_tensor_patch


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
        self.l1 = nn.Linear(512, 6)
        self.l1.bias.data[0] = 0  # base x
        self.l1.bias.data[1] = 0  # base y
        self.l1.bias.data[2] = 0  # upper_height
        self.l1.bias.data[3] = 0  # lower_height
        self.l1.bias.data[4] = 0  # angle
        self.l1.bias.data[5] = -6  # stop confidence

    def forward(self, y):
        # y = self.l2(y)
        y = self.l1(y)

        return y


class LineOutlinerTsa(nn.Module):

    def __init__(self, path=None, patch_ratio=5, tsa_size=3, patch_size=64):
        super(LineOutlinerTsa, self).__init__()
        self.tsa_size = tsa_size
        self.patch_size = patch_size
        self.patch_ratio = patch_ratio

        self.tsa = TemporalAttension(3)
        self.initial_convolutions = PreLstmConvolutions().cuda()

        # summary(self.initial_convolutions, (3, 32, 32), batch_size=1)
        self.memory_layer = MemoryLayer().cuda()

        self.final_convolutions = PostLstmConvolutions().cuda()
        # summary(self.final_convolutions, (64, 4, 4))

        self.fully_connected = FullyConnectedLayer().cuda()
        # summary(self.fully_connected, (1, 128))

        if path is not None and os.path.exists(path):
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state)
            self.eval()
        elif path is not None:
            print("\nCould not find path", path, "\n")

    def forward(self,
                img,
                sol,
                steps,
                reset_threshold=None,
                max_steps=None,
                disturb_sol=True,
                height_disturbance=0.5,
                angle_disturbance=30,
                translate_disturbance=10):

        img.cuda()

        # tensor([tsa, channels, width, height])
        input = torch.zeros((self.tsa_size, 3, self.patch_size, self.patch_size)).cuda()

        steps_ran = 0

        sol = {
            "upper_point": sol[0],
            "base_point": sol[1],
            "angle": sol[3][0],
        }

        if disturb_sol:
            x = random.uniform(0, translate_disturbance)
            y = random.uniform(0, translate_disturbance)
            sol["upper_point"][0] += x
            sol["upper_point"][1] += y
            sol["base_point"][0] += x
            sol["base_point"][1] += y
            sol["angle"] += random.uniform(0, angle_disturbance)

        current_height = torch.dist(sol["upper_point"].clone(), sol["base_point"].clone()).cuda()
        current_height = current_height * (1 if not disturb_sol else (1 + random.uniform(0, height_disturbance)))
        current_angle = sol["angle"].clone().cuda()
        current_base = sol["base_point"].clone().cuda()

        results = []
        tsa_sequence = []

        while (max_steps is None or steps_ran < max_steps):

            current_scale = (self.patch_ratio * current_height / self.patch_size).cuda()

            img_width = img.shape[2]
            img_height = img.shape[3]

            # current_angle = torch.mul(current_angle, -1)

            if img_width < current_base[0].item() or current_base[0].item() < 0:
                break
            if img_height < current_base[1].item() or current_base[0].item() < 0:
                break
            patch_parameters = torch.stack([current_base[0],  # x
                                            current_base[1],  # y
                                            torch.mul(torch.deg2rad(current_angle), -1),  # angle
                                            current_height]).unsqueeze(0)
            patch_parameters = patch_parameters.cuda()
            try:
                patch = extract_tensor_patch(img, patch_parameters, size=self.patch_size)  # size
            except:
                break
            # patch = interpolate(patch, (self.patch_size, self.patch_size, 3))

            # Shift input left
            input = torch.stack([pic for pic in input[1:]] + [patch.squeeze(0)])

            y = input.cuda().unsqueeze(0)
            y = self.tsa(y)
            after_tsa_copy = y.detach().cpu().clone()
            tsa_sequence.append(after_tsa_copy)
            y = y.squeeze(0)
            y = self.initial_convolutions(y)
            y = y.unsqueeze(0)
            y = self.memory_layer(y)
            y = y.unsqueeze(0)
            y = self.final_convolutions(y)
            y = y.unsqueeze(0)
            y = torch.flatten(y, 1)
            y = torch.flatten(y, 0)
            y = self.fully_connected(y)

            # Biases
            size = input[-1, :, :, :].shape[1] / self.patch_ratio
            base_prior_x = size
            base_prior_y = 0
            y[0] = torch.add(y[0], base_prior_x)
            y[1] = torch.add(y[1], base_prior_y)
            y[2] = torch.add(y[2], size)

            base_point = torch.stack([y[0], y[1]])
            current_angle = torch.add(current_angle, y[4])

            rotation_matrix = torch.tensor(
                [[torch.cos(torch.deg2rad(current_angle)), -1.0 * torch.sin(torch.deg2rad(current_angle))],
                 [1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))]]).cuda()

            scale_matrix = torch.tensor([[current_scale, 0.], [0., current_scale]]).cuda()

            upper_point = torch.stack([torch.tensor(0.0).cuda(), torch.mul(y[2], -1)])
            lower_point = torch.stack([torch.tensor(0.0).cuda(), y[3]])

            # Trasnform base_point
            base_point = torch.matmul(base_point, rotation_matrix.t())
            base_point = torch.matmul(base_point, scale_matrix)
            current_base = torch.add(base_point, current_base)

            # Rotate outlines
            upper_point = torch.matmul(upper_point, scale_matrix)
            lower_point = torch.matmul(lower_point, scale_matrix)
            upper_point = torch.matmul(upper_point, rotation_matrix.t())
            lower_point = torch.matmul(lower_point, rotation_matrix.t())
            upper_point = torch.add(upper_point, current_base)
            lower_point = torch.add(lower_point, current_base)

            current_height = torch.dist(upper_point, current_base)
            current_height = torch.max(current_height, torch.tensor(16).cuda())
            stop_confidence = torch.sigmoid(y[5])

            results.append(torch.stack([
                upper_point.clone(),
                current_base.clone(),
                lower_point.clone(),
                torch.stack([current_angle.clone(), torch.tensor(0).cuda()]),
                torch.stack([stop_confidence.clone(), torch.tensor(0).cuda()])
            ], dim=0))

            steps_ran += 1

            if max_steps is None and steps_ran >= len(steps) - 1:
                break
            elif reset_threshold is not None:
                upper_distance = torch.dist(upper_point.clone().detach().cpu(), steps[steps_ran][0])
                lower_distance = torch.dist(lower_point.clone().detach().cpu(), steps[steps_ran][2])
                current_threshold = reset_threshold
                if upper_distance > current_threshold \
                        or lower_distance > current_threshold:
                    break

        return None if len(results) == 0 else torch.stack(results), steps_ran, tsa_sequence
