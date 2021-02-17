import os
import random

import torch
from shapely.geometry import Point, Polygon, LineString
from torch import nn

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
        self.l1 = nn.Linear(512, 7)
        # 1 first_angle
        # 2 next_angle
        self.l1.bias.data[0] = 0  # base x
        self.l1.bias.data[1] = 0  # base y
        self.l1.bias.data[2] = 0  # upper height sigmoid
        self.l1.bias.data[3] = -5  # lower height sigmoid
        self.l1.bias.data[4] = 0  # angle

    def forward(self, y):
        # y = self.l2(y)
        y = self.l1(y)
        return y


class StopModule(nn.Module):

    def __init__(self):
        super(StopModule, self).__init__()

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

        self.c4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(32)
        self.r4 = nn.ReLU()
        self.p4 = nn.MaxPool2d(2, 2)

        self.c5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(16)
        self.r5 = nn.ReLU()
        self.p5 = nn.MaxPool2d(2, 2)

        self.l1 = nn.Linear(192, 1)
        self.l1.bias.data[0] = -6

    def forward(self, y):
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

        y = self.c4(y)
        y = self.b4(y)
        y = self.r4(y)
        y = self.p4(y)

        y = self.c5(y)
        y = self.b5(y)
        y = self.r5(y)
        y = self.p5(y)
        y = torch.flatten(y)
        y = self.l1(y)

        return torch.sigmoid(y[0])


class LineOutlinerTsa(nn.Module):

    def __init__(self, path=None, patch_ratio=5, tsa_size=3, min_height=32, patch_size=64):
        super(LineOutlinerTsa, self).__init__()
        self.tsa_size = tsa_size
        self.min_height = min_height
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

        self.stop = StopModule()

        if path is not None and os.path.exists(path):
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state)
            self.eval()
        elif path is not None:
            print("\nCould not find path", path, "\n")

    def forward(self,
                img,
                sol_tensor,
                steps,
                reset_threshold=None,
                max_steps=None,
                disturb_sol=True,
                confidence_threshold=None,
                height_disturbance=0.5,
                angle_disturbance=30,
                translate_disturbance=10):

        desired_polygon_steps = torch.stack([sol_tensor.cuda()] + [p.cuda() for p in steps])

        img.cuda()

        # tensor([tsa, channels, width, height])
        input = ((255 / 128) - 1) * torch.ones((1, 3, self.patch_size, self.patch_size)).cuda()

        steps_ran = 0

        sol = {
            "upper_point": sol_tensor[0],
            "base_point": sol_tensor[1],
            "angle": sol_tensor[3][0],
        }

        if disturb_sol:
            x = random.uniform(0, translate_disturbance)
            y = random.uniform(0, translate_disturbance)
            sol["upper_point"][0] += x
            sol["upper_point"][1] += y
            sol["base_point"][0] += x
            sol["base_point"][1] += y
            sol["angle"] += random.uniform(-angle_disturbance, angle_disturbance)

        current_height = torch.dist(sol["upper_point"].clone(), sol["base_point"].clone()).cuda()
        current_height = current_height * (1 if not disturb_sol else (1 + random.uniform(0, height_disturbance)))

        current_angle = sol["angle"].clone().cuda()
        current_base = sol["base_point"].clone().cuda()

        results = []
        tsa_sequence = []

        upper_height_stack = []
        lower_height_stack = []
        baseline_stack = []  # [sol_tensor[1].cuda()]
        angle_stack = []
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

            # Shift input left
            # input = torch.stack([pic for pic in input[1:]] + [patch.squeeze(0)])

            input = torch.stack([pic for pic in input[-self.tsa_size:]] + [patch.squeeze(0)])

            y = input.cuda().unsqueeze(0)
            y = self.tsa(y)
            y = y[:, 1:, :, :, :]
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

            size = input[0, :, :, :].shape[1] / self.patch_ratio

            y[0] = torch.add(y[0], size)

            y[2] = torch.sigmoid(y[2])
            y[3] = torch.sigmoid(y[3])
            # y[2] = torch.add(y[2], size)
            # y[3] = torch.add(y[3], -size)
            # y[4] = torch.add(y[4], size)

            scale_matrix = torch.stack([torch.stack([current_scale, torch.tensor(0.).cuda()]),
                                        torch.stack([torch.tensor(0.).cuda(), current_scale])]).cuda()

            # Finds the next base point
            base_rotation_matrix = torch.stack(
                [torch.stack([torch.cos(torch.deg2rad(current_angle)), -1.0 * torch.sin(torch.deg2rad(current_angle))]),
                 torch.stack(
                     [1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))])]).cuda()

            # Create a vector to represent the new base

            base_point = torch.stack([y[0], y[1]])

            # upper_point = torch.stack([torch.tensor(0, dtype=torch.float32).cuda(), -upper_height])
            # lower_point = torch.stack([torch.tensor(0, dtype=torch.float32).cuda(), lower_height])

            base_point = torch.matmul(base_point, base_rotation_matrix.t())
            base_point = torch.matmul(base_point, scale_matrix)
            current_base = torch.add(base_point, current_base)
            upper_height = torch.mul(y[2], self.patch_size / 2)
            lower_height = torch.mul(y[3], self.patch_size / 2)
            current_angle = torch.add(current_angle, y[4])

            angle_stack.append(current_angle.clone().detach())
            baseline_stack.append(current_base)
            upper_height_stack.append(torch.mul(upper_height, current_scale))
            lower_height_stack.append(torch.mul(lower_height, current_scale))

            # Finds the next base point
            # point_rotation_matrix = torch.stack(
            #    [torch.stack([torch.cos(torch.deg2rad(current_angle)), -1.0 * torch.sin(torch.deg2rad(current_angle))]),
            #     torch.stack(
            #         [1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))])]).cuda()

            # lower_point = torch.matmul(lower_point, point_rotation_matrix.t())
            # lower_point = torch.matmul(lower_point, scale_matrix)
            # lower_point = torch.add(lower_point, current_base)

            # upper_point = torch.matmul(upper_point, point_rotation_matrix.t())
            # upper_point = torch.matmul(upper_point, scale_matrix)
            # upper_point = torch.add(upper_point, current_base)

            # look_ahead_ratio = 3
            # look_ahead_base = current_base.clone().detach()
            # look_ahead_angle = current_angle.clone().detach()
            # look_ahead_height = current_height.clone().detach()
            # extraction_params = []
            # for i in range(look_ahead_ratio):
            #     look_ahead_params = torch.stack([look_ahead_base[0],  # x
            #                                      look_ahead_base[1],  # y
            #                                      torch.mul(torch.deg2rad(look_ahead_angle), -1),  # angle
            #                                      current_height]).unsqueeze(0)
            #     extraction_params.append(look_ahead_params.cuda())
            #     base_point = Point(look_ahead_base[0].item(), look_ahead_base[1].item())
            #     next_point = get_new_point(base_point, look_ahead_angle.item(), look_ahead_height.item())
            #     look_ahead_base = torch.tensor([next_point.x, next_point.y]).cuda()
            # patches = [extract_tensor_patch(img, p, size=self.patch_size) for p in extraction_params]
            # concatenated_patches = torch.cat(patches, dim=2)
            # stop_result = self.stop(concatenated_patches.clone().detach())
            #
            # gt_polygon = to_polygon(torch.stack([s.cuda() for s in steps]))
            # upper_p = Point(upper_point[0].item(), upper_point[1].item())
            # lower_p = Point(lower_point[0].item(), lower_point[1].item())
            # step_line = LineString([upper_p, lower_p])
            # total_length = upper_p.distance(lower_p)
            # intersection = step_line.intersection(gt_polygon)
            # desired_confidence = 1
            # if intersection is not None and isinstance(intersection, LineString):
            #     desired_confidence = 1 - (intersection.length / total_length)
            #
            # confidence_loss = torch.nn.MSELoss()(stop_result, torch.tensor(desired_confidence, dtype=torch.float32).cuda())
            # confidence_loss.backward(retain_graph=True)

            # Decide whether to stop based on last step DICE AFFINITY
            steps_ran += 1

            # if confidence_threshold is not None and stop_result.item() > confidence_threshold:
            #    break

            # Minimum steps to fill TSA
            # if min_run_tsa and steps_ran < self.tsa_size:
            #    continue

            if reset_threshold is not None:
                base_as_point = Point(current_base[0].item(), current_base[1].item())
                # upper_as_point = Point(upper_point[0].item(), upper_point[1].item())
                gt_step = steps[steps_ran]
                # gt_upper = Point(gt_step[0][0].item(), gt_step[0][1].item())
                gt_base_point = Point(gt_step[1][0].item(), gt_step[1][1].item())
                # is_upper_violated = upper_as_point.distance(gt_upper) > reset_threshold

                if base_as_point.distance(gt_base_point) > reset_threshold:
                    break

            if max_steps is None and steps_ran >= len(steps) - 1:
                break

        for i in range(len(baseline_stack)):

            if (i == 0 and len(baseline_stack) == 1) or i == len(baseline_stack) - 1:
                angle_to_next = torch.deg2rad(angle_stack[i])
            else:
                difference = baseline_stack[i + 1] - baseline_stack[i]
                angle_to_next = torch.atan2(difference[1], difference[0])

            point_rotation_matrix = torch.stack(
                [torch.stack([torch.cos(angle_to_next), -1.0 * torch.sin(angle_to_next)]),
                 torch.stack([1.0 * torch.sin(angle_to_next), torch.cos(angle_to_next)])]).cuda()

            upper_point = torch.stack([torch.tensor(0.).cuda(), -upper_height_stack[i]])
            upper_point = torch.matmul(upper_point, point_rotation_matrix.t())
            upper_point = torch.add(upper_point, baseline_stack[i].clone().detach())

            lower_point = torch.stack([torch.tensor(0.).cuda(), lower_height_stack[i]])
            lower_point = torch.matmul(lower_point, point_rotation_matrix.t())
            lower_point = torch.add(lower_point, baseline_stack[i].clone().detach())

            results.append(torch.stack([
                upper_point,
                baseline_stack[i],
                lower_point,
                torch.tensor([0.0, 0.0]).cuda()
            ]))
            # point_rotation_matrix = torch.stack(
            #    [torch.stack([torch.cos(torch.deg2rad(current_angle)), -1.0 * torch.sin(torch.deg2rad(current_angle))]),
            #     torch.stack(
            #         [1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))])]).cuda()

        if len(results) == 0:
            return torch.zeros((0, 5, 2)).cuda(), 0, []
        else:
            return torch.stack(results), steps_ran, tsa_sequence
