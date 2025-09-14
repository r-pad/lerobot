"""
HACK: COPIED OVER/simplified FROM

https://github.com/r-pad/lfd3d/blob/main/src/lfd3d/models/articubot.py

Which in turn was copied from the articubot repo -> model_invariant.py
"""

# NOTE:
# Trying to implement PointNet++
# Borrowed from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import random
from collections import defaultdict
from time import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers import get_cosine_schedule_with_warmup
from torch import nn, optim
import argparse

def timeit(tag, t):
    print(f"{tag}: {time() - t}s")
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz_, npoint, keep_gripper_in_fps=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if keep_gripper_in_fps:  # NOTE: assuming there are 4 gripper points
        xyz = xyz_[:, :-4, :]
        npoint = npoint - 4
    else:
        xyz = xyz_

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = farthest * 0  # set to 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    if keep_gripper_in_fps:
        gripper_indices = torch.Tensor([N, N + 1, N + 2, N + 3]).long().to(device)
        gripper_indices = gripper_indices.unsqueeze(0).repeat(B, 1)
        centroids = torch.cat([centroids, gripper_indices], dim=1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(
        self,
        npoint,
        radius_list,
        nsample_list,
        in_channel,
        mlp_list,
        keep_gripper_in_fps=False,
    ):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.keep_gripper_in_fps = keep_gripper_in_fps
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(
            xyz, farthest_point_sample(xyz, S, self.keep_gripper_in_fps)
        )
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=0,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]]
        )
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 16) (B, 1024, 16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 512, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes


class PointNet2_super(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, use_text_embedding=False, use_dual_head=False):
        super(PointNet2_super, self).__init__()

        self.use_text_embedding = use_text_embedding
        self.use_dual_head = use_dual_head
        self.encoded_text_dim = 128       
        if self.use_text_embedding:
            self.text_encoder = nn.Linear(
                1152, self.encoded_text_dim
            )  # SIGLIP input dim
            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # [B, 256] -> [B, 2048]
            )
            # Init as gamma=0 and beta=1
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(1024), torch.zeros(1024)])
            )
        
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],
            [16, 32],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64,
            [0.4, 0.8],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16,
            [0.8, 1.6],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)

        # Dual head architecture
        if self.use_dual_head:
            # Human prediction head
            self.human_conv = nn.Conv1d(128, 128, 1)
            self.human_bn = nn.BatchNorm1d(128)
            self.human_head = nn.Conv1d(128, num_classes, 1)

            # Robot prediction head
            self.robot_conv = nn.Conv1d(128, 128, 1)
            self.robot_bn = nn.BatchNorm1d(128)
            self.robot_head = nn.Conv1d(128, num_classes, 1)
        else:
            # Single head
            self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, text_embedding=None, data_source=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        # Apply FiLM conditioning at bottleneck
        if self.use_text_embedding:
            encoded_text = self.text_encoder(text_embedding)  # [B, 128]
            film_params = self.film_predictor(encoded_text)  # [B, 1024 * 2]
            gamma, beta = film_params.chunk(2, dim=1)  # [B, 1024] each
            gamma = gamma.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            beta = beta.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            l6_points = gamma * l6_points + beta  # FiLM modulation: [B, 1024, 16]

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        # Shared backbone features
        backbone_features = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)

        if self.use_dual_head:
            assert data_source is not None
            # Dual head prediction - Compute both and mask out.
            human_features = F.relu(self.human_bn(self.human_conv(backbone_features)))
            robot_features = F.relu(self.robot_bn(self.robot_conv(backbone_features)))

            human_output = self.human_head(human_features)  # (B, num_classes, N)
            robot_output = self.robot_head(robot_features)  # (B, num_classes, N)

            # Create final output by selecting appropriate head for each batch item
            human_mask = torch.tensor([ds == "human" for ds in data_source],
                                    device=human_output.device, dtype=torch.bool)
            human_mask = human_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1] for broadcasting
            x = torch.where(human_mask, human_output, robot_output)
        else:
            x = self.conv2(backbone_features)  # (B, num_classes, N)

        x = x.permute(0, 2, 1)  # (B, N, num_classes)
        return x

class PointNet2_superplus(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_superplus, self).__init__()
        self.sa0 = PointNetSetAbstractionMsg(
            npoint=2048,
            radius_list=[0.0125, 0.025],
            nsample_list=[16, 32],
            in_channel=0,
            mlp_list=[[32, 32, 64], [64, 64, 128]],
        )
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=64 + 128,
            mlp_list=[[64, 64, 128], [128, 196, 256]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=128 + 256,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32], 256 + 256, [[256, 384, 512], [256, 384, 512]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4], [16, 32], 512 + 512, [[256, 384, 512], [256, 384, 512]]
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64, [0.4, 0.8], [16, 32], 512 + 512, [[512, 512, 512], [512, 512, 512]]
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16, [0.8, 1.6], [16, 32], 512 + 512, [[512, 512, 512], [512, 512, 512]]
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 512, [512, 512, 512])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 512, [512, 384, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 256 + 256, [256, 256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 256 + 128, [256, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 128 + 64, [128, 128, 128])
        self.fp0 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l01_xyz, l01_points = self.sa0(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l01_xyz, l01_points)  # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l01_points = self.fp1(
            l01_xyz, l1_xyz, l01_points, l1_points
        )  # (B, 128, num_point)
        l0_points = self.fp0(l0_xyz, l01_xyz, None, l01_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes


class ArticubotNetwork(nn.Module):
    """
    Modified version of PointNet2_super to work with this codebase
    """

    def __init__(self, model_cfg):
        super(ArticubotNetwork, self).__init__()

        num_classes = model_cfg.num_classes
        input_channel = model_cfg.in_channels
        keep_gripper_in_fps = model_cfg.keep_gripper_in_fps

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],
            [16, 32],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64,
            [0.4, 0.8],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16,
            [0.8, 1.6],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes


class ArticubotSmallNetwork(nn.Module):
    def __init__(self, model_cfg):
        super(ArticubotSmallNetwork, self).__init__()

        num_classes = model_cfg.num_classes
        input_channel = model_cfg.in_channels
        keep_gripper_in_fps = model_cfg.keep_gripper_in_fps

        # Reduced SA layers: Only sa1, sa2, sa3 with smaller MLPs
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,  # Reduced from 1024
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16], [32, 32]],  # Smaller MLP
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,  # Reduced from 512
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=48,  # Adjusted based on sa1 output
            mlp_list=[[32, 32, 64], [64, 64]],  # Smaller MLP
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=128,  # Reduced from 256
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=128,  # Adjusted based on sa2 output
            mlp_list=[[64, 64, 128], [128, 128]],  # Smaller MLP
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # Reduced FP layers: Only fp3, fp2, fp1
        self.fp3 = PointNetFeaturePropagation(
            128 + 128 + 128, [128, 128]
        )  # Adjusted input channels
        self.fp2 = PointNetFeaturePropagation(
            48 + 128, [128, 64]
        )  # Adjusted input channels
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])

        self.conv1 = nn.Conv1d(64, 64, 1)  # Reduced channels
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = nn.functional.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)  # [B, num_classes, N]
        return x.permute(0, 2, 1)  # [B, N, num_classes]

def get_weighted_displacement(scene_pcd, outputs):
    """
    Extract weighted displacement from network output
    """
    batch_size, num_points, _ = outputs.shape
    scene_pcd = scene_pcd[:, :, None, :3]
    
    weights = outputs[:, :, -1]  # B, N
    outputs = outputs[:, :, :-1]  # B, N, 12

    # softmax the weights
    weights = torch.nn.functional.softmax(weights, dim=1)

    # sum the displacement of the predicted gripper point cloud according to the weights
    pred_points = weights[:, :, None, None] * (
        scene_pcd + outputs.reshape(batch_size, num_points, 4, 3)
    )
    pred_points = pred_points.sum(dim=1)
    return pred_points


def sample_from_gmm(scene_pcd, outputs):
    batch_size, num_points, _ = outputs.shape
    scene_pcd = scene_pcd[:, :, None, :3]

    weights = outputs[:, :, -1]  # B, N
    # Extract displacement predictions
    outputs = outputs[:, :, :-1].reshape(
        batch_size, num_points, 4, 3
    )  # B, N, 4, 3

    # softmax the weights
    weights = torch.nn.functional.softmax(weights, dim=1)
    # Sample point indices based on weights for each batch element
    sampled_indices = torch.multinomial(weights, num_samples=1)  # B, 1
    batch_indices = torch.arange(batch_size, device=outputs.device).unsqueeze(
        1
    )  # B, 1

    # Get the Gaussian means: scene_point + displacement
    # Broadcasting will expand scene_points from [B, N, 1, 3] to [B, N, 4, 3]
    means = scene_pcd + outputs  # B, N, 4, 3

    sampled_means = means[batch_indices, sampled_indices].squeeze(1)  # B, 4, 3

    # NOTE: We can sample from these gaussians as well, but just using the mean for now.
    # noise = torch.randn_like(sampled_means) * (self.fixed_variance**0.5)
    # pred_points = sampled_means + noise

    pred_points = sampled_means
    return pred_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model from WandB and run inference")
    parser.add_argument('--run_id', type=str, required=True, help='WandB run ID (e.g., abc123)')
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    # Initialize WandB API and download artifact
    artifact_dir = "wandb"
    checkpoint_reference = f"r-pad/lfd3d/best_rmse_model-{args.run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference, type="model")
    ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    ckpt = torch.load(ckpt_file)
    # Remove the "network." prefix, since we're not using Lightning here.
    state_dict = {k.replace("network.",""): v for k, v in ckpt["state_dict"].items()}

    model = PointNet2_super(num_classes=13)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    # Run random inference
    input_data = torch.rand(1, 3, 2000).to(args.device)
    with torch.no_grad():
        outputs = model(input_data)  # Output shape: [1, 2000, 13]

    weighted_displacement = get_weighted_displacement(outputs)
    print(f"Weighted displacement shape: {weighted_displacement.shape}") # [1, 4, 3]
