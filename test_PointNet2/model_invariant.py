# NOTE:
# Trying to implement PointNet++
# Borrowed from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
try:
    from pointnet2_ops import pointnet2_utils
    use_pointnet2_ops = True
except ImportError:
    use_pointnet2_ops = False
    
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)

        # print(src_module)
        tgt_module = func(src_module)
        
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    # replace_submodules(
    #     root_module=root_module,
    #     predicate=lambda x: isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.BatchNorm1d),
    #     func=lambda x: nn.GroupNorm(
    #         num_groups=x.num_features//features_per_group,
    #         num_channels=x.num_features)
    # )
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d) or isinstance(x, nn.BatchNorm1d),
        func=lambda x: nn.GroupNorm(
            num_groups=1,
            num_channels=x.num_features)
    )
    return root_module

def replace_bn_with_in(root_module: nn.Module) -> nn.Module:
    """
    Replace all BatchNorm1d and BatchNorm2d layers with InstanceNorm1d/2d,
    with track_running_stats=False.
    """
    def replace_fn(module):
        if isinstance(module, nn.BatchNorm1d):
            return nn.InstanceNorm1d(
                num_features=module.num_features,
                eps=module.eps,
                affine=module.affine,
                track_running_stats=False
            )
        elif isinstance(module, nn.BatchNorm2d):
            return nn.InstanceNorm2d(
                num_features=module.num_features,
                eps=module.eps,
                affine=module.affine,
                track_running_stats=False
            )
        return module

    def recursive_replace(module):
        for name, child in module.named_children():
            new_child = replace_fn(child)
            if new_child is not child:
                setattr(module, name, new_child)
            else:
                recursive_replace(child)

    recursive_replace(root_module)
    return root_module

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
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
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
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
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
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
    if keep_gripper_in_fps: ### NOTE: assuming there are 4 gripper points
        xyz = xyz_[:, :-4, :]
        npoint = npoint - 4
    else:
        xyz = xyz_
    
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = farthest * 0 # set to 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    if keep_gripper_in_fps:
        gripper_indices = torch.Tensor([N, N+1, N+2, N+3]).long().to(device)
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
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
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
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
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

class FiLM(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(embedding_dim, feature_dim)
        self.shift = nn.Linear(embedding_dim, feature_dim)

    def forward(self, x, embedding):
        """
        Input:
            x: input points data, [B, D, N]
            embedding: embedding data, [B, E]
        Return:
            transformed points data, [B, D, N]
        """
        num_shape = len(x.shape)
        if num_shape == 3:
            gamma = self.scale(embedding).unsqueeze(-1)  # [B, D, 1]
            beta = self.shift(embedding).unsqueeze(-1)  # [B, D, 1]
        elif num_shape == 4:
            gamma = self.scale(embedding).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1]
            beta = self.shift(embedding).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1]
        return x * gamma + beta

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
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, keep_gripper_in_fps=False, embedding_dim=False,
                 layernorm=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.keep_gripper_in_fps = keep_gripper_in_fps
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.layernorm = layernorm
        if not layernorm:
            self.bn_blocks = nn.ModuleList()
        else:
            self.ln_blocks = nn.ModuleList()

        self.film_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            if not layernorm:
                bns = nn.ModuleList()
            else:
                lns = nn.ModuleList()
            last_channel = in_channel + 3
            if embedding_dim:
                self.film_blocks.append(FiLM(embedding_dim, last_channel))
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if not layernorm:
                    bns.append(nn.BatchNorm2d(out_channel))
                else:
                    lns.append(nn.LayerNorm(out_channel))  # LayerNorm expects [B, *, C]
                last_channel = out_channel
            self.conv_blocks.append(convs)
            if not layernorm:
                self.bn_blocks.append(bns)
            else:
                self.ln_blocks.append(lns)

    def forward(self, xyz, points, embedding=None):
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
        if not use_pointnet2_ops:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, S, self.keep_gripper_in_fps))
        else:
            new_xyz = fps(xyz, S)
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
            ### NOTE: apply film here so hopefully things can be mapped to the same space between different tasks so batchnorm works correctly
            # import pdb; pdb.set_trace()
            grouped_points = self.film_blocks[i](grouped_points, embedding) if embedding is not None else grouped_points
            for j in range(len(self.conv_blocks[i])):
                if not self.layernorm:
                    conv = self.conv_blocks[i][j]
                    bn = self.bn_blocks[i][j]
                    grouped_points =  F.relu(bn(conv(grouped_points)))
                else:
                    conv = self.conv_blocks[i][j]
                    ln = self.ln_blocks[i][j]

                    x = conv(grouped_points)  # [B, D', K, S]
                    x = x.permute(0, 2, 3, 1)  # → [B, K, S, D']
                    x = ln(x)
                    x = x.permute(0, 3, 1, 2)  # → [B, D', K, S]
                    grouped_points = F.relu(x)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, embedding_dim=None, layernorm=False):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        if not layernorm:
            self.mlp_bns = nn.ModuleList()
        else:
            self.mlp_lns = nn.ModuleList()
        last_channel = in_channel
        self.film = FiLM(embedding_dim, last_channel) if embedding_dim is not None else None
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if not layernorm:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            else:
                self.mlp_lns.append(nn.LayerNorm(out_channel))
            last_channel = out_channel
            
        self.layernorm = layernorm

    def forward(self, xyz1, xyz2, points1, points2, embedding=None):
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
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # import pdb; pdb.set_trace()
        if embedding is not None:
            new_points = self.film(new_points, embedding)
        for i, conv in enumerate(self.mlp_convs):
            if not self.layernorm:
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
            else:
                ln = self.mlp_lns[i]
                x = conv(new_points)              # [B, C, N]
                x = x.permute(0, 2, 1)            # [B, N, C]
                x = ln(x)
                new_points = F.relu(x.permute(0, 2, 1))  # [B, C, N]
        return new_points

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=0, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 16) (B, 1024, 16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 512, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes
    

class PointNet2_small2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_small2, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=0, mlp_list=[[16, 16, 16], [32, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=48, mlp_list=[[64, 64, 64], [64, 96, 64]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128, [[128, 196, 128], [128, 196, 128]])

        self.fp3 = PointNetFeaturePropagation(64+64+128+128, [128, 128])
        self.fp2 = PointNetFeaturePropagation(16+32+128, [64, 64])
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 512) (B, 96, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 128) (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 32) (B, 512, 32)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 128)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 512)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes: outputing logtis

class PointNet2_super(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, embedding_dim=None):
        super(PointNet2_super, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps)
        if embedding_dim is not None:
            self.film = FiLM(embedding_dim, 1024)
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, embedding=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        # add film
        if embedding is not None:
            l6_points = self.film(l6_points, embedding) # (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes
    
class PointNet2_super_multitask(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, embedding_dim=None,
                 first_sa_point=2048, fp_to_full=False, replace_bn_w_gn=False, replace_bn_w_in=False, film_in_sa_and_fp=False, 
                 embedding_as_input=False,
                 replace_bn_w_ln=False,):
                #  first_sa_point=1024, fp_to_full=True, replace_bn_w_gn=False, replace_bn_w_in=True):
        super(PointNet2_super_multitask, self).__init__()
        # self.sa0 = PointNetSetAbstractionMsg(npoint=2048, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=96, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        if embedding_as_input:
            in_channel = input_channel - 3 + embedding_dim
        else:
            in_channel = input_channel - 3
        self.embedding_as_input = embedding_as_input
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=first_sa_point, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=in_channel, mlp_list=[[16, 16, 32], [32, 32, 64]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        if embedding_dim is not None:
            self.film = FiLM(embedding_dim, 1024)
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        if fp_to_full:
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp_to_full = fp_to_full
        self.film_in_sa_and_fp = film_in_sa_and_fp
        
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # nn.Dropout(self.model_config.get("score_dropout", 0.5)),  # 0.5 in original code
            nn.Conv1d(128, 1, 1, padding=0)
        )
                
        ### this will be the displacement for each point
        self.four_point_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # nn.Dropout(self.model_config.get("displacement_dropout", 0.3)),  # 0.5 in original code
            nn.Conv1d(128, 12, 1, padding=0)
        )

        if replace_bn_w_gn:
            print("replacing all batchnorm layers to be group norm layers!")
            replace_bn_with_gn(self)
        if replace_bn_w_in:
            print("replacing all batchnorm layers to be instance norm layers!")
            replace_bn_with_in(self)

    def forward(self, xyz, embedding=None, build_grasp=False, articubot_format=False):
        # assert embedding is not None
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if self.embedding_as_input:
            # import pdb; pdb.set_trace()
            input_embedding = embedding.unsqueeze(2).repeat(1, 1, l0_xyz.shape[2])
            xyz = torch.cat([xyz, input_embedding], dim=1)
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :], embedding=embedding if self.film_in_sa_and_fp else None)
        else:
            # l0_xyz, l0_points = self.sa0(l0_xyz, None)
            # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)

            l1_xyz, l1_points = self.sa1(l0_xyz, None, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 1024) (B, 96, 1024)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 16) (B, 1024, 16)

        # add film
        if embedding is not None:
            print("using language embedding in film")
            l6_points = self.film(l6_points, embedding) # (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 128, 1024)
        if self.fp_to_full:
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 128, num_point)
            pred_points = l0_xyz #### 2048 points
            feature = l0_points
        else:
            pred_points = l1_xyz #### 2048 points
            feature = l1_points
            
        binary_seg_head = self.binary_seg_head(feature)
        four_point_head_offset = self.four_point_head(feature).permute(0, 2, 1)
        
        pred_scores = binary_seg_head.permute(0, 2, 1)
        pred_points = pred_points.permute(0, 2, 1) # B x N x 3
        pred_offsets = four_point_head_offset.view(four_point_head_offset.shape[0], four_point_head_offset.shape[1], 4, 3)  # B x N x 4 x 3

        if build_grasp:
            pred_4_points = pred_points.unsqueeze(2).repeat(1, 1, 4, 1) + four_point_head_offset.reshape(four_point_head_offset.shape[0], four_point_head_offset.shape[1], 4, 3) # B x N x 4 x 3
            pred_grasps_cam, offset = self.build_6d_grasp_from_four_points(pred_4_points, articubot_format=articubot_format)  # B x N x 4 x 4
        else:
            pred_grasps_cam, offset = None, None
    
        pred = dict(
            pred_scores = pred_scores,
            pred_points =pred_points,
            pred_offsets=pred_offsets,  
            pred_grasps_cam= pred_grasps_cam,  # B x N x 4 x 4
            offset_pred=offset
        )
        
        return pred
        
    def build_6d_grasp_from_four_points(self, four_point_head, gripper_depth = 0.105, articubot_format=False):
        B, N, _, _ = four_point_head.shape
        
        grasp_t = four_point_head[:, :, 0].unsqueeze(3)  # B x N x 3 x 1
        
        
        approach_direction = four_point_head[:, :, -1] - four_point_head[:, :, 0]  # B x N x 3
        baseline_direction = four_point_head[:, :, 2] - four_point_head[:, :, 1]  # B x N x 3
        
        # baseline_direction_normed = F.normalize(baseline_direction, p=2, dim=2)  # B x N x 3
        # dot_product = torch.sum(approach_direction * baseline_direction_normed, dim=2, keepdim=True)  # B x N x 1
        # projection = dot_product * baseline_direction_normed  # B x N x 3
        # approach_direction_orthog = F.normalize(approach_direction - projection, p=2, dim=2)  # B x N x 3
        # grasp_R = torch.stack([baseline_direction_normed, torch.cross(approach_direction_orthog, baseline_direction_normed),approach_direction_orthog], dim=3)  # B x N x 3 x 3
        
        approach_direction_normed = F.normalize(approach_direction, p=2, dim=2)  # B x N x 3
        dot_product = torch.sum(baseline_direction * approach_direction_normed, dim=2, keepdim=True)  # B x N x 1
        projection = dot_product * approach_direction_normed  # B x N x 3
        baseline_direction_orthog = F.normalize(baseline_direction - projection, p=2, dim=2)  # B x N x 3
        grasp_R = torch.stack([baseline_direction_orthog, torch.cross(approach_direction_normed, baseline_direction_orthog),approach_direction_normed], dim=3)  # B x N x 3 x 3
        
        if articubot_format:
            from termcolor import cprint
            cprint("Using articubot format for 6d grasp!", "yellow")
            eef_position = four_point_head[:, :, -1] # B x N x 3
            hand_position = eef_position - gripper_depth * approach_direction_normed # B x N x 3
            grasp_t = hand_position.unsqueeze(3)  # B x N x 3 x 1
        
        
        ones = torch.ones((B, N, 1, 1), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 1
        zeros = torch.zeros((B, N, 1, 3), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4
        
        offset = torch.norm(four_point_head[:, :, 2] - four_point_head[:, :, 1], dim=-1, keepdim=True)  # B x N x 1
        
        return grasps, offset
    

        
class PointNet2_superplus(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_superplus, self).__init__()
        self.sa0 = PointNetSetAbstractionMsg(npoint=2048, radius_list=[0.0125, 0.025], nsample_list=[16, 32], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=64+128, mlp_list=[[64, 64, 128], [128, 196, 256]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=128+256, mlp_list=[[128, 196, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 256+256, [[256, 384, 512], [256, 384, 512]])
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 512+512, [[256, 384, 512], [256, 384, 512]])
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]])
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]])
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512, 512])
        self.fp5 = PointNetFeaturePropagation(512+512+512, [512, 512, 512])
        self.fp4 = PointNetFeaturePropagation(512+512+512, [512, 384, 256])
        self.fp3 = PointNetFeaturePropagation(256+256+256, [256, 256, 256])
        self.fp2 = PointNetFeaturePropagation(256+256+128, [256, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128+128+64, [128, 128, 128])
        self.fp0 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l01_xyz, l01_points = self.sa0(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l01_xyz, l01_points) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l01_points = self.fp1(l01_xyz, l1_xyz, l01_points, l1_points) # (B, 128, num_point)
        l0_points = self.fp0(l0_xyz, l01_xyz, None, l01_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes
    
    
# from PointNeXt.openpoints.models.backbone.pointnext import InvResMLP
class PointNet2_super_next_multitask(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, embedding_dim=None,
                 first_sa_point=2048, fp_to_full=False, replace_bn_w_gn=False, replace_bn_w_in=False, film_in_sa_and_fp=False, 
                 embedding_as_input=False,
                 replace_bn_w_ln=False,):
                #  first_sa_point=1024, fp_to_full=True, replace_bn_w_gn=False, replace_bn_w_in=True):
        super(PointNet2_super_next_multitask, self).__init__()
        # self.sa0 = PointNetSetAbstractionMsg(npoint=2048, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=96, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        if embedding_as_input:
            in_channel = input_channel - 3 + embedding_dim
        else:
            in_channel = input_channel - 3
        self.embedding_as_input = embedding_as_input
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=first_sa_point, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=in_channel, mlp_list=[[16, 16, 32], [32, 32, 64]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)

        group_args = {'NAME': 'ballquery'}
        group_args['radius'] = 0.05
        group_args['nsample'] = 32
        aggr_args = {"feature_type": 'dp_fj', "reduction": 'max'}
        # norm_args = {"norm": 'bn'}
        norm_args = {"norm": 'ln'}
        conv_args = {"order": "conv-norm-act"}
        act_args = {"act": 'relu'}
        expansion = 4
        self.invresmlp_1 = InvResMLP(32 + 64,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
                                             
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 0.1
        group_args['nsample'] = 32
        self.invresmlp_2 = InvResMLP(128 + 128,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
                                             
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 0.2
        group_args['nsample'] = 32
        self.invresmlp_3 = InvResMLP(256 + 256,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
                                             
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 0.4
        group_args['nsample'] = 32
        self.invresmlp_4 = InvResMLP(512 + 512,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
        
        
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 0.8
        group_args['nsample'] = 32
        self.invresmlp_5 = InvResMLP(512 + 512,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
                                             
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 1.6
        group_args['nsample'] = 32
        self.invresmlp_6 = InvResMLP(512 + 512,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
                                             
        if embedding_dim is not None:
            self.film = FiLM(embedding_dim, 1024)
            
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        if fp_to_full:
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp_to_full = fp_to_full
        self.film_in_sa_and_fp = film_in_sa_and_fp
        
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # nn.Dropout(self.model_config.get("score_dropout", 0.5)),  # 0.5 in original code
            nn.Conv1d(128, 1, 1, padding=0)
        )
                
        ### this will be the displacement for each point
        self.four_point_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # nn.Dropout(self.model_config.get("displacement_dropout", 0.3)),  # 0.5 in original code
            nn.Conv1d(128, 12, 1, padding=0)
        )

        if replace_bn_w_gn:
            print("replacing all batchnorm layers to be group norm layers!")
            replace_bn_with_gn(self)
        if replace_bn_w_in:
            print("replacing all batchnorm layers to be instance norm layers!")
            replace_bn_with_in(self)

    def forward(self, xyz, embedding=None, build_grasp=False, articubot_format=False):
        assert embedding is not None
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if self.embedding_as_input:
            # import pdb; pdb.set_trace()
            input_embedding = embedding.unsqueeze(2).repeat(1, 1, l0_xyz.shape[2])
            xyz = torch.cat([xyz, input_embedding], dim=1)
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :], embedding=embedding if self.film_in_sa_and_fp else None)
        else:
            # l0_xyz, l0_points = self.sa0(l0_xyz, None)
            # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)

            l1_xyz, l1_points = self.sa1(l0_xyz, None, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 1024) (B, 96, 1024)

        # import pdb; pdb.set_trace()
        l1_xyz, l1_points = self.invresmlp_1([l1_xyz.permute(0, 2, 1), l1_points])     
        l1_xyz = l1_xyz.permute(0, 2, 1)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 512) (B, 256, 512)
        l2_xyz, l2_points = self.invresmlp_2([l2_xyz.permute(0, 2, 1), l2_points])        
        l2_xyz = l2_xyz.permute(0, 2, 1)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 256) (B, 512, 256)
        l3_xyz, l3_points = self.invresmlp_3([l3_xyz.permute(0, 2, 1), l3_points])        
        l3_xyz = l3_xyz.permute(0, 2, 1)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 128) (B, 1024, 16)
        l4_xyz, l4_points = self.invresmlp_4([l4_xyz.permute(0, 2, 1), l4_points])        
        l4_xyz = l4_xyz.permute(0, 2, 1)
        
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 64) (B , 1024, 64)
        l5_xyz, l5_points = self.invresmlp_5([l5_xyz.permute(0, 2, 1), l5_points])        
        l5_xyz = l5_xyz.permute(0, 2, 1)
        
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 16) (B, 1024, 16)
        l6_xyz, l6_points = self.invresmlp_6([l6_xyz.permute(0, 2, 1), l6_points])        
        l6_xyz = l6_xyz.permute(0, 2, 1)

        # add film
        if embedding is not None:
            # print("using language embedding in film")
            l6_points = self.film(l6_points, embedding) # (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 128, 1024)
        if self.fp_to_full:
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 128, num_point)
            pred_points = l0_xyz #### 2048 points
            feature = l0_points
        else:
            pred_points = l1_xyz #### 2048 points
            feature = l1_points
            
        binary_seg_head = self.binary_seg_head(feature)
        four_point_head_offset = self.four_point_head(feature).permute(0, 2, 1)
        
        pred_scores = binary_seg_head.permute(0, 2, 1)
        pred_points = pred_points.permute(0, 2, 1) # B x N x 3
        pred_offsets = four_point_head_offset.view(four_point_head_offset.shape[0], four_point_head_offset.shape[1], 4, 3)  # B x N x 4 x 3

        if build_grasp:
            pred_4_points = pred_points.unsqueeze(2).repeat(1, 1, 4, 1) + four_point_head_offset.reshape(four_point_head_offset.shape[0], four_point_head_offset.shape[1], 4, 3) # B x N x 4 x 3
            pred_grasps_cam, offset = self.build_6d_grasp_from_four_points(pred_4_points, articubot_format=articubot_format)  # B x N x 4 x 4
        else:
            pred_grasps_cam, offset = None, None
    
        pred = dict(
            pred_scores = pred_scores,
            pred_points =pred_points,
            pred_offsets=pred_offsets,  
            pred_grasps_cam= pred_grasps_cam,  # B x N x 4 x 4
            offset_pred=offset
        )
        
        return pred
        
    def build_6d_grasp_from_four_points(self, four_point_head, gripper_depth = 0.105, articubot_format=False):
        B, N, _, _ = four_point_head.shape
        
        grasp_t = four_point_head[:, :, 0].unsqueeze(3)  # B x N x 3 x 1
        
        
        approach_direction = four_point_head[:, :, -1] - four_point_head[:, :, 0]  # B x N x 3
        baseline_direction = four_point_head[:, :, 2] - four_point_head[:, :, 1]  # B x N x 3
        
        # baseline_direction_normed = F.normalize(baseline_direction, p=2, dim=2)  # B x N x 3
        # dot_product = torch.sum(approach_direction * baseline_direction_normed, dim=2, keepdim=True)  # B x N x 1
        # projection = dot_product * baseline_direction_normed  # B x N x 3
        # approach_direction_orthog = F.normalize(approach_direction - projection, p=2, dim=2)  # B x N x 3
        # grasp_R = torch.stack([baseline_direction_normed, torch.cross(approach_direction_orthog, baseline_direction_normed),approach_direction_orthog], dim=3)  # B x N x 3 x 3
        
        approach_direction_normed = F.normalize(approach_direction, p=2, dim=2)  # B x N x 3
        dot_product = torch.sum(baseline_direction * approach_direction_normed, dim=2, keepdim=True)  # B x N x 1
        projection = dot_product * approach_direction_normed  # B x N x 3
        baseline_direction_orthog = F.normalize(baseline_direction - projection, p=2, dim=2)  # B x N x 3
        grasp_R = torch.stack([baseline_direction_orthog, torch.cross(approach_direction_normed, baseline_direction_orthog),approach_direction_normed], dim=3)  # B x N x 3 x 3
        
        if articubot_format:
            from termcolor import cprint
            cprint("Using articubot format for 6d grasp!", "yellow")
            eef_position = four_point_head[:, :, -1] # B x N x 3
            hand_position = eef_position - gripper_depth * approach_direction_normed # B x N x 3
            grasp_t = hand_position.unsqueeze(3)  # B x N x 3 x 1
        
        
        ones = torch.ones((B, N, 1, 1), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 1
        zeros = torch.zeros((B, N, 1, 3), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4
        
        offset = torch.norm(four_point_head[:, :, 2] - four_point_head[:, :, 1], dim=-1, keepdim=True)  # B x N x 1
        
        return grasps, offset
    
class PointNet2_super_next_fp_multitask(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, embedding_dim=None,
                 first_sa_point=2048, fp_to_full=False, replace_bn_w_gn=False, replace_bn_w_in=False, film_in_sa_and_fp=False, 
                 embedding_as_input=False,
                 replace_bn_w_ln=False,):
                #  first_sa_point=1024, fp_to_full=True, replace_bn_w_gn=False, replace_bn_w_in=True):
        super(PointNet2_super_next_fp_multitask, self).__init__()
        # self.sa0 = PointNetSetAbstractionMsg(npoint=2048, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=96, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps)
        if embedding_as_input:
            in_channel = input_channel - 3 + embedding_dim
        else:
            in_channel = input_channel - 3
        self.embedding_as_input = embedding_as_input
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=first_sa_point, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=in_channel, mlp_list=[[16, 16, 32], [32, 32, 64]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)

        group_args = {'NAME': 'ballquery'}
        group_args['radius'] = 0.05
        group_args['nsample'] = 32
        aggr_args = {"feature_type": 'dp_fj', "reduction": 'max'}
        # norm_args = {"norm": 'bn'}
        norm_args = {"norm": 'ln'}
        conv_args = {"order": "conv-norm-act"}
        act_args = {"act": 'relu'}
        expansion = 2
        
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)                                     
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)   
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], 
                                             keep_gripper_in_fps=keep_gripper_in_fps, embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
                                             
        if embedding_dim is not None:
            self.film = FiLM(embedding_dim, 1024)
            
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 1.6
        self.invresmlp_6 = InvResMLP(512,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        group_args['radius'] = 0.8
        self.invresmlp_5 = InvResMLP(512,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.invresmlp_4 = InvResMLP(256,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.invresmlp_3 = InvResMLP(256,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.invresmlp_2 = InvResMLP(128,
                                aggr_args=aggr_args,
                                norm_args=norm_args, act_args=act_args, group_args=group_args,
                                conv_args=conv_args, expansion=expansion,
                                use_res=True)
        if fp_to_full:
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], embedding_dim=embedding_dim if film_in_sa_and_fp else None, layernorm=replace_bn_w_ln)
        self.fp_to_full = fp_to_full
        self.film_in_sa_and_fp = film_in_sa_and_fp
        
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # nn.Dropout(self.model_config.get("score_dropout", 0.5)),  # 0.5 in original code
            nn.Conv1d(128, 1, 1, padding=0)
        )
                
        ### this will be the displacement for each point
        self.four_point_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, padding=0),
            # nn.BatchNorm1d(128),
            nn.GroupNorm(32, 128),
            nn.ReLU(),
            # nn.Dropout(self.model_config.get("displacement_dropout", 0.3)),  # 0.5 in original code
            nn.Conv1d(128, 12, 1, padding=0)
        )

        if replace_bn_w_gn:
            print("replacing all batchnorm layers to be group norm layers!")
            replace_bn_with_gn(self)
        if replace_bn_w_in:
            print("replacing all batchnorm layers to be instance norm layers!")
            replace_bn_with_in(self)

    def forward(self, xyz, embedding=None, build_grasp=False, articubot_format=False):
        assert embedding is not None
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if self.embedding_as_input:
            # import pdb; pdb.set_trace()
            input_embedding = embedding.unsqueeze(2).repeat(1, 1, l0_xyz.shape[2])
            xyz = torch.cat([xyz, input_embedding], dim=1)
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :], embedding=embedding if self.film_in_sa_and_fp else None)
        else:
            # l0_xyz, l0_points = self.sa0(l0_xyz, None)
            # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)

            l1_xyz, l1_points = self.sa1(l0_xyz, None, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 1024) (B, 96, 1024)

        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 512) (B, 256, 512)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 256) (B, 512, 256)
        
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 128) (B, 1024, 16)
        
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 64) (B , 1024, 64)
        
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 3, 16) (B, 1024, 16)

        # add film
        if embedding is not None:
            # print("using language embedding in film")
            l6_points = self.film(l6_points, embedding) # (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 512, 64)
        l5_xyz, l5_points = self.invresmlp_6([l5_xyz.permute(0, 2, 1).contiguous(), l5_points.contiguous()])        
        l5_xyz = l5_xyz.permute(0, 2, 1)

        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 512, 128)
        l4_xyz, l4_points = self.invresmlp_5([l4_xyz.permute(0, 2, 1).contiguous(), l4_points.contiguous()])        
        l4_xyz = l4_xyz.permute(0, 2, 1)
        
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 256, 256)
        l3_xyz, l3_points = self.invresmlp_4([l3_xyz.permute(0, 2, 1).contiguous(), l3_points.contiguous()])        
        l3_xyz = l3_xyz.permute(0, 2, 1)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 256, 512)
        l2_xyz, l2_points = self.invresmlp_3([l2_xyz.permute(0, 2, 1).contiguous(), l2_points.contiguous()])        
        l2_xyz = l2_xyz.permute(0, 2, 1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 128, 1024)
        l1_xyz, l1_points = self.invresmlp_2([l1_xyz.permute(0, 2, 1).contiguous(), l1_points.contiguous()])        
        l1_xyz = l1_xyz.permute(0, 2, 1)

        
        if self.fp_to_full:
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, embedding=embedding if self.film_in_sa_and_fp else None) # (B, 128, num_point)
            pred_points = l0_xyz #### 2048 points
            feature = l0_points
        else:
            pred_points = l1_xyz #### 2048 points
            feature = l1_points
            
        binary_seg_head = self.binary_seg_head(feature)
        four_point_head_offset = self.four_point_head(feature).permute(0, 2, 1)
        
        pred_scores = binary_seg_head.permute(0, 2, 1)
        pred_points = pred_points.permute(0, 2, 1) # B x N x 3
        pred_offsets = four_point_head_offset.view(four_point_head_offset.shape[0], four_point_head_offset.shape[1], 4, 3)  # B x N x 4 x 3

        if build_grasp:
            pred_4_points = pred_points.unsqueeze(2).repeat(1, 1, 4, 1) + four_point_head_offset.reshape(four_point_head_offset.shape[0], four_point_head_offset.shape[1], 4, 3) # B x N x 4 x 3
            pred_grasps_cam, offset = self.build_6d_grasp_from_four_points(pred_4_points, articubot_format=articubot_format)  # B x N x 4 x 4
        else:
            pred_grasps_cam, offset = None, None
    
        pred = dict(
            pred_scores = pred_scores,
            pred_points =pred_points,
            pred_offsets=pred_offsets,  
            pred_grasps_cam= pred_grasps_cam,  # B x N x 4 x 4
            offset_pred=offset
        )
        
        return pred
        
    def build_6d_grasp_from_four_points(self, four_point_head, gripper_depth = 0.105, articubot_format=False):
        B, N, _, _ = four_point_head.shape
        
        grasp_t = four_point_head[:, :, 0].unsqueeze(3)  # B x N x 3 x 1
        
        
        approach_direction = four_point_head[:, :, -1] - four_point_head[:, :, 0]  # B x N x 3
        baseline_direction = four_point_head[:, :, 2] - four_point_head[:, :, 1]  # B x N x 3
        
        # baseline_direction_normed = F.normalize(baseline_direction, p=2, dim=2)  # B x N x 3
        # dot_product = torch.sum(approach_direction * baseline_direction_normed, dim=2, keepdim=True)  # B x N x 1
        # projection = dot_product * baseline_direction_normed  # B x N x 3
        # approach_direction_orthog = F.normalize(approach_direction - projection, p=2, dim=2)  # B x N x 3
        # grasp_R = torch.stack([baseline_direction_normed, torch.cross(approach_direction_orthog, baseline_direction_normed),approach_direction_orthog], dim=3)  # B x N x 3 x 3
        
        approach_direction_normed = F.normalize(approach_direction, p=2, dim=2)  # B x N x 3
        dot_product = torch.sum(baseline_direction * approach_direction_normed, dim=2, keepdim=True)  # B x N x 1
        projection = dot_product * approach_direction_normed  # B x N x 3
        baseline_direction_orthog = F.normalize(baseline_direction - projection, p=2, dim=2)  # B x N x 3
        grasp_R = torch.stack([baseline_direction_orthog, torch.cross(approach_direction_normed, baseline_direction_orthog),approach_direction_normed], dim=3)  # B x N x 3 x 3
        
        if articubot_format:
            from termcolor import cprint
            cprint("Using articubot format for 6d grasp!", "yellow")
            eef_position = four_point_head[:, :, -1] # B x N x 3
            hand_position = eef_position - gripper_depth * approach_direction_normed # B x N x 3
            grasp_t = hand_position.unsqueeze(3)  # B x N x 3 x 1
        
        
        ones = torch.ones((B, N, 1, 1), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 1
        zeros = torch.zeros((B, N, 1, 3), dtype=torch.float32).to(four_point_head.device)  # B x N x 1 x 3
        homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
        grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4
        
        offset = torch.norm(four_point_head[:, :, 2] - four_point_head[:, :, 1], dim=-1, keepdim=True)  # B x N x 1
        
        return grasps, offset

if __name__ == '__main__':

    from tqdm import tqdm
    model = PointNet2(num_classes=10).cuda()
    model.eval()
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    inpput = torch.rand(1, 3, 2000).cuda()
    out = model(inpput)
    max_diff = -1
    for _ in range(1):
        inpput_translated = inpput + 50
        out_translated = model(inpput_translated)
        diff = torch.norm(out-out_translated)
        max_diff = max(max_diff, diff)
        print("difference: ", diff)