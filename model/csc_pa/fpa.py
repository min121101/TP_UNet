"""Foreground prototype / correlation modules from CSC-PA (2D)."""
import torch
import torch.nn.functional as F
from torch import nn


def get_prototype(x, ss_map):
    b, _, h, w = x.size()
    ss_map = ss_map.view(b, -1, h * w)
    x = x.view(b, -1, h * w)
    return torch.bmm(ss_map, x.transpose(1, 2))


def get_correlation_map(x, prototype_block):
    b, c, h, w = x.size()
    n_p = prototype_block / prototype_block.norm(dim=2, keepdim=True).clamp(min=1e-6)
    n_x = x.view(b, c, -1) / x.view(b, c, -1).norm(dim=1, keepdim=True).clamp(min=1e-6)
    return torch.bmm(n_p, n_x).view(b, -1, h, w)


def get_ocr_vector(x):
    b, c, h, w = x.size()
    probs = x.view(b, c, -1)
    ss_map = F.softmax(probs, dim=2).view(b, c, h, w)
    pb = get_prototype(x, ss_map.clone().detach())
    return pb


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inter_channels = in_channels // 2
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(in_channels), nn.ReLU(inplace=True))
        self.theta = nn.Linear(in_channels, self.inter_channels)
        self.phi = nn.Linear(in_channels, self.inter_channels)
        self.g = nn.Linear(in_channels, self.inter_channels)
        self.W = nn.Linear(self.inter_channels, in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)
        feature = self.bn_relu(ori_feature).permute(0, 2, 1)
        b, n, c = feature.size()
        x_theta = self.theta(feature)
        x_phi = self.phi(feature).permute(0, 2, 1)
        attention = F.softmax(torch.matmul(x_theta, x_phi), dim=-1)
        g_x = self.g(feature)
        y = torch.matmul(attention, g_x)
        w_y = self.W(y).contiguous().view(b, n, c)
        return ori_feature.permute(0, 2, 1) + w_y


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.transformer = Transformer(in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, prototype_block):
        att_prototype_block = self.transformer(prototype_block)
        prototype_for_graph = att_prototype_block.permute(0, 2, 1)
        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv1(graph_prototype).max(dim=-1, keepdim=False)[0]
        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv2(graph_prototype).max(dim=-1, keepdim=False)[0]
        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv3(graph_prototype).max(dim=-1, keepdim=False)[0]
        return graph_prototype.permute(0, 2, 1)


class PrototypeCorrelationGeneration(nn.Module):
    """Single-stream prototype correlation (supervised 2D; no labeled/unlabeled split)."""

    def __init__(self, in_channels):
        super().__init__()
        self.gan = GraphAttentionNetwork(in_channels)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pb = get_ocr_vector(x)
        graph_pb = self.gan(pb)
        corr = get_correlation_map(x, graph_pb)
        return self.out(torch.cat([x, corr], dim=1))
