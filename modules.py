import torch
import dgl
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax
import numpy as np

import torch.nn.functional as F


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res

        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class FeedForwardModuleHeToHo(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')


class TransformerAttentionSepHeToHoModule(nn.Module):
    def __init__(self, dim, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)

        self.feed_forward_module = FeedForwardModuleHeToHo(dim=dim,
                                                       hidden_dim_multiplier=num_heads,
                                                       dropout=dropout)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.conv1x1 = nn.Conv1d(in_channels=dim * 5, out_channels=dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(dim * 5)

        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

        self.linear_1 = nn.Linear(in_features=dim * 8, out_features=dim)
        self.act_1 = nn.GELU()
        self.act_2 = nn.GELU()
        self.act_3 = nn.GELU()
        self.act_4 = nn.GELU()
        self.act_5 = nn.GELU()
        self.linear_2 = nn.Linear(in_features=dim, out_features=dim)
        self.linear_3 = nn.Linear(in_features=32, out_features=32)
        self.linear_4 = nn.Linear(in_features=32, out_features=32)
        self.linear_5 = nn.Linear(in_features=32, out_features=10)

    def forward(self, graph, x_):
        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        x_0 = ops.u_add_e_mean(graph, x_, norm_coefs)
        x_1 = ops.u_sub_e_mean(graph, x_, norm_coefs)
        x_2 = ops.u_mul_e_mean(graph, x_, norm_coefs)
        x_3 = ops.u_div_e_mean(graph, x_, norm_coefs)

        x_concat = torch.cat((x_1, x_2, x_0, x_, x_3), dim=1)

        x = self.conv1x1(x_concat.unsqueeze(2)).squeeze(2)
        x_concat = self.bn(x_concat)

        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        values_mean = torch.mean(queries)
        values = values - values_mean

        sigma = torch.std(values)
        values = values / sigma

        queries_mean = torch.mean(queries)
        queries = queries - queries_mean

        sigma = torch.std(queries)
        queries = queries / sigma

        keys_mean = torch.mean(keys)
        keys = keys - keys_mean

        sigma = torch.std(keys)
        keys = keys / sigma

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)

        score = torch.einsum("bhd,bhd->bh", [queries, keys])

        score = nn.functional.softmax(score, dim=-1)

        mu = torch.mean(score)
        sim = score - mu

        sigma = torch.std(sim)
        sim = sim / sigma

        noise_std = 0.004
        noise = torch.randn_like(sim) * noise_std
        sim += noise

        sim_squared = torch.mm(sim, sim.t())

        sim_squared = F.softmax(sim_squared, dim=1)

        src, dst = torch.nonzero(sim_squared, as_tuple=True)
        weights = sim_squared[src, dst]

        # 使用DGL创建图形
        g = dgl.graph((src, dst))

        x = ops.u_mul_e_sum(g, values, weights)

        x = self.feed_forward_module(x)

        return x
