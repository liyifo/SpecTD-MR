import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class PMA(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'

        self.lin_K = Linear(in_channels, self.heads*self.hidden)
        self.lin_V = Linear(in_channels, self.heads*self.hidden)
        self.att_r = Parameter(torch.Tensor(1, heads, self.hidden))
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None',)
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)
        self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index: Adj,
                size: Size = None, return_attention_weights=None,
                edge_weight=None, edge_rel_bias=None):
        H, C = self.heads, self.hidden
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `PMA`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
        else:
            raise ValueError('Only Tensor inputs supported.')

        out = self.propagate(edge_index.clone(), x=x_V,
                             alpha=alpha_r, aggr=self.aggr,
                             edge_weight=edge_weight,
                             edge_rel_bias=edge_rel_bias)

        alpha = self._alpha
        self._alpha = None
        out += self.att_r
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        out = self.ln1(out+F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j, edge_weight, edge_rel_bias):
        alpha = alpha_j
        if edge_rel_bias is not None:
            bias = edge_rel_bias
            if not torch.is_tensor(bias):
                bias = torch.as_tensor(bias, dtype=alpha_j.dtype, device=alpha_j.device)
            else:
                bias = bias.to(alpha_j.dtype).to(alpha_j.device)
            if bias.dim() == 1:
                bias = bias.unsqueeze(-1)
            if bias.size(-1) not in (1, self.heads):
                raise ValueError('edge_rel_bias last dim must be 1 or match heads count.')
            if bias.size(-1) == 1:
                bias = bias.expand(-1, self.heads)
            alpha = alpha + bias
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if edge_weight is None:
            return x_j * alpha.unsqueeze(-1)
        weight = edge_weight.view(-1, 1, 1)
        return x_j * alpha.unsqueeze(-1) * weight

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='add'):
        if aggr is None:
            raise ValueError('aggr was not passed!')
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return x


class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn',
                 InputNorm=False,
                 heads=1,
                 attention=True
                 ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    def reset_parameters(self):
        if self.attention:
            self.prop.reset_parameters()
        else:
            if self.f_enc.__class__.__name__ != 'Identity':
                self.f_enc.reset_parameters()
            if self.f_dec.__class__.__name__ != 'Identity':
                self.f_dec.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add', edge_weight=None, edge_rel_bias=None):
        weight_tuple = None
        if self.attention:
            x, weight_tuple = self.prop(x, edge_index,
                                        edge_weight=edge_weight,
                                        edge_rel_bias=edge_rel_bias,
                                        return_attention_weights=True)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr, edge_weight=edge_weight)
            x = F.relu(self.f_dec(x))

        return x, weight_tuple

    def message(self, x_j, norm, edge_weight):
        if edge_weight is None:
            return norm.view(-1, 1) * x_j
        return norm.view(-1, 1) * x_j * edge_weight.view(-1, 1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='add'):
        if aggr is None:
            raise ValueError('aggr was not passed!')
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)
