import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros

import math
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

from torch_geometric.nn import aggr
import os

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # X: [N, in_channels]
        # edge_index: [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1/2)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

         # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] OUT PUT DIMS = [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim)


    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        conv_last = GCNConv(in_channels=hidden_dim, out_channels=embedding_dim)
        return conv_first, conv_last
    
    def forward(self, x, edge_index):
        # import pdb; pdb.set_trace()
        x = self.conv_first(x, edge_index)
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)
        return x

class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        beta (bool, optional): If set, will combine aggregation and
            skip information via

            .. math::
                \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

            with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
            [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
            \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). Edge features are added to the keys after
            linear transformation, that is, prior to computing the
            attention dot product. They are also added to final values
            after the same linear transformation. The model is:

            .. math::
                \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                \right),

            where the attention coefficients :math:`\alpha_{i,j}` are now
            computed via:

            .. math::
                \alpha_{i,j} = \textrm{softmax} \left(
                \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                {\sqrt{d}} \right)

            (default :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output and the
            option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        fold_n=None,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.fold_n = fold_n

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        layer: str,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        # forward_type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # forward_type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        self.layer = layer
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # import pdb; pdb.set_trace()

        internal_gene_edge_num_all_df = pd.read_csv('./ROSMAP-graph-data/internal-gene-edge-num-all.csv') # read internal edge data
        weight_gene_edge_num_all_df = internal_gene_edge_num_all_df.copy()
        edge_weight_from = alpha[:,0].cpu().detach().numpy()
        edge_weight_to = alpha[:,1].cpu().detach().numpy()
        weight_gene_edge_num_all_df['edge_weight_from'] = edge_weight_from
        weight_gene_edge_num_all_df['edge_weight_to'] = edge_weight_to
        weight_gene_edge_num_all_df['edge_weight_avg'] = (edge_weight_from + edge_weight_to) / 2

        save_path = './ROSMAP-analysis/fold_' + str(self.fold_n)
        while os.path.exists(save_path) == False:
            os.mkdir(save_path)
        weight_gene_edge_num_all_df.to_csv(save_path + '/' + self.layer + '_internal_link_edge_weight.csv', index=False, header=True)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GraphFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_head, fold_n):
        super(GraphFormer, self).__init__()
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.fold_n = fold_n

        self.conv_first, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim * num_head)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim * num_head)

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=self.num_head, fold_n = self.fold_n)
        conv_last = TransformerConv(in_channels=hidden_dim * self.num_head, out_channels=embedding_dim, heads=self.num_head, fold_n = self.fold_n)
        return conv_first, conv_last

    def forward(self, x, edge_index):
        # edge index stnads for internal links
        x = self.conv_first(x=x, layer='first', edge_index=edge_index)
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_last(x=x, layer='last', edge_index=edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)
        return x

class WeBConv(MessagePassing):
    def __init__(self, in_channels, out_channels, node_num, num_edge, device):
        super(WeBConv, self).__init__(aggr='add')
        self.node_num = node_num
        self.num_edge = num_edge

        self.up_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.down_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias_proj = torch.nn.Linear(in_channels, out_channels, bias=False)

        ##### [edge_weight] FOR ALL EDGES IN ONE [(gene)] GRAPH #####
        ### [up_gene_edge_weight] [num_gene_edge / 21729] ###
        up_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.up_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_edge) * up_std_gene_edge).to(device))
        ### [down_gene_edge_weight] [num_gene_edge / 21729] ###
        down_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.down_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_edge) * down_std_gene_edge).to(device))


    def forward(self, x, edge_index):
        # [batch_size]
        batch_size = int(x.shape[0] / self.node_num)
        # TEST PARAMETERS
        print(torch.sum(self.up_gene_edge_weight))
        print(torch.sum(self.down_gene_edge_weight))

        ### [edge_index, x] ###
        up_edge_index = edge_index
        up_x = self.up_proj(x)
        down_edge_index = torch.flipud(edge_index)
        down_x = self.down_proj(x)
        bias_x = self.bias_proj(x)

        ### [edge_weight] ###
        up_edge_weight =self.up_gene_edge_weight
        down_edge_weight = self.down_gene_edge_weight
        # [batch_up/down_edge_weight] [N*21845]
        batch_up_edge_weight = up_edge_weight.repeat(1, batch_size)
        batch_down_edge_weight = down_edge_weight.repeat(1, batch_size)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        # [up]
        up_row, up_col = up_edge_index
        up_deg = degree(up_col, x.size(0), dtype=x.dtype)
        up_deg_inv_sqrt = up_deg.pow(-1)
        up_deg_inv_sqrt[up_deg_inv_sqrt == float('inf')] = 0
        up_norm = up_deg_inv_sqrt[up_col]
        # [down]
        down_row, down_col = down_edge_index
        down_deg = degree(down_col, x.size(0), dtype=x.dtype)
        down_deg_inv_sqrt = down_deg.pow(-1)
        down_deg_inv_sqrt[down_deg_inv_sqrt == float('inf')] = 0
        down_norm = down_deg_inv_sqrt[down_col]
        # Check [ torch.sum(up_norm[0:21729]==up_norm[21845:43574])==21729 ]

        # Step 4-5: Start propagating messages.
        x_up = self.propagate(up_edge_index, x=up_x, norm=up_norm, edge_weight=batch_up_edge_weight)
        x_down = self.propagate(down_edge_index, x=down_x, norm=down_norm, edge_weight=batch_down_edge_weight)
        x_bias = bias_x
        concat_x = torch.cat((x_up, x_down, x_bias), dim=-1)
        concat_x = F.normalize(concat_x, p=2, dim=-1)
        return concat_x, up_edge_weight, down_edge_weight

    def message(self, x_j, norm, edge_weight):
        # [x_j] has shape [E, out_channels]
        # Step 4: Normalize node features.
        weight_norm = torch.mul(norm, edge_weight)
        return weight_norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] has shape [N, out_channels]
        return aggr_out


class SubGraphAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, head, negative_slope, aggr, device, graph_output_folder):
        super(SubGraphAttentionConv, self).__init__(node_dim=0)
        assert out_channels % head == 0
        self.k = out_channels // head

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_head = head
        self.negative_slope = negative_slope
        self.aggr = aggr
        self.device = device
        self.graph_output_folder = graph_output_folder

        self.weight_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, head, 2 * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight.data)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, mask, batch_random_final_dl_input_df, sp_path):
        graph_output_folder = self.graph_output_folder
        ### ADD SELF LOOPS IN THE EDGE SPACE
        # import pdb; pdb.set_trace()
        x = self.weight_linear(x).view(-1, self.num_head, self.k) # N * num_head * h
                # import pdb; pdb.set_trace()
        batch_size = batch_random_final_dl_input_df.shape[0]
        sub_khop_node_num = int(x.shape[0] / batch_size)
        return self.propagate(edge_index, x=x, mask=mask, batch_df=batch_random_final_dl_input_df, path=sp_path, sub_khop_node_num=sub_khop_node_num,graph_output_folder=graph_output_folder)

    def message(self, edge_index, x_i, x_j, mask, batch_df, path, sub_khop_node_num, graph_output_folder):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) # E * num_head * h
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        head_mask = torch.tile(mask, (1, self.num_head)).reshape(x_j.shape[0], self.num_head, 1)
        x_j.masked_fill_(head_mask==0, 0.)

        # CACULATE THE ATTENTION
        batch_size = batch_df.shape[0]
        batch_alpha = alpha.clone()
        mask = mask.unsqueeze(-1)
        batch_mask = mask.reshape(batch_size, -1)
        batch_alpha = batch_alpha.masked_fill_(mask==0, 0.)
        batch_alpha = torch.mean(batch_alpha, dim=1)
        batch_alpha = batch_alpha.reshape(batch_size, -1)
        batch_survival_label_list = list(batch_df['individualID'])
        batch_edgeindex = edge_index.clone()
        batch_edgeindex = batch_edgeindex.t().reshape(batch_size, -1, 2)

        # PREPARE [cell line] MAP
        survival_label_map_df = pd.read_csv('./' + graph_output_folder + '/survival_label_map_dict.csv')
        survival_label_map_dict = dict(zip(survival_label_map_df.individualID, survival_label_map_df.individualID_Num))

        # import pdb; pdb.set_trace()
        for batch_idx in range(batch_size):
            input_row = batch_df.iloc[batch_idx]
            survival_label = input_row.iloc[0]
            survival_label_num = survival_label_map_dict[survival_label]
            survival_label_save_path = path + '/survival' + str(survival_label_num) + '.csv'
            if os.path.isfile(survival_label_save_path) == True:
                continue

            # import pdb; pdb.set_trace()
            
            hop_edge_num = int((torch.sum(batch_mask[batch_idx])).cpu().detach().numpy())
            hop1_list = ['hop1' for x in range(hop_edge_num)]
            hop2_list = ['hop2' for x in range(hop_edge_num)]
            hop3_list = ['hop3' for x in range(hop_edge_num)]
            hop_list = hop1_list + hop2_list + hop3_list


            from_array = batch_edgeindex[batch_idx, :, 0] - (batch_idx * sub_khop_node_num)
            to_array = batch_edgeindex[batch_idx, :, 1] - (batch_idx * sub_khop_node_num)

            from_list =  list(from_array.cpu().detach().numpy())
            to_list =  list(to_array.cpu().detach().numpy())
            mask_list = list(batch_mask[batch_idx].cpu().detach().numpy())
            attention_list = list(batch_alpha[batch_idx].cpu().detach().numpy())
            survival_label_att_df = pd.DataFrame({'From': from_list,
                                             'To': to_list,
                                             'Mask': mask_list,
                                             'Attention': attention_list,
                                             'Hop': hop_list})
            # import pdb; pdb.set_trace()
            survival_label_att_df.to_csv(survival_label_save_path, index=False, header=True)

        return x_j * alpha.view(-1, self.num_head, 1) # E * num_head * h

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        aggr_out = aggr_out + self.bias
        return aggr_out


class TraverseSubGNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, head, max_layer, device, graph_output_folder):
        super(TraverseSubGNN, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.max_layer = max_layer
        self.device = device
        self.graph_output_folder = graph_output_folder
        self.subgat_khop = self.build_traverse_subgat_layer(
                            input_dim=input_dim, embedding_dim=embedding_dim, head=head, max_layer=max_layer, device=device)
        self.act2 = nn.LeakyReLU(negative_slope=0.1)
        self.norm = nn.BatchNorm1d(embedding_dim)

    def build_traverse_subgat_layer(self, input_dim, embedding_dim, head, max_layer, device):
        subgat_khop = SubGraphAttentionConv(in_channels=input_dim, out_channels=embedding_dim, head=3, negative_slope=0.2, aggr="add", device=device,graph_output_folder=self.graph_output_folder)
        return subgat_khop
    
    def forward(self, subx, subadj_edgeindex, sub_mask_edgeindex, batch_size, subgraph_size, batch_random_final_dl_input_df, sp_path):
        khop_subx = self.subgat_khop(subx, subadj_edgeindex, sub_mask_edgeindex, batch_random_final_dl_input_df, sp_path)
        khop_subx = self.norm(self.act2(khop_subx))
        khop_subx = khop_subx.reshape(batch_size, self.max_layer, subgraph_size, -1)
        khop_subx = torch.mean(khop_subx, dim=1)
        return khop_subx


class GlobalWeBGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim,
                            node_num, num_edge, device):
        super(GlobalWeBGNN, self).__init__()
        self.node_num = node_num
        self.num_edge = num_edge

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.device = device
        self.webconv_first, self.webconv_block, self.webconv_last = self.build_webconv_layer(
                    input_dim, hidden_dim, embedding_dim, node_num, num_edge)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)


    def build_webconv_layer(self, input_dim, hidden_dim, embedding_dim, node_num, num_edge):
        # webconv_first [input_dim, hidden_dim]
        webconv_first = WeBConv(in_channels=input_dim, out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, device=self.device)
        # webconv_block [hidden_dim*3, hidden_dim]
        webconv_block = WeBConv(in_channels=int(hidden_dim*3), out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, device=self.device)
        # webconv_last [hidden_dim*3, embedding_dim]
        webconv_last = WeBConv(in_channels=int(hidden_dim*3), out_channels=embedding_dim,
                node_num=node_num, num_edge=num_edge, device=self.device)
        return webconv_first, webconv_block, webconv_last

    def forward(self, x, edge_index):
        # webconv_first
        web_x, first_up_edge_weight, first_down_edge_weight = self.webconv_first(x, edge_index)
        web_x = self.act2(web_x)
        # webconv_block
        web_x, block_up_edge_weight, block_down_edge_weight = self.webconv_block(web_x, edge_index)
        web_x = self.act2(web_x)
        # webconv_last
        web_x, last_up_edge_weight, last_down_edge_weight = self.webconv_last(web_x, edge_index)
        web_x = self.act2(web_x)
        # [mean_up_edge_weight / mean_down_edge_weight]
        mean_up_edge_weight = (1/3) * (first_up_edge_weight + block_up_edge_weight + last_up_edge_weight)
        mean_down_edge_weight = (1/3) * (first_down_edge_weight + block_down_edge_weight + last_down_edge_weight)
        return web_x, mean_up_edge_weight, mean_down_edge_weight


class TSGNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_edge, device, num_class, graph_output_folder, fold_n):
        super(TSGNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.graph_output_folder = graph_output_folder

        self.node_num = node_num
        self.num_edge = num_edge

        self.num_class = num_class
        self.fold_n = fold_n

        self.gcn = GCN(input_dim=input_dim, hidden_dim=input_dim, embedding_dim=input_dim)
        self.gformer = GraphFormer(input_dim=input_dim, hidden_dim=input_dim, embedding_dim=input_dim, num_head=2, fold_n=fold_n)

        self.max_layer = 3
        self.traverse_subgnn = TraverseSubGNN(input_dim=input_dim, embedding_dim=input_dim*3, head=3, max_layer=self.max_layer, device=device, graph_output_folder=graph_output_folder)
        self.linear_traverse_x = nn.Linear(input_dim*3, 1)
        self.linear_x = nn.Linear(input_dim, input_dim)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.x_norm = nn.BatchNorm1d(input_dim)

        ##### GLOBAL PROPAGATION LAYERS
        self.global_gnn = GlobalWeBGNN(input_dim=input_dim+1, hidden_dim=hidden_dim, embedding_dim=embedding_dim,
                                       node_num=node_num, num_edge=num_edge, device=device)

        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()
        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)


        self.att_linear = torch.nn.Linear(input_dim * 2, input_dim)
        self.graph_prediction = torch.nn.Linear(embedding_dim * 6, num_class) # torch.nn.Linear(180, num_class)

    def forward(self, x, edge_index, internal_edge_index, graph_output_folder, batch_random_final_dl_input_df, analysis_save_path):
        ### BUILD UP ASSIGNMENT MATRIX
        # import pdb; pdb.set_trace()

        # x_norm = self.x_norm(x)
        # x_norm = x_norm.reshape(-1, self.node_num, self.input_dim)
        # x = self.gcn(x, internal_edge_index)
        x = self.gformer(x, internal_edge_index)
        x = self.att_linear(x)
        # x = self.gcn(x, edge_index)
        # X += NEW_X
        x = x.reshape(-1, self.node_num, self.input_dim)
        
        ### TRAVERSE SUBGRAPH
        # FORM NOTATION TO [signaling pathways]
        form_data_path = './' + graph_output_folder + '/form_data'
        kegg_path_gene_interaction_df = pd.read_csv('./' + graph_output_folder + '/keggpath-gene-edge-num-all.csv')
        kegg_sp_list = list(set(kegg_path_gene_interaction_df['path']))
        kegg_sp_list.sort()
        kegg_sp_notation_list = ['sp' + str(x) for x in range(1, len(kegg_sp_list)+1)]
        kegg_num_notation_list = [x for x in range(len(kegg_sp_list))]
        
        # FOR LOOP TO START TRAVERSE
        batch_size = x.shape[0] # batch_size
        # INITIALIZE [traverse_x]
        traverse_sp_x = torch.zeros(x.shape[0], len(kegg_sp_list), self.node_num, self.input_dim*3).to(device='cuda')
        traverse_spnum_sum = torch.zeros([self.node_num, 1]).to(device='cuda')

        for sp_num_notation in kegg_num_notation_list:
            # ASSIGN CERTAIN SUBGRAPH NODE
            sp_notation = kegg_sp_notation_list[sp_num_notation]
            subassign_index = np.load(form_data_path + '/' + sp_notation + '_gene_idx.npy')
            subgraph_size = subassign_index.shape[0]

            # SP PATH CREATION
            sp_path = analysis_save_path + '/' + sp_notation
            os.makedirs(sp_path, exist_ok=True)

            # EXPAND NODE TO MULTIPLE HOPs
            subx = x[:, subassign_index, :]
            batch_subx = torch.tile(subx, (1, self.max_layer, 1))
            batch_subx = batch_subx.reshape(-1, subx.shape[2])
            # GET [sp_adj_edgeindex, sp_mask_edgeindex]
            sp_notation = kegg_sp_notation_list[sp_num_notation]
            sp_adj_edgeindex = np.load(form_data_path + '/' + sp_notation + '_khop_subadj_edgeindex.npy')
            sp_adj_edgeindex = torch.from_numpy(sp_adj_edgeindex).to(device='cuda')
            sp_mask_edgeindex = np.load(form_data_path + '/' + sp_notation + '_khop_mask_edgeindex.npy')
            sp_mask_edgeindex = torch.from_numpy(sp_mask_edgeindex).to(device='cuda')
            # EXPAND [adj_edgeindex, mask_edgeindex] TO [batch_size]
            tmp_sp_adj_edgeindex = sp_adj_edgeindex.clone()
            batch_sp_adj_edgeindex = sp_adj_edgeindex
            batch_sp_mask_edgeindex = sp_mask_edgeindex
            for batch_idx in range(2, batch_size + 1):
                tmp_sp_adj_edgeindex += (subgraph_size) * (self.max_layer)
                batch_sp_adj_edgeindex = torch.cat([batch_sp_adj_edgeindex, tmp_sp_adj_edgeindex], dim=1)
                batch_sp_mask_edgeindex = torch.cat([batch_sp_mask_edgeindex, sp_mask_edgeindex])
            # RUN [traverse_subgnn]
            khop_subx = self.traverse_subgnn(batch_subx, batch_sp_adj_edgeindex, batch_sp_mask_edgeindex, batch_size, subgraph_size, batch_random_final_dl_input_df, sp_path)
            traverse_sp_x[:, sp_num_notation, subassign_index, :] = khop_subx
            traverse_sp_tmp_num = torch.zeros([self.node_num, 1]).to(device='cuda')
            traverse_sp_tmp_num[subassign_index, :] = 1
            traverse_spnum_sum += traverse_sp_tmp_num

        # import pdb; pdb.set_trace()
        
        traverse_sum_x = torch.sum(traverse_sp_x, axis=1)
        traverse_spnum_sum = torch.tile(traverse_spnum_sum, (batch_size, 1, 1))
        traverse_x = torch.div(traverse_sum_x, traverse_spnum_sum)
        traverse_x = torch.nan_to_num(traverse_x, nan=0)
        
        # import pdb; pdb.set_trace()
        transformed_traverse_x = self.linear_traverse_x(traverse_x)
        # transformed_x = self.linear_x(x)

        # # USE RES-NET IDEA
        # transformed_traverse_x = transformed_traverse_x.reshape(-1, self.input_dim)
        # norm_transformed_traverse_x = self.x_norm(transformed_traverse_x)
        # norm_transformed_traverse_x = norm_transformed_traverse_x.reshape(-1, self.node_num, self.input_dim)
        # global_x = x + norm_transformed_traverse_x

        global_x = torch.cat((x, transformed_traverse_x), dim=-1)
        # global_x = torch.cat((transformed_x, transformed_traverse_x), dim=-1)
        # global_x = torch.cat((x_norm, traverse_x), dim=-1)
        global_x = global_x.reshape(-1, global_x.shape[2])
        global_x, global_mean_up_edge_weight, global_mean_down_edge_weight = self.global_gnn(global_x, edge_index)
        final_x = global_x

        # max pooling
        final_x_reshaped = final_x.view(-1, self.node_num, self.embedding_dim * 3)
        x_max_pool = torch.max(final_x_reshaped, dim=1)[0]
        final_features = torch.cat([x_max_pool, final_x_reshaped.mean(dim=1)], dim=1)
        final_features = F.relu(final_features)
        
        output = self.graph_prediction(final_features)
        _, ypred = torch.max(output, dim=1)
        return output, ypred

    def loss(self, output, label):
        # import pdb; pdb.set_trace()
        num_class = self.num_class
        # Use weight vector to balance the loss
        weight_vector = torch.zeros([num_class]).to(device='cuda')
        label = label.long()
        for i in range(num_class):
            n_samplei = torch.sum(label == i)
            if n_samplei == 0:
                weight_vector[i] = 0
            else:
                weight_vector[i] = len(label) / (n_samplei)
        # Calculate the loss
        output = torch.log_softmax(output, dim=-1)
        loss = F.nll_loss(output, label, weight_vector)
        return loss