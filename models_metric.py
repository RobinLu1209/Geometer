
import torch_geometric
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, LEConv, TransformerConv
from torch.nn import functional as F

import yaml
import argparse
from itertools import combinations

class GATv2model(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature):
        super(GATv2model, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()
        print("GATv2Conv init.")

    def build(self):
        self.gat_layer_1 = GATv2Conv(in_channels=self.in_feature, out_channels=self.hidden_feature, heads=3, concat=False)
        self.gat_layer_2 = GATv2Conv(in_channels=self.hidden_feature, out_channels=self.hidden_feature, heads=3, concat=False)
    
    def forward(self, x, edge_index):
        gout_1 = self.gat_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2)
        gout_2 = self.gat_layer_2(gout_2, edge_index)
        return gout_2

class GATmodel(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature):
        super(GATmodel, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()
        print("GATConv init.")

    def build(self):
        self.gat_layer_1 = GATConv(in_channels=self.in_feature, out_channels=self.hidden_feature, heads=3, concat=False)
        self.gat_layer_2 = GATConv(in_channels=self.hidden_feature, out_channels=self.hidden_feature, heads=3, concat=False)
    
    def forward(self, x, edge_index):
        gout_1 = self.gat_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2)
        gout_2 = self.gat_layer_2(gout_2, edge_index)
        return gout_2

class Transformermodel(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature):
        super(Transformermodel, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()

    def build(self):
        self.gat_layer_1 = TransformerConv(in_channels=self.in_feature, out_channels=self.hidden_feature, heads=1, concat=False)
        self.gat_layer_2 = TransformerConv(in_channels=self.hidden_feature, out_channels=self.hidden_feature, heads=1, concat=False)

    
    def forward(self, x, edge_index):
        gout_1 = self.gat_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2)
        gout_2 = self.gat_layer_2(gout_2, edge_index)

        return gout_2

class GCNmodel(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature):
        super(GCNmodel, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()

    def build(self):
        self.gcn_layer_1 = GCNConv(in_channels=self.in_feature, out_channels=self.hidden_feature)
        self.gcn_layer_2 = GCNConv(in_channels=self.hidden_feature, out_channels=self.hidden_feature)
    
    def forward(self, x, edge_index):
        gout_1 = self.gcn_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2)
        gout_2 = self.gcn_layer_2(gout_2, edge_index)
        return gout_2

class SAGEmodel(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature):
        super(SAGEmodel, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()

    def build(self):
        self.sage_layer_1 = SAGEConv(in_channels=self.in_feature, out_channels=self.hidden_feature)
        self.sage_layer_2 = SAGEConv(in_channels=self.hidden_feature, out_channels=self.hidden_feature)
    
    def forward(self, x, edge_index):
        gout_1 = self.sage_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2)
        gout_2 = self.sage_layer_2(gout_2, edge_index)
        return gout_2

class LEmodel(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature):
        super(LEmodel, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()

    def build(self):
        self.sage_layer_1 = LEConv(in_channels=self.in_feature, out_channels=self.hidden_feature)
        self.sage_layer_2 = LEConv(in_channels=self.hidden_feature, out_channels=self.hidden_feature)
    
    def forward(self, x, edge_index):
        gout_1 = self.sage_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2)
        gout_2 = self.sage_layer_2(gout_2, edge_index)
        return gout_2


class proto_calculator(nn.Module):
    def __init__(self, z_dim, head=4, dropout=0.1):
        super(proto_calculator, self).__init__()
        self.z_dim = z_dim
        self.head = head
        self.dropout = dropout
        self.build()

    def build(self):
        self.atten = nn.MultiheadAttention(self.z_dim, self.head, self.dropout)
    
    def forward(self, embeddings, spt_idx_list):
        # 思路一：attention之后加起来
        # 思路二：用avg去attention，取第一个
        for i, spt_idx in enumerate(spt_idx_list):
            # print("spt_idx_i: ", spt_idx_i)
            spt_embedding_i = embeddings[spt_idx].unsqueeze(1)
            avg_spt_embedding = torch.sum(spt_embedding_i, dim=0) / len(spt_idx)
            attn_output, _ = self.atten(
                avg_spt_embedding.unsqueeze(1), spt_embedding_i, spt_embedding_i
            )
            if i == 0:
                proto_embedding = attn_output[:,0,:]
            else:
                proto_embedding = torch.cat((proto_embedding, attn_output[:,0,:]), dim=0)
        return proto_embedding
            
