import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch_geometric
import random
from tensorboardX import SummaryWriter
from torch_geometric.utils.degree import degree
from utils_metric import *
from utils import *
import yaml
import argparse
import time
import os
from sklearn.metrics import f1_score, classification_report, precision_score
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import math 
import warnings
warnings.filterwarnings("ignore")


def get_class_by_id(y_label):
    labels = np.unique(y_label.cpu().numpy())
    class_by_id = {}
    for label in labels:
        class_by_id[str(label)] = []
    for i in range(len(y_label)):
        class_by_id[str(int(y_label[i]))].append(i)
    return class_by_id

def get_spt_qry_id_old(class_by_id, k_spt, k_qry=None):
    spt_idx, qry_idx = [], []
    # spt_idx_list, qry_idx = [], []
    class_list = list(class_by_id.keys())
    for class_key in class_list:
        class_i_spt_idx = random.sample(
            class_by_id[class_key], k_spt
        )
        spt_idx = spt_idx + class_i_spt_idx
        # spt_idx_list.append(class_i_spt_idx)
        if k_qry == None:
            class_i_qry_idx = [n for n in class_by_id[class_key] if n not in class_i_spt_idx]
        else:
            class_i_qry_idx = random.sample(
                [n for n in class_by_id[class_key] not in class_i_spt_idx], k_qry
            )
        qry_idx = qry_idx + class_i_qry_idx
    return spt_idx, qry_idx

def get_spt_qry_id(class_by_id, k_spt, k_qry=None):
    # spt_idx, qry_idx = [], []
    spt_idx_list, qry_idx = [], []
    class_list = list(class_by_id.keys())
    for class_key in class_list:
        class_i_spt_idx = random.sample(
            class_by_id[class_key], k_spt
        )
        # spt_idx = spt_idx + class_i_spt_idx
        spt_idx_list.append(class_i_spt_idx)
        if k_qry == None:
            class_i_qry_idx = [n for n in class_by_id[class_key] if n not in class_i_spt_idx]
        else:
            class_i_qry_idx = random.sample(
                [n for n in class_by_id[class_key] not in class_i_spt_idx], k_qry
            )
        qry_idx = qry_idx + class_i_qry_idx
    return spt_idx_list, qry_idx

def get_ft_spt_qry_id(class_by_id, k_spt, k_qry=None):
    spt_idx_list, qry_idx = [], []
    class_list = list(class_by_id.keys())
    for class_key in class_list:
        if len(class_by_id[class_key]) == k_spt:
            class_i_spt_idx = class_by_id[class_key]
        else:
            ratio = int(random.uniform(1, 5))
            class_i_spt_idx = random.sample(
                class_by_id[class_key], k_spt * ratio
            )
        spt_idx_list.append(class_i_spt_idx)
        if k_qry == None:
            class_i_qry_idx = [n for n in class_by_id[class_key] if n not in class_i_spt_idx]
        else:
            class_i_qry_idx = random.sample(
                [n for n in class_by_id[class_key] if n not in class_i_spt_idx], k_qry
            )
        qry_idx = qry_idx + class_i_qry_idx
    return spt_idx_list, qry_idx

def get_unbalanced_spt_qry_id(class_by_id, k_spt_max, k_qry=None):
    spt_idx_list, qry_idx = [], []
    class_list = list(class_by_id.keys())

    for class_key in class_list:
        k_spt = int(random.uniform(1, k_spt_max))
        class_i_spt_idx = random.sample(
            class_by_id[class_key], k_spt
        )
        spt_idx_list.append(class_i_spt_idx)
        
        if k_qry == None:
            class_i_qry_idx = [n for n in class_by_id[class_key] if n not in class_i_spt_idx]
        else:
            class_i_qry_idx = random.sample(
                [n for n in class_by_id[class_key] if n not in class_i_spt_idx], k_qry
            )
        qry_idx = qry_idx + class_i_qry_idx

    return spt_idx_list, qry_idx

def get_random_fewshot_spt_qry_id(class_by_id, k_spt_min, k_spt_max, ratio=0.5, k_qry=None):
    spt_idx_list, qry_idx = [], []
    class_list = list(class_by_id.keys())

    for class_key in class_list:
        
        if random.random() > 0.5:
            k_spt = k_spt_max
        else:
            k_spt = k_spt_min
        class_i_spt_idx = random.sample(
            class_by_id[class_key], k_spt
        )
        spt_idx_list.append(class_i_spt_idx)
        
        if k_qry == None:
            class_i_qry_idx = [n for n in class_by_id[class_key] if n not in class_i_spt_idx]
        else:
            class_i_qry_idx = random.sample(
                [n for n in class_by_id[class_key] not in class_i_spt_idx], k_qry
            )
        qry_idx = qry_idx + class_i_qry_idx

    return spt_idx_list, qry_idx

def get_test_spt_list(y_label, mapping_idx):
    labels = np.unique(y_label.cpu().numpy())
    class_by_id = {}
    # class_by_id_test_qry = {}
    
    for label in labels:
        class_by_id[str(label)] = []
        # class_by_id_test_qry[str(label)] = []

    for i in range(len(y_label)):
        if i not in mapping_idx:
            class_by_id[str(int(y_label[i]))].append(i)
        # else:
        #     class_by_id_test_qry[str(int(y_label[i]))].append(i)

    spt_idx_list = []
    class_list = list(class_by_id.keys())

    # sample_num_list = []
    # for class_key in class_list:
    #     sample_num_list.append(len(class_by_id_test_qry[class_key]))
    # test_qry_num = np.min(np.array(sample_num_list))
    # test_qry_num = np.max(np.array(sample_num_list))
    
    for class_key in class_list:
        class_i_spt_idx = class_by_id[class_key]
        spt_idx_list.append(class_i_spt_idx)
        # test_qry_idx = test_qry_idx + random.sample(
        #     class_by_id_test_qry[class_key], test_qry_num
        # )
        # test_qry_idx = test_qry_idx + list(np.random.choice(
        #     class_by_id_test_qry[class_key], test_qry_num
        # ))
    return spt_idx_list

def get_proto_embedding(proto_model, spt_idx_list, embeddings, degree_list=None):
    for i, spt_idx in enumerate(spt_idx_list):
        spt_embedding_i = embeddings[spt_idx]
        if degree_list == None:
            proto_embedding_i, _ = proto_model(spt_embedding_i)
        else:
            degree_list_i = degree_list[spt_idx]
            proto_embedding_i, _ = proto_model(spt_embedding_i, degree_list_i)
        if i == 0:
            proto_embedding = proto_embedding_i
        else:
            proto_embedding = torch.cat((proto_embedding, proto_embedding_i), dim=0)
    return proto_embedding

def get_degree_list(edge_index, num_nodes):
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    edge_index, tmp_edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, 1., num_nodes
    )
    edge_weight = tmp_edge_weight
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    return deg

def get_degree_proto_embedding(spt_idx_list, embeddings, degree_list):
    for i, spt_idx in enumerate(spt_idx_list):
        spt_embedding_i = embeddings[spt_idx]
        degree_list_i = degree_list[spt_idx]
        norm_degree = degree_list_i / torch.sum(degree_list_i)
        norm_degree = norm_degree.unsqueeze(1)
        # print("spt_embedding_i shape", spt_embedding_i.shape, "norm_degree shape", norm_degree.shape)
        proto_embedding_i = torch.sum(
            torch.mul(spt_embedding_i, norm_degree), 0
        )
        if i == 0:
            proto_embedding = proto_embedding_i.unsqueeze(0)
        else:
            proto_embedding = torch.cat((proto_embedding, proto_embedding_i.unsqueeze(0)), dim=0)
    return proto_embedding
