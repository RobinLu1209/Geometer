import torch
import torch.nn.functional as F
import torch_geometric
import random
import numpy as np
from models_metric import *
from collections import Counter
from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist

def get_dataset(dataset_name):
    if dataset_name == 'cora_ml':
        dataset = torch_geometric.datasets.CitationFull(root='../dataset', name='Cora_ML')
    elif dataset_name == 'karate':
        dataset = torch_geometric.datasets.KarateClub()
    elif dataset_name == 'amazon_computers':
        dataset = torch_geometric.datasets.Amazon(root='../dataset', name='Computers')
    elif dataset_name == 'reddit':
        dataset = torch_geometric.datasets.Reddit2(root='../dataset/reddit')
    elif dataset_name == 'cora':
        torch_geometric.datasets.CitationFull(root='../dataset', name='Cora')
    elif dataset_name == 'flickr':
        dataset = torch_geometric.datasets.AttributedGraphDataset(root='dataset', name='Flickr')
    else:
        raise NotImplementedError
    x, y = dataset.data.x, dataset.data.y
    print("[INFO] {} dataset x shape is {}, y shape is {}, edge_index shape is {}".format(dataset_name, x.shape, y.shape, dataset.data.edge_index.shape))
    label = np.unique(y.numpy())
    print("[INFO] {} dataset has {} classes: {}".format(dataset_name, len(label), label))
    label_count = Counter(dataset.data.y.numpy())
    print("label count:", label_count)
    return dataset, len(label)

def get_model(model_name, in_feature, hidden_feature):
    if model_name == 'gatv2':
        model = GATv2model(in_feature, hidden_feature)
    elif model_name == 'gat':
        model = GATmodel(in_feature, hidden_feature)
    elif model_name == 'gcn':
        model = GCNmodel(in_feature, hidden_feature)
    elif model_name == 'sage':
        model = SAGEmodel(in_feature, hidden_feature)
    elif model_name == 'le':
        model = LEmodel(in_feature, hidden_feature)
    elif model_name == 'transformer':
        model = Transformermodel(in_feature, hidden_feature)
    else:
        raise NotImplementedError
    return model

def inner_product_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    return torch.mm(x, y.transpose(0, 1))

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M

def ball_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 1 / (1 + torch.exp(torch.mm(x, y.transpose(0, 1))))
    return dist


def cosine_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return torch.mm(x, y.transpose(0, 1))  # N x M
    
def base_novel_dist_loss(base_proto, novel_proto):
    # base_proto: N_base x D
    # novel_proto: N_novel x D
    n = base_proto.size(0)
    m = novel_proto.size(0)
    d = base_proto.size(1)
    assert d == novel_proto.size(1)

    base_proto = base_proto.unsqueeze(1).expand(n, m, d)
    novel_proto = novel_proto.unsqueeze(0).expand(n, m, d)

    dist_matrix = torch.pow(base_proto - novel_proto, 2).sum(2)
    
    item_num = dist_matrix.shape[0] * dist_matrix.shape[1]
    base_novel_dist_loss =  torch.sum(torch.sum(dist_matrix, 0), 0) / item_num
    return base_novel_dist_loss