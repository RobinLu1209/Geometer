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
from sklearn.metrics import f1_score, classification_report, precision_score

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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    acc = precision_score(labels, preds, average='macro')
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    # return correct / len(labels)
    return acc

def knowledge_dist_loss(student_output, teacher_output, T=2.0):
    # p = F.log_softmax(student_output / T, dim=1)
    p = F.softmax(student_output / T, dim=1)
    q = F.softmax(teacher_output / T, dim=1)
    # soft_loss = -torch.mean(torch.sum(q * p, dim=1))
    soft_loss = torch.mean(p * torch.log(p/q))
    return soft_loss

def proto_knowledge_loss(student_proto, teacher_proto, teacher_class_num):
    # proto_dist = euclidean_dist(student_proto[0:teacher_class_num], teacher_proto[0:teacher_class_num])
    # proto_dist = -1 * torch.log(torch.exp(-1 * proto_dist))
    # loss = torch.mean(proto_dist)

    # KLloss = nn.KLDivLoss(reduction = 'mean', log_target=True)
    # loss = KLloss(student_proto[0:teacher_class_num], teacher_proto[0:teacher_class_num])
    x = student_proto[0:teacher_class_num].softmax(dim=-1)
    y = teacher_proto[0:teacher_class_num].softmax(dim=-1)
    loss = torch.mean(x * torch.log(x/y))
    # loss = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
    return loss

def proto_separ_loss(proto_embedding, teacher_class_num):
    base_proto = proto_embedding[0: teacher_class_num]
    novel_proto = proto_embedding[teacher_class_num:]
    dist_matrix = euclidean_dist(novel_proto, base_proto)
    # dist_matrix = ball_dist(novel_proto, base_proto)

    min_dist = torch.exp(-1 * torch.min(dist_matrix, dim=1).values)
    loss = torch.mean(min_dist)
    return loss

def cos_dist_loss(proto_embedding):
    center_proto_embedding = torch.mean(proto_embedding, dim=0).unsqueeze(0)
    normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding)
    # print("normalize_proto_embedding", normalize_proto_embedding.shape)
    cos_dist_matrix = torch.mm(normalize_proto_embedding, normalize_proto_embedding.T)
    unit_matrix = torch.eye(cos_dist_matrix.shape[0]).cuda()
    # print("cos_dist_matrix", cos_dist_matrix.shape)
    cos_dist_matrix = cos_dist_matrix - unit_matrix
    loss = torch.max(cos_dist_matrix, 1).values + 1
    return torch.mean(loss)

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        # batch_loss = -alpha*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def ProximityLoss(qry_embedding, proto_embedding, labels, focal=False, class_num=None, base_class_num=None):
    dists = euclidean_dist(qry_embedding, proto_embedding)
    if focal:
        alpha = torch.tensor([1.0] * base_class_num + [0.5] * (class_num - base_class_num))
        focal_loss = FocalLoss(class_num, alpha=alpha, gamma=1)
        loss = focal_loss(-dists, labels)
    else:
        loss = F.cross_entropy(-dists, labels)
    return loss

def UniformityLoss(proto_embedding):
    center_proto_embedding = torch.mean(proto_embedding, dim=0).unsqueeze(0)
    normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding)
    # print("normalize_proto_embedding", normalize_proto_embedding.shape)
    cos_dist_matrix = torch.mm(normalize_proto_embedding, normalize_proto_embedding.T)
    unit_matrix = torch.eye(cos_dist_matrix.shape[0]).cuda()
    # unit_matrix = torch.eye(cos_dist_matrix.shape[0])
    # print("cos_dist_matrix", cos_dist_matrix.shape)
    # cos_dist_matrix = torch.sigmoid(cos_dist_matrix - unit_matrix)
    cos_dist_matrix = cos_dist_matrix - unit_matrix
    # loss = torch.mean(cos_dist_matrix, 1)
    loss = torch.max(cos_dist_matrix, 1).values
    return torch.mean(loss)

def SeparabilityLoss(proto_embedding, teacher_class_num):
    base_proto = proto_embedding[0: teacher_class_num]
    novel_proto = proto_embedding[teacher_class_num:]
    dist_matrix = euclidean_dist(novel_proto, base_proto)
    #dist_matrix = ball_dist(novel_proto, base_proto)

    min_dist = torch.exp(-1 * torch.min(dist_matrix, dim=1).values)
    loss = torch.mean(min_dist)
    return loss

def MemorabilityLoss(student_proto, teacher_proto, teacher_class_num):
    x = student_proto[0:teacher_class_num].softmax(dim=-1)
    y = teacher_proto[0:teacher_class_num].softmax(dim=-1)
    loss = torch.mean(x * torch.log(x/y))
    return loss

