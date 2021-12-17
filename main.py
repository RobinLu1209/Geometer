import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import random
from tensorboardX import SummaryWriter
from torch_geometric.utils.degree import degree
from utils_metric import *
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
from torch.distributions import kl_divergence


parser = argparse.ArgumentParser(description='incremental')
parser.add_argument('--config_filename', default='../config_cora_ml_stream.yaml', type=str)
parser.add_argument('--model', default='gat', type=str)
parser.add_argument('--method', default='fc_finetune', type=str)
parser.add_argument('--iteration', default=5, type=int, help='times of repeated experiments')
parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train.')
parser.add_argument('--ft_episodes', type=int, default=200, help='Number of episodes to train.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--ft_epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--ft_lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--loss_ratio', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--k_shot', type=int, default=5)
parser.add_argument('--memo', type=str)
parser.add_argument('--k_spt_max', type=int, default=20)
parser.add_argument('--tf', type=bool, default=True)
parser.add_argument('--loss', type=int, default=[1,1,1,1])
parser.add_argument('--start', type=int, default=0)

args = parser.parse_args()
str_s = args.config_filename.find("_") + 1
str_e = args.config_filename.find(".")
dataname = args.config_filename[str_s: str_e]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

def get_proto_embedding_old(spt_idx_list, embeddings):
    for i, spt_idx in enumerate(spt_idx_list):
        spt_embedding_i = embeddings[spt_idx]
        proto_embedding_i = torch.sum(spt_embedding_i, 0) / len(spt_idx)
        if i == 0:
            proto_embedding = proto_embedding_i.unsqueeze(0)
        else:
            proto_embedding = torch.cat((proto_embedding, proto_embedding_i.unsqueeze(0)), dim=0)
    return proto_embedding

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

class ProtoRepre(nn.Module):
    def __init__(self, hidden_state):
        super(ProtoRepre, self).__init__()
        self.hidden_state = hidden_state
        # self.gat_layer = GATConv(in_channels=hidden_state, out_channels=hidden_state, heads=1, concat=False)
        self.atten = nn.MultiheadAttention(hidden_state, 2)
    
    def forward(self, spt_embedding_i, degree_list_i=None):
        if degree_list_i == None:
            avg_proto_i = torch.sum(spt_embedding_i, 0) / spt_embedding_i.shape[0]
            attn_input = torch.cat((avg_proto_i.unsqueeze(0), spt_embedding_i), dim=0).unsqueeze(1)
        else:
            norm_degree = degree_list_i / torch.sum(degree_list_i)
            norm_degree = norm_degree.unsqueeze(1)
            avg_proto_i = torch.sum(
            torch.mul(spt_embedding_i, norm_degree), 0
            )
            attn_input = torch.cat((avg_proto_i.unsqueeze(0), spt_embedding_i), dim=0).unsqueeze(1)
        attn_output, attn_output_weights = self.atten(attn_input, attn_input, attn_input)
        proto_embedding_i = attn_output[0] + avg_proto_i.unsqueeze(0)
        return proto_embedding_i, attn_output_weights

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


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("[INFO] device: GPU")
    else:
        device = torch.device('cpu')
        print("[INFO] device: CPU")


    with open(args.config_filename) as f:
        config = yaml.load(f)
    
    data_args = config['data']
    model_args = config['model']

    processed_data_dir = "../processed_dataset/{}_stream/".format(data_args['name'])

    total_accuracy_meta_test = []

    dataname = data_args['name']

    pkl_path = "model_pkl/{}/".format(dataname)
    result_path = "result/{}/".format(dataname)

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    total_acc_list = []

    if args.tf:
        writer = SummaryWriter('log_dir')

    for i in range(args.iteration):

        acc_list = []

        pkl_filename = pkl_path + "{}_{}_{}_model_{}_{}.pkl".format(data_args['name'], args.model, args.method, i, args.memo)
        proto_pkl_filename = pkl_path + "{}_{}_{}_proto_{}_{}.pkl".format(data_args['name'], args.model, args.method, i, args.memo)
        prediction_filename = result_path + "{}_{}_prediction_{}.npy".format(args.model, args.method, i)

        start_time = time.time()
        print("Iteration of experiment: [{}/{}]".format(i, args.iteration))

        encoder = get_model(args.model, model_args['in_feature'], model_args['hidden_feature']).to(device)

        base_npz_filename = "{}_{}base_stream.npz".format(data_args['name'], data_args['n_base'])
        npz_file = np.load(processed_data_dir + base_npz_filename, allow_pickle=True)

        x_train, y_train, edge_index_train = torch.tensor(npz_file['base_train_x_feature']).to(device), torch.tensor(npz_file['base_train_y_label']).to(device), torch.tensor(npz_file['base_train_edge_index']).to(device)
        train_class_by_id = get_class_by_id(y_train)
        train_node_num = y_train.shape[0]
        train_degree_list = get_degree_list(edge_index_train, train_node_num)

        x_val, y_val, edge_index_val, val_mapping_idx = torch.tensor(npz_file['base_val_x_feature']).to(device), torch.tensor(npz_file['base_val_y_label']).to(device), torch.tensor(npz_file['base_val_edge_index']).to(device), npz_file['val_mapping_idx']
        val_class_by_id = get_class_by_id(y_val)
        val_node_num = y_val.shape[0]
        val_degree_list = get_degree_list(edge_index_val, val_node_num)

        optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
        
        base_class_num = data_args['n_base']
        class_num = data_args['n_base']
        n_novel_list = data_args['n_novel_list']

        best_base_acc = -1

        proto_model = ProtoRepre(model_args['hidden_feature']).to(device)
        optimizer_proto = optim.Adam(proto_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

        if args.start == 0:
            for episode in tqdm(range(args.episodes)):
                encoder.train()
                proto_model.train()
                optimizer_encoder.zero_grad()
                optimizer_proto.zero_grad()
                # edge_index_train, _ = dropout_adj(edge_index_train, p=0.2)
                embeddings = encoder(x_train, edge_index_train)

                # -----[Uniform sampling]--------------------------------
                spt_idx_list, qry_idx = get_unbalanced_spt_qry_id(train_class_by_id, k_spt_max=args.k_spt_max, k_qry=30)
                proto_embedding = get_proto_embedding(proto_model, spt_idx_list, embeddings, train_degree_list)
                # proto_embedding = get_proto_embedding_old(spt_idx_list, embeddings)
                # proto_embedding = get_degree_proto_embedding(spt_idx_list, embeddings, train_degree_list)
                qry_embedding = embeddings[qry_idx]
                # -------------------------------------------------------

                # -----[Equal few-shot sampling]-------------------------
                # spt_idx, qry_idx = get_spt_qry_id_old(train_class_by_id, 5)
                # spt_embedding = embeddings[spt_idx]
                # spt_embedding = spt_embedding.view([class_num, 5, -1])
                # proto_embedding = spt_embedding.sum(1)
                # qry_embedding = embeddings[qry_idx]
                # -------------------------------------------------------

                dists = euclidean_dist(qry_embedding, proto_embedding)
                # dists = inner_product_dist(qry_embedding, proto_embedding)
                # dists = ball_dist(qry_embedding, proto_embedding)

                labels = y_train[qry_idx]

                # loss_train = F.cross_entropy(-dists, labels)
                # loss_circle = cos_dist_loss(proto_embedding)

                # loss_train = loss_train + 0.1 * loss_circle

                # loss_train.backward()

                p_loss = ProximityLoss(qry_embedding, proto_embedding, labels)
                u_loss = UniformityLoss(proto_embedding)

                loss_train = p_loss + 0.1 * u_loss
                loss_train.backward()
                optimizer_encoder.step()
                optimizer_proto.step()

                acc_train = accuracy(-dists, labels)

                if episode % 5 == 0:    
                    # validation
                    encoder.eval()
                    proto_model.eval()
                    with torch.no_grad():
                        val_embeddings = encoder(x_val, edge_index_val)
                        
                        val_spt_idx_list = get_test_spt_list(y_val, val_mapping_idx)
                        val_qry_idx = val_mapping_idx

                        # spt_idx_list, qry_idx = get_unbalanced_spt_qry_id(val_class_by_id, k_spt_max=20)

                        proto_embedding = get_proto_embedding(proto_model, val_spt_idx_list, val_embeddings, val_degree_list)
                        # proto_embedding = get_proto_embedding_old(val_spt_idx_list, val_embeddings)
                        # proto_embedding = get_degree_proto_embedding(val_spt_idx_list, val_embeddings, val_degree_list)
                        qry_embedding = val_embeddings[val_qry_idx]
                        dists = euclidean_dist(qry_embedding, proto_embedding)
                        # dists = inner_product_dist(qry_embedding, proto_embedding)
                        # dists = ball_dist(qry_embedding, proto_embedding)

                        labels = y_val[val_qry_idx]

                    acc_val = accuracy(-dists, labels)

                    if acc_val > best_base_acc:
                        best_base_acc = acc_val
                        torch.save(encoder.state_dict(), pkl_filename)
                        torch.save(proto_model.state_dict(), proto_pkl_filename)
                        best_episode = episode
                        val_proto = proto_embedding
                        val_qry = qry_embedding
                        val_gt = labels
                
            print("[Base training] Best acc: {} @episode {}".format(best_base_acc, best_episode))
            acc_list.append(best_base_acc)

            proto_embedding_filename = result_path + "{}_proto_embedding_stream{}.npy".format(args.model, 0)
            qry_embedding_filename = result_path + "{}_qry_embedding_stream{}.npy".format(args.model, 0)
            groundtruth_filename = result_path + "{}_groundtruth_stream{}.npy".format(args.model, 0)
            np.save(proto_embedding_filename, val_proto.detach().cpu().numpy())
            np.save(qry_embedding_filename, val_qry.detach().cpu().numpy())
            np.save(groundtruth_filename, val_gt.detach().cpu().numpy())

            del x_train, y_train, edge_index_train, x_val, y_val, edge_index_val
            del encoder
        
        if args.start > 0:
            class_num += np.sum(np.array(n_novel_list[0: args.start]))

        # grandpa_encoder = get_model(args.model, model_args['in_feature'], model_args['hidden_feature']).to(device)
        # grandpa_encoder.load_state_dict(torch.load(pkl_filename))

        for j in range(args.start, len(n_novel_list)):

            proto_embedding_filename = result_path + "{}_proto_embedding_stream{}.npy".format(args.model, j+1)
            qry_embedding_filename = result_path + "{}_qry_embedding_stream{}.npy".format(args.model, j+1)
            groundtruth_filename = result_path + "{}_groundtruth_stream{}.npy".format(args.model, j+1)

            print("============ Streaming {} ============".format(j))
            ft_encoder = get_model(args.model, model_args['in_feature'], model_args['hidden_feature']).to(device)
            ft_encoder.load_state_dict(torch.load(pkl_filename))

            ft_proto_model = ProtoRepre(model_args['hidden_feature']).to(device)
            ft_proto_model.load_state_dict(torch.load(proto_pkl_filename))

            # ============================================================
            teacher_encoder = get_model(args.model, model_args['in_feature'], model_args['hidden_feature']).to(device)
            teacher_encoder.load_state_dict(torch.load(pkl_filename))
            teacher_proto_model = ProtoRepre(model_args['hidden_feature']).to(device)
            teacher_proto_model.load_state_dict(torch.load(proto_pkl_filename))

            teacher_encoder.eval()
            teacher_proto_model.eval()
            # ============================================================

            # if j == 0:
            #     optimizer_proto = optim.Adam(proto_model.parameters(),
            #            lr=args.lr, weight_decay=args.weight_decay)
            # else:
            #     proto_model.load_state_dict(torch.load(proto_pkl_filename))
            #     optimizer_proto = optim.Adam(proto_model.parameters(),
            #            lr=args.ft_lr, weight_decay=args.weight_decay)
            

            print("Novel class in streaming ...")
            teacher_class_num = class_num
            class_num = class_num + n_novel_list[j]

            print("teacher_class_num is", teacher_class_num)

            best_ft_test_acc = -1
            optimizer_encoder = optim.Adam(ft_encoder.parameters(),
                        lr=args.ft_lr, weight_decay=args.weight_decay)
            optimizer_proto = optim.Adam(ft_proto_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
            
            alpha = torch.tensor([1.0] * data_args['n_base'] + [0.5] * (teacher_class_num - data_args['n_base']) + [0.2] * n_novel_list[j])
            focal_loss = FocalLoss(class_num, gamma=0, alpha=alpha)

            stream_npz_iter_filename = "{}_{}base_{}idx_{}novel_{}shot_stream.npz".format(data_args['name'], data_args['n_base'], j, n_novel_list[j], args.k_shot)
            npz_file = np.load(processed_data_dir + stream_npz_iter_filename, allow_pickle=True)

            x_ft, y_ft, edge_index_ft = torch.tensor(npz_file['ft_x_feature']).to(device), torch.tensor(npz_file['ft_y_label']).to(device), torch.tensor(npz_file['ft_edge_index']).to(device)
            ft_class_by_id = get_class_by_id(y_ft)
            ft_node_num = y_ft.shape[0]
            ft_degree_list = get_degree_list(edge_index_ft, ft_node_num)

            x_test, y_test, edge_index_test = torch.tensor(npz_file['test_x_feature']).to(device), torch.tensor(npz_file['test_y_label']).to(device), torch.tensor(npz_file['test_edge_index']).to(device)
            test_mapping_idx, ft_mapping_idx = npz_file['test_mapping_idx'], npz_file['ft_mapping_idx']
            test_node_num = y_test.shape[0]
            test_degree_list = get_degree_list(edge_index_test, test_node_num)

            for episode in range(args.ft_episodes):

                ft_encoder.train()
                ft_proto_model.train()
                optimizer_encoder.zero_grad()
                optimizer_proto.zero_grad()
                # optimizer_proto.zero_grad()
                ft_embeddings = ft_encoder(x_ft, edge_index_ft)
                
                # ------Normal---------
                # spt_idx_list, qry_idx = get_spt_qry_id(ft_class_by_id, 5)
                # # spt_embedding = ft_embeddings[spt_idx]
                # # print("spt_embedding shape is", spt_embedding.shape)
                # # spt_embedding = spt_embedding.view([class_num, 5, -1])
                # # proto_embedding = spt_embedding.sum(1)
                # proto_embedding = get_proto_embedding(ft_proto_model, spt_idx_list, ft_embeddings, ft_degree_list)
                # qry_embedding = ft_embeddings[qry_idx]

                # teacher_embeddings = teacher_encoder(x_ft, edge_index_ft)
                # # teacher_spt_embedding = teacher_embeddings[spt_idx].view([class_num, 5, -1])
                # teacher_proto_embedding = get_proto_embedding(teacher_proto_model, spt_idx_list, teacher_embeddings, ft_degree_list)
                # ---------------------

                # ------不平衡spt样本----
                # spt_idx_list, qry_idx = get_ft_spt_qry_id(ft_class_by_id, 5)
                # proto_embedding = get_proto_embedding(spt_idx_list, ft_embeddings)
                # qry_embedding = ft_embeddings[qry_idx]
                # ---------------------

                # -----不平衡spt样本+平衡qry样本---------------
                spt_idx_list, _ = get_ft_spt_qry_id(ft_class_by_id, k_spt=5, k_qry=None)
                proto_embedding = get_proto_embedding(ft_proto_model, spt_idx_list, ft_embeddings, ft_degree_list)
                # proto_embedding = get_proto_embedding_old(spt_idx_list, ft_embeddings)
                # proto_embedding = get_degree_proto_embedding(spt_idx_list, ft_embeddings, ft_degree_list)
                qry_idx =  ft_mapping_idx 
                # qry_idx = random.sample(list(ft_mapping_idx), int(random.random() * len(ft_mapping_idx)))
                random_sample_idx = random.sample(list(np.arange(y_ft.shape[0])), int(len(ft_mapping_idx)))
                qry_idx = np.concatenate((qry_idx, random_sample_idx))
                # qry_idx = random.sample(list(np.arange(y_ft.shape[0])), 2 * int(len(ft_mapping_idx)))
                qry_embedding = ft_embeddings[qry_idx]
                teacher_embeddings = teacher_encoder(x_ft, edge_index_ft)
                # teacher_proto_embedding = get_proto_embedding_old(spt_idx_list, teacher_embeddings)
                teacher_proto_embedding = get_proto_embedding(teacher_proto_model, spt_idx_list, teacher_embeddings, ft_degree_list)
                # teacher_proto_embedding = get_degree_proto_embedding(spt_idx_list, teacher_embeddings, ft_degree_list)
                # --------------------------------------------

                # =================================================
                # Use gate model to update prototypes
                # proto_embedding = proto_model(proto_embedding, teacher_proto_embedding, teacher_class_num)
                # =================================================

                dists = euclidean_dist(qry_embedding, proto_embedding)
                # dists = inner_product_dist(qry_embedding, proto_embedding)
                # dists = ball_dist(qry_embedding, proto_embedding)
                labels = y_ft[qry_idx]

                # =======================================================
                
                # grandpa_embeddings = grandpa_encoder(x_ft, edge_index_ft)
                # teacher_spt_embedding = teacher_embeddings[spt_idx].view([class_num, 5, -1])
                # teacher_proto_embedding = teacher_spt_embedding.sum(1)

                
                # grandpa_proto_embedding = get_proto_embedding(spt_idx_list, grandpa_embeddings)

                teacher_qry_embedding = teacher_embeddings[qry_idx]
                # grandpa_qry_embedding = grandpa_embeddings[qry_idx]
                teacher_dists = euclidean_dist(teacher_qry_embedding, teacher_proto_embedding)
                # teacher_dists = inner_product_dist(teacher_qry_embedding, teacher_proto_embedding)
                # teacher_dists = ball_dist(teacher_qry_embedding, teacher_proto_embedding)

                # grandpa_dists = euclidean_dist(grandpa_qry_embedding, grandpa_proto_embedding)
                # loss_knowledge = knowledge_dist_loss(
                # dists[0: data_args['n_base']], 
                # teacher_dists[0: data_args['n_base']]
                # )   

                # print("dists shape is {}. teacher dists shape is {}".format(dists.shape, teacher_dists.shape))

                # loss_knowledge = knowledge_dist_loss(
                #     dists[:, 0: teacher_class_num], 
                #     teacher_dists[:, 0: teacher_class_num], T=2
                # ) 

                # loss_knowledge_grandpa = knowledge_dist_loss(
                # dists[0: data_args['n_base']], 
                # grandpa_dists[0: data_args['n_base']]
                # ) 
                # =======================================================

                # loss_proximity = F.cross_entropy(-dists, labels)
                # loss_proximity = focal_loss(-dists, labels)
                # class_dist_loss = base_novel_dist_loss(proto_embedding[0: data_args['n_base']], proto_embedding[data_args['n_base']:])
                # class_dist_loss = base_novel_dist_loss(
                #     proto_embedding[0: class_num],
                #     proto_embedding[class_num:]
                # )
                # print("loss_ft loss {}, loss_knowledge loss {}".format(loss_ft, loss_knowledge * 0.1))
                # loss_ft = loss_ft + 0.1 * loss_knowledge + 1 / class_dist_loss

                # loss_circle = cos_dist_loss(proto_embedding)

                # (1 / math.log(10.1+j, 10)) * 
                # 0.2 * loss_circle + 
                # loss_ft = loss_ft + 0.2 * loss_circle + args.loss_ratio * loss_knowledge
                # loss_knowledge = proto_knowledge_loss(proto_embedding, teacher_proto_embedding, teacher_class_num)


                # print("loss_ft: {}, loss_knowledge: {}, loss_circle: {}".format(
                #     loss_ft, args.loss_ratio * loss_knowledge, 0.2 * loss_circle
                # ))
                # loss_separ = proto_separ_loss(proto_embedding, teacher_class_num)

                # loss_ft = loss_ft + 0.2 * loss_circle + args.loss_ratio * loss_knowledge

                # loss_ft = loss_proximity + loss_circle + loss_separ +  10000 * loss_knowledge

                p_loss = ProximityLoss(qry_embedding, proto_embedding, labels)
                # p_loss = ProximityLoss(qry_embedding, proto_embedding, labels, focal=True, class_num=class_num, base_class_num=teacher_class_num)
                # p_loss = focal_loss(-dists, labels)
                u_loss = UniformityLoss(proto_embedding)
                s_loss = SeparabilityLoss(proto_embedding, teacher_class_num)
                m_loss = knowledge_dist_loss(
                    dists[:, 0: teacher_class_num], 
                    teacher_dists[:, 0: teacher_class_num], T=2
                ) 
                # m_loss = MemorabilityLoss(proto_embedding, teacher_proto_embedding, teacher_class_num)

                
                # ----------------------------
                # cora_ml #
                # loss_ft = (1 / math.log(5.1+j, 5)) * 1.5 * p_loss + 10 * m_loss + s_loss + u_loss
                # loss_ft = 1.5 * p_loss + 10 * m_loss + s_loss + u_loss
                # ----------------------------
                # loss_ft = 1.5 * p_loss + 2 * s_loss + u_loss + 2 * m_loss # latest
                # loss_ft = 1.5 * p_loss + s_loss + 0.8 * u_loss + 2 * m_loss # test
                # 2 * m_loss + 
                # ----------------------------
                # Amazon #
                # loss_ft = 1.5 * p_loss + 10 * m_loss + 10 * s_loss + u_loss
                # loss_ft = 1.5 * p_loss + 10 * m_loss + 2 * s_loss + u_loss
                # ----------------------------

                # ----------------------------
                # Flickr #
                # loss_ft =  (1 / math.log(5.1+j, 5)) * 1.5 * p_loss + 10 * m_loss + 2 * s_loss + u_loss
                loss_ft = 1.5 * p_loss + 2 * m_loss + 2 * s_loss + u_loss
                # ----------------------------

                # ----------------------------
                # Cora #
                # loss_ft = (1 / math.log(5.1+j, 5)) * 1.5 * p_loss + 10 * m_loss + 2 * s_loss + 0.3 * u_loss
                # focal
                # loss_ft = (1 / math.log(10.1+j, 10)) * 3 * p_loss + 10 * m_loss + 10 * s_loss + u_loss
                # none
                # loss_ft = 3 * p_loss + 10 * m_loss + 10 * s_loss + 1.2 * u_loss
                # test 2 (saved)
                # loss_ft = 2 * p_loss + 5 * m_loss + 2 * s_loss + u_loss
                # test 2
                # loss_ft = 4 * p_loss + 15 * m_loss + 2 * s_loss + u_loss
                # ----------------------------

                # loss_ft = p_loss + 10 * m_loss + 2 * s_loss + 0.6 * u_loss
                #  + u_loss + s_loss + 10 * m_loss
                # loss_ft = p_loss


                # loss_ft = (1 / math.log(10.1+j, 10)) * 1.2 * p_loss + u_loss + 10 * s_loss + 10 * m_loss


                # loss_ft = p_loss + u_loss + 1.2 * s_loss + 10 * m_loss

                if args.tf:
                    writer.add_scalar('total_loss_'+str(j), loss_ft, episode)
                    writer.add_scalar('proximity_loss_'+str(j), p_loss, episode)
                    writer.add_scalar('uniformity_loss_'+str(j), u_loss, episode)
                    writer.add_scalar('separability_loss_'+str(j), s_loss, episode)
                    writer.add_scalar('memorability_loss_'+str(j), 10 * m_loss, episode)

                # + args.loss_ratio * loss_knowledge
                #  + 0.2 * loss_circle
                # loss_ft = loss_ft + args.loss_ratio * loss_knowledge

                # print("loss_ft loss {}, class_dist_loss loss {}".format(loss_ft, 0.5 / class_dist_loss))
                # loss_ft = loss_ft + 0.5 / class_dist_loss

                loss_ft.backward()
                optimizer_encoder.step()
                optimizer_proto.step()

                ft_encoder.eval()
                ft_proto_model.eval()
                with torch.no_grad():
                    test_embedding = ft_encoder(x_test, edge_index_test)
                    # teacher_embeddings = teacher_encoder(x_test, edge_index_test)
                    
                    test_spt_idx_list = get_test_spt_list(y_test, test_mapping_idx)
                    # for i, spt_idx in enumerate(test_spt_idx_list):
                    #     print("label {}, spt node num is {}".format(i, len(spt_idx)))
    
                    test_qry_idx = test_mapping_idx

                    proto_embedding = get_proto_embedding(ft_proto_model, test_spt_idx_list, test_embedding, test_degree_list)
                    # proto_embedding = get_proto_embedding_old(test_spt_idx_list, test_embedding)
                    # proto_embedding = get_degree_proto_embedding(test_spt_idx_list, test_embedding, test_degree_list)
                    # teacher_proto_embedding = get_degree_proto_embedding(test_spt_idx_list, teacher_embeddings, test_degree_list)

                    qry_embedding = test_embedding[test_qry_idx]
                    # print("proto_embedding shape is", proto_embedding.shape)
                    # print("qry_embedding shape is", qry_embedding.shape)
                
                # =================================================
                # Use gate model to update prototypes
                # proto_embedding = proto_model(proto_embedding, teacher_proto_embedding, teacher_class_num)
                # =================================================

                dists = euclidean_dist(qry_embedding, proto_embedding)
                # dists = inner_product_dist(qry_embedding, proto_embedding)

                # dists = ball_dist(qry_embedding, proto_embedding)

                labels = y_test[test_qry_idx]

                acc_test = accuracy(-dists, labels)

                if acc_test > best_ft_test_acc:
                    best_ft_test_acc = acc_test
                    best_ft_episode = episode
                    test_groundtruth = labels
                    test_prediction = -dists
                    test_proto = proto_embedding
                    best_qry_embedding = qry_embedding
                    torch.save(ft_encoder.state_dict(), pkl_filename)
                    torch.save(ft_proto_model.state_dict(), proto_pkl_filename)

            acc_list.append(best_ft_test_acc)
            print("[Finetune] best test acc after ft is {}, @epoch {}".format(best_ft_test_acc, best_ft_episode))
            test_groundtruth = test_groundtruth.detach().cpu().numpy()
            test_prediction = test_prediction.max(1)[1].detach().cpu().numpy()

            test_proto = test_proto.detach().cpu().numpy()
            np.save(proto_embedding_filename, test_proto)
            best_qry_embedding = best_qry_embedding.detach().cpu().numpy()
            np.save(qry_embedding_filename, best_qry_embedding)
            np.save(groundtruth_filename, test_groundtruth)
            # print("Save at", proto_embedding_filename)
            conf_matrix = confusion_matrix(test_groundtruth, test_prediction)
            print(conf_matrix)

            # if j == len(n_novel_list) - 1:
            #     confusion_matrix_filename = result_path + "{}_confusion_matrix_{}.npy".format('last', args.model)
            #     np.save(confusion_matrix_filename, conf_matrix)

            del x_ft, y_ft, edge_index_ft, x_test, y_test, edge_index_test

        total_acc_list.append(acc_list)
    
    avg_acc = np.mean(np.array(total_acc_list), 0)
    std_acc = np.std(np.array(total_acc_list), 0)
    print(total_acc_list)
    print("------------------")
    print("Avg:", avg_acc * 100)
    print("STD:", std_acc * 100)
    print(args.memo)
    print("------------------")
    for i in range(len(avg_acc)):
        print("{}±{}%".format(round(avg_acc[i] * 100, 2), round(std_acc[i] * 100, 2)))