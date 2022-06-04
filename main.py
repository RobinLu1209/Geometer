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

parser = argparse.ArgumentParser(description='Geometer')
parser.add_argument('--config_filename', default='config/config_cora_ml_stream.yaml', type=str)
parser.add_argument('--model', default='gat', type=str)
parser.add_argument('--iteration', default=5, type=int, help='times of repeated experiments')
parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train.')
parser.add_argument('--ft_episodes', type=int, default=200, help='Number of episodes to train.')
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

    processed_data_dir = "dataset/{}_stream/".format(data_args['name'])

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
                embeddings = encoder(x_train, edge_index_train)

                spt_idx_list, qry_idx = get_unbalanced_spt_qry_id(train_class_by_id, k_spt_max=args.k_spt_max, k_qry=30)
                proto_embedding = get_proto_embedding(proto_model, spt_idx_list, embeddings, train_degree_list)
                qry_embedding = embeddings[qry_idx]

                dists = euclidean_dist(qry_embedding, proto_embedding)
                labels = y_train[qry_idx]

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

                        proto_embedding = get_proto_embedding(proto_model, val_spt_idx_list, val_embeddings, val_degree_list)
                        qry_embedding = val_embeddings[val_qry_idx]
                        dists = euclidean_dist(qry_embedding, proto_embedding)
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
            # =============================================================            

            print("Novel class in streaming ...")
            teacher_class_num = class_num
            class_num = class_num + n_novel_list[j]

            print("teacher_class_num is", teacher_class_num)

            best_ft_test_acc = -1
            optimizer_encoder = optim.Adam(ft_encoder.parameters(),
                        lr=args.ft_lr, weight_decay=args.weight_decay)
            optimizer_proto = optim.Adam(ft_proto_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

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
                
                spt_idx_list, _ = get_ft_spt_qry_id(ft_class_by_id, k_spt=5, k_qry=None)
                proto_embedding = get_proto_embedding(ft_proto_model, spt_idx_list, ft_embeddings, ft_degree_list)
                qry_idx =  ft_mapping_idx 
                random_sample_idx = random.sample(list(np.arange(y_ft.shape[0])), int(len(ft_mapping_idx)))
                qry_idx = np.concatenate((qry_idx, random_sample_idx))
                qry_embedding = ft_embeddings[qry_idx]
                teacher_embeddings = teacher_encoder(x_ft, edge_index_ft)
                teacher_proto_embedding = get_proto_embedding(teacher_proto_model, spt_idx_list, teacher_embeddings, ft_degree_list)

                dists = euclidean_dist(qry_embedding, proto_embedding)
                labels = y_ft[qry_idx]

                teacher_qry_embedding = teacher_embeddings[qry_idx]
                teacher_dists = euclidean_dist(teacher_qry_embedding, teacher_proto_embedding)

                p_loss = ProximityLoss(qry_embedding, proto_embedding, labels)
                u_loss = UniformityLoss(proto_embedding)
                s_loss = SeparabilityLoss(proto_embedding, teacher_class_num)
                m_loss = knowledge_dist_loss(
                    dists[:, 0: teacher_class_num], 
                    teacher_dists[:, 0: teacher_class_num], T=2
                ) 
                
                loss_ft = 1.5 * p_loss + 2 * m_loss + 2 * s_loss + u_loss

                if args.tf:
                    writer.add_scalar('total_loss_'+str(j), loss_ft, episode)
                    writer.add_scalar('proximity_loss_'+str(j), p_loss, episode)
                    writer.add_scalar('uniformity_loss_'+str(j), u_loss, episode)
                    writer.add_scalar('separability_loss_'+str(j), s_loss, episode)
                    writer.add_scalar('memorability_loss_'+str(j), 10 * m_loss, episode)


                loss_ft.backward()
                optimizer_encoder.step()
                optimizer_proto.step()

                ft_encoder.eval()
                ft_proto_model.eval()
                with torch.no_grad():
                    test_embedding = ft_encoder(x_test, edge_index_test)
                    test_spt_idx_list = get_test_spt_list(y_test, test_mapping_idx)
                    test_qry_idx = test_mapping_idx
                    proto_embedding = get_proto_embedding(ft_proto_model, test_spt_idx_list, test_embedding, test_degree_list)
                    qry_embedding = test_embedding[test_qry_idx]

                dists = euclidean_dist(qry_embedding, proto_embedding)
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