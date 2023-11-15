import random
import numpy as np                  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from tqdm import tqdm
import pickle


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.n_users = n_users

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, interact_mat, region_weight_matrix,
                weight):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        
        """Region aggregate"""
#         # beijing
#         new_entity_vectors = entity_emb.clone().detach()
#         new_entity_vectors[29667:31567, :] = torch.matmul(region_weight_matrix, new_entity_vectors[29667:31567, :])
#         entity_emb = entity_emb * 0.7 + new_entity_vectors * 0.3
        
        # shanghai
        new_entity_vectors = entity_emb.clone().detach()
        new_entity_vectors[42033:44630, :] = torch.matmul(region_weight_matrix, new_entity_vectors[42033:44630, :])
        entity_emb = entity_emb * 0.8 + new_entity_vectors * 0.2 # best parameter on shanghai
#         entity_emb = entity_emb * 0.5 + new_entity_vectors * 0.5 # hyper-parameter study, alpha
        
        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        
        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
#         user_r_agg = torch.sparse.mm(interact_mat[1], entity_emb)

        return entity_agg, user_agg#, user_r_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_relations, interact_mat, region_weight_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.region_weight_mat = region_weight_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]


        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, region_weight_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = 0#self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                 edge_index, edge_type, interact_mat, region_weight_mat,
                                                 self.weight)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.emb_size = args_config.dim
        self.dataset = args_config.dataset
        print(args_config.dataset)
        if args_config.dataset == 'beijing':
            self.n_regions = 1900
            self.user_region_emb = torch.zeros((self.n_users, self.emb_size)).to(self.device).detach()
#         elif args_config.dataset = 'shanghai':
#             self.n_regions = 1800
        else:
            self.n_regions = 2597
            self.user_region_emb = torch.zeros((self.n_users, self.emb_size)).to(self.device).detach()

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        
        self.region_weight_mat, self.poi_region_mat = self._compute_region_weight_matrix()
        self.user_region_dict, self.user_region_pair_dict = self._generate_user_region_dict(args_config)
        self.gcn = self._init_model()
                
        
    def _compute_region_weight_matrix(self, k=5):
        """
        k: nearest neighbor number"""
        if self.dataset == 'beijing':
            region_distance_mat = torch.FloatTensor(pickle.load(open('./region_distance_matrix.pickle3', 'rb'))).to(self.device)
            poi_region_mat = torch.LongTensor(pickle.load(open('./filter_mat_region.pickle3', 'rb'))).to(self.device).reshape(-1, 1)
        else:
            region_distance_mat = torch.FloatTensor(pickle.load(open('./region_distance_matrix.pickle3', 'rb'))).to(self.device)
            poi_region_mat = torch.LongTensor(pickle.load(open('./filter_mat_region.pickle3', 'rb'))).to(self.device).reshape(-1, 1)
            
        region_weight_mat = torch.zeros((self.n_regions, self.n_regions)).to(self.device)
        values, indices = torch.topk(-region_distance_mat, k, dim=1)
        
        
        ### Version 1: sim = 1/ (d_ij)
        for row in range(self.n_regions):
            idx = torch.LongTensor([row] * k)
            region_weight_mat[idx, indices[row]] = - 1 / values[row]
        region_weight_mat = region_weight_mat / torch.sum(region_weight_mat, dim=1).reshape((-1, 1))
        
        
#         ### Version 2: sim = 1/k
#         for row in range(self.n_regions):
#             idx = torch.LongTensor([row] * k)
#             region_weight_mat[idx, indices[row]] = - 1 / k
#         region_weight_mat = region_weight_mat / torch.sum(region_weight_mat, dim=1).reshape((-1, 1))
        return region_weight_mat, poi_region_mat

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         region_weight_mat=self.region_weight_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)
    
    def _generate_user_region_dict(self, args):
        def read_cf(poi_region_mat, file_name):
            inter_mat = list()
            user_inter_dict = dict()
            user_region_dict = dict()
            user_region_pair_dict = dict()
            poi_region_mat = self.poi_region_mat.reshape(-1, ).detach().tolist()
            lines = open(file_name, "r").readlines()
            for l in lines:
                user, poi = l.strip('\n').split('\t')
                user, region = int(user), poi_region_mat[int(poi)]
                if user not in user_inter_dict:
                    user_inter_dict[user] = []
                user_inter_dict[user].append(region)
            user_num = len(user_inter_dict)
            for user in range(user_num):
                u_id, r_ids = user, list(set(user_inter_dict[user]))
                user_region_dict[u_id] = r_ids
                for r_id in r_ids:
                    user_region_pair_dict[(u_id, r_id)] = 1
            return user_region_dict, user_region_pair_dict #np.array(inter_mat)
        file_path = args.data_path+args.dataset+'/train_mini.txt'
        user_region_dict, user_region_pair_dict = read_cf(self.poi_region_mat, file_path)
        return user_region_dict, user_region_pair_dict
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)
    
#     def compute_user_region_vector(self, users, user_gcn_emb, entity_gcn_emb):
        
#         u_e = user_gcn_emb[users]
#         users = users.detach().cpu().numpy().tolist()
#         user_num = len(users)
#         user_region_emb = torch.zeros((user_num, self.emb_size)).to(self.device)
#         user_heads, user_heads_pooling, region_tails = [], [], []
#         tmp_user_map_dict = {}
#         for uid in range(user_num):
#             tmp_user_map_dict[users[uid]] = uid
#             regions = self.user_region_dict[users[uid]]
#             region_tails.extend(regions)
#             user_heads.extend([users[uid]] * len(regions))
#             user_heads_pooling.extend([uid] * len(regions))
#         user_heads_pooling = torch.LongTensor(user_heads_pooling).to(self.device).reshape(-1, 1)
#         user_heads = torch.LongTensor(user_heads).to(self.device).reshape(-1, 1)
#         region_tails = torch.LongTensor(region_tails).to(self.device).reshape(-1, 1)
#         user_emb_r = torch.sum(user_gcn_emb[user_heads, :], 1)
#         region_emb_r = torch.sum(entity_gcn_emb[region_tails, :], 1) 
# #             print(user_emb_r.shape, region_emb_r.shape)
#         inner_products = torch.exp(torch.sum(user_emb_r * region_emb_r, 1)).reshape(-1, 1)
# #             print(inner_products.shape, user_heads_pooling.shape)
#         normalize_factor = scatter_sum(inner_products, user_heads_pooling, dim_size=u_e.shape[0], dim=0)
#         normalize_vector = normalize_factor[user_heads_pooling].reshape(-1, 1)
# #   
# #             print(region_emb_r.shape, normalize_vector.shape)
#         region_emb_r = region_emb_r * normalize_vector
#         user_r_agg = scatter_mean(region_emb_r, user_heads_pooling, dim_size=u_e.shape[0], dim=0)
#         return user_r_agg
        
    
    def update_user_region_embedding(self):
        print('Updating user-region preference embedding now...')
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     self.region_weight_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        BATCH_SIZE = 2048
        BATCH_NUM = self.n_users // BATCH_SIZE + 1
        for u_batch_id in tqdm(range(BATCH_NUM)):
            start = u_batch_id * BATCH_SIZE
            end = min((u_batch_id + 1) * BATCH_SIZE, self.n_users)
            u_e = user_gcn_emb[start: end]
            user_heads, region_tails = [], []
            tmp_user_map_dict = {}
            for uid in range(start, end):
                tmp_user_map_dict[uid] = uid - start
                regions = self.user_region_dict[uid]
                region_tails.extend(regions)
                user_heads.extend([uid] * len(regions))
            user_heads_pooling = torch.LongTensor([tmp_user_map_dict[i] for i in user_heads]).to(self.device).reshape(-1, 1)
            user_heads = torch.LongTensor(user_heads).to(self.device).reshape(-1, 1)
            region_tails = torch.LongTensor(region_tails).to(self.device).reshape(-1, 1)
            user_emb_r = torch.sum(user_gcn_emb[user_heads, :], 1)
            region_emb_r = torch.sum(entity_gcn_emb[region_tails, :], 1) 
#             print(user_emb_r.shape, region_emb_r.shape)
            inner_products = torch.exp(torch.sum(user_emb_r * region_emb_r, 1)).reshape(-1, 1)
#             print(inner_products.shape, user_heads_pooling.shape)
            normalize_factor = scatter_sum(inner_products, user_heads_pooling, dim_size=u_e.shape[0], dim=0)
            normalize_vector = normalize_factor[user_heads_pooling].reshape(-1, 1)
#   
#             print(region_emb_r.shape, normalize_vector.shape)
            region_emb_r = region_emb_r * normalize_vector
            user_r_agg = scatter_mean(region_emb_r, user_heads_pooling, dim_size=u_e.shape[0], dim=0)
#             print(user_r_agg.shape, self.user_region_emb[start:end, :].shape)
            self.user_region_emb[start:end, :] = user_r_agg
        self.user_region_emb = self.user_region_emb.detach()
        
    def forward(self, batch=None):
        user = batch['users']
#         print(user)
#         print(min(user), max(user), max(user)-min(user))
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'] 

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     self.region_weight_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        pos_regions, neg_regions = self.poi_region_mat[pos_item], self.poi_region_mat[neg_item]
        pos_r_e, neg_r_e = entity_gcn_emb[pos_regions], entity_gcn_emb[neg_regions]
        user_r_agg = self.user_region_emb[user]
        neg_regions = neg_regions.reshape(-1, ).detach().cpu().tolist()
        user = user.detach().cpu().tolist()
        
#         print((user[:5], neg_regions[:5]))
#         print(list(self.user_region_pair_dict.keys())[:5])
#         print((0, 30948) in self.user_region_pair_dict)
        neg_region_indicator = [0 if (user[i], neg_regions[i]) in self.user_region_pair_dict  else 1.0 for i in range(len(neg_regions))]
#         print(neg_region_indicator[:50])
        neg_region_indicator = torch.FloatTensor(neg_region_indicator).reshape(-1, 1).to(self.device)
        
        
        
#         users = user.detach().cpu().numpy().tolist()
#         user_num = len(users)
#         user_heads, user_heads_pooling, region_tails = [], [], []
#         tmp_user_map_dict = {}
#         for uid in range(user_num):
#             tmp_user_map_dict[users[uid]] = uid
#             regions = self.user_region_dict[users[uid]]
#             region_tails.extend(regions)
#             user_heads.extend([users[uid]] * len(regions))
#             user_heads_pooling.extend([uid] * len(regions))
#         user_heads_pooling = torch.LongTensor(user_heads_pooling).to(self.device).reshape(-1, 1)
#         user_heads = torch.LongTensor(user_heads).to(self.device).reshape(-1, 1)
#         region_tails = torch.LongTensor(region_tails).to(self.device).reshape(-1, 1)
#         user_emb_r = torch.sum(user_gcn_emb[user_heads, :], 1)
#         region_emb_r = torch.sum(entity_gcn_emb[region_tails, :], 1) 
# #             print(user_emb_r.shape, region_emb_r.shape)
#         inner_products = torch.exp(torch.sum(user_emb_r * region_emb_r, 1)).reshape(-1, 1)
# #             print(inner_products.shape, user_heads_pooling.shape)
#         normalize_factor = scatter_sum(inner_products, user_heads_pooling, dim_size=u_e.shape[0], dim=0)
#         normalize_vector = normalize_factor[user_heads_pooling].reshape(-1, 1)
# #   
# #             print(region_emb_r.shape, normalize_vector.shape)
#         region_emb_r = region_emb_r * normalize_vector
#         user_r_agg = scatter_mean(region_emb_r, user_heads_pooling, dim_size=u_e.shape[0], dim=0)
        user_r_agg = self.user_region_emb[user]
        
        return self.create_bpr_loss(u_e, pos_e, neg_e, pos_r_e, neg_r_e , user_r_agg, neg_region_indicator)
    
    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
#         self.update_user_region_embedding()
        return self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        self.region_weight_mat,
                        mess_dropout=False, node_dropout=False)[:-1], self.user_region_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, pos_regions, neg_regions, user_regions, neg_region_indicator):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        r_pos_scores = torch.sum(torch.mul(user_regions, pos_regions), axis=1)
        r_neg_scores = torch.sum(torch.mul(user_regions, neg_regions), axis=1)
        
        beta = 0.0001
#         beta = 0 #Ablation Study, POI only 
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))\
                  -1 * torch.mean(nn.LogSigmoid()((r_pos_scores - r_neg_scores) * neg_region_indicator)) * batch_size / torch.sum(neg_region_indicator) * beta

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        lambda_reg = 1e-7
#         lambda_reg = 1e-7
        geo_regularizer = lambda_reg * (torch.norm(pos_items - pos_regions) ** 2
                           + torch.norm(neg_items - neg_regions) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

#         poi_region_loss = self.decay * geo_regularizer / batch_size
#         cor_loss = self.sim_decay * cor

        poi_region_loss = geo_regularizer / batch_size
        cor = 0
        return mf_loss + emb_loss  + poi_region_loss, mf_loss, emb_loss, cor
