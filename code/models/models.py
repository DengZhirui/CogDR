import torch.nn as nn
from tqdm import tqdm
import torch as th
import pdb
import torch.nn.functional as F
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import GraphConv
from models.layers import DGRecLayer
import numpy as np
import faiss
import math

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']

class BaseGraphModel(nn.Module):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args
        self.hid_dim = args.embed_size
        self.layer_num = args.layers
        self.graph = dataloader.train_graph
        self.user2category = dataloader.user2category
        self.user_kg_emb = dataloader.user_kg_emb
        self.item_kg_emb = dataloader.item_kg_emb
        self.user_number = dataloader.user_number
        self.item_number = dataloader.item_number

        #self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim))
        #self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('item').shape[0], self.hid_dim))
        #nn.init.xavier_normal_(self.user_embedding)
        #nn.init.xavier_normal_(self.item_embedding)#, std=0.1)
        self.user_embedding = torch.nn.Embedding.from_pretrained(self.user_kg_emb, freeze=False).cuda()
        self.item_embedding = torch.nn.Embedding.from_pretrained(self.item_kg_emb, freeze=False).cuda()

        self.predictor = HeteroDotProductPredictor()

        self.build_model()

        #self.node_features = {'user': self.user_embedding, 'item': self.item_embedding}

    def build_layer(self, idx):
        pass

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def get_embedding(self):
        h = self.node_features

        graph_user2item = dgl.edge_type_subgraph(self.graph, ['rate'])
        graph_item2user = dgl.edge_type_subgraph(self.graph, ['rated by'])

        for layer in self.layers:
            user_feat = h['user']
            item_feat = h['item']

            h_item = layer(graph_user2item, (user_feat, item_feat))
            h_user = layer(graph_item2user, (item_feat, user_feat))

            h = {'user': h_user, 'item': h_item}
        return h

    def forward(self, graph_pos, graph_neg):
        self.node_features = {'user': self.user_embedding(self.graph.nodes('user')), 'item': self.item_embedding(self.graph.nodes('item'))}
        h = self.get_embedding()

        loss_contrast_l = []
        anchor_user_embed = self.generate_similar_user_embed()
        cl_times = math.ceil(h['user'].shape[0] / self.args.cl_bs)
        for i in range(cl_times):
            loss_contrast_l.append(self.calculate_contrast_loss(h['user'][i*self.args.cl_bs:(i+1)*self.args.cl_bs], anchor_user_embed[i*self.args.cl_bs:(i+1)*self.args.cl_bs]))
        loss_contrast = torch.concat(loss_contrast_l).mean()
        score_pos = self.predictor(graph_pos, h, 'rate')
        score_neg = self.predictor(graph_neg, h, 'rate')
        return score_pos, score_neg, loss_contrast

    def get_score(self, h, users):
        user_embed = h['user'][users]
        item_embed = h['item']
        scores = torch.mm(user_embed, item_embed.t())
        return scores

class DGRec(BaseGraphModel):
    def __init__(self, args, dataloader):
        super(DGRec, self).__init__(args, dataloader)
        self.W = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))
        self.a = torch.nn.Parameter(torch.randn(self.args.embed_size))

        nn.init.xavier_normal_(self.W)#, std=0.1)
        nn.init.xavier_normal_(self.a.unsqueeze(0))#, std=0.1)

        self.WU = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size))
        self.aU = torch.nn.Parameter(torch.randn(self.args.embed_size * 2))
        nn.init.xavier_normal_(self.WU)#, std=0.1)
        nn.init.xavier_normal_(self.aU.unsqueeze(0))#, std=0.1)

        self.LeakyReLU = torch.nn.LeakyReLU(0.1)
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.complex_weight = nn.Parameter(torch.randn(1, self.args.topk // 2 + 1, self.args.embed_size, 2, dtype=torch.float32) * 0.02)

    def build_layer(self, idx):
        return DGRecLayer(self.args)

    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim = 0)
        #weight = 1/(self.layer_num+1)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(torch.matmul(weight, a), dim = 0).unsqueeze(-1)
        tensor_layers = torch.sum(tensor_layers * weight, dim = 0)
        return tensor_layers

    def get_embedding(self):
        user_embed = [self.user_embedding(self.graph.nodes('user'))]
        item_embed = [self.item_embedding(self.graph.nodes('item'))]
        #user_embed = [self.user_embedding]
        #item_embed = [self.item_embedding]
        h = self.node_features

        for layer in self.layers:

            h_item = layer(self.graph, h, ('user', 'rate', 'item'))
            h_user = layer(self.graph, h, ('item', 'rated by', 'user'))
            h = {'user': h_user, 'item': h_item}
            user_embed.append(h_user)
            item_embed.append(h_item)
        user_embed = self.layer_attention(user_embed, self.W, self.a)
        item_embed = self.layer_attention(item_embed, self.W, self.a)
        h = {'user': user_embed, 'item': item_embed}
        return h
    
    def generate_similar_user_embed(self, k=5, k1=100):

        user_embedding = self.user_embedding(self.graph.nodes('user')).cpu().detach().numpy().astype('float32')

        # retrieval similar users
        #res = faiss.StandardGpuResources()
        dim, measure = self.hid_dim, faiss.METRIC_L2
        param = 'HNSW64'
        index = faiss.index_factory(dim, param, measure)
        index.add(user_embedding)
        D, I = index.search(user_embedding, k1)
        
        #gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        #gpu_index.add(user_embedding)
        #D, I = gpu_index.search(user_embedding, k)

        # remove anchor user
        # D, I = D[:,1:], I[:,1:]
        simu_category = np.array(self.user2category[I], dtype=bool)
        currentu_category = np.array(np.tile(self.user2category,(1,k1)).reshape(self.user2category.shape[0],k1, -1), dtype=bool)
        remain_category = (currentu_category ^ simu_category) & simu_category  # [user_num, k1, category_num]
        mask = 1e20 * (1-np.array(np.sum(remain_category, axis=2), dtype=bool))
        D = D + mask
        index = np.argsort(D, axis=0)[:,:k]
        line_id = np.tile(np.arange(index.shape[0]).reshape(-1,1), (1, index.shape[1]))*index.shape[1]
        new_I = I.reshape(-1)[(line_id+index).reshape(-1)].reshape(index.shape[0], -1)

        simu_embed = torch.tensor(user_embedding[new_I]).to(self.args.device)  # [user_number, topk-1, emb_size]

        batch_size = simu_embed.shape[0]
        repeat_times = simu_embed.shape[1]

        #FFT
        x = torch.fft.rfft(simu_embed, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight[:,:self.args.topk // 2 + 1,:])
        x = x * weight
        simu_embed = torch.fft.irfft(x, n=self.args.topk, dim=1, norm='ortho')

        Wsimu_embed = torch.einsum('bi,ij->bj', simu_embed.reshape(-1, self.args.embed_size), self.WU).reshape(batch_size, -1, self.args.embed_size)
        Wu_embed = torch.einsum('bi,ij->bj', torch.tensor(user_embedding).to(self.args.device), self.WU).unsqueeze(1).repeat(1,repeat_times,1)
        W_embed = torch.cat([Wsimu_embed, Wu_embed], dim=2)  # [user_num, , emb_size * 2]
        Wa_embed = self.LeakyReLU(torch.einsum('bij,j->bi', W_embed, self.aU))
        alpha_embed = F.softmax(Wa_embed, dim=1)  # [user_num, repeat_times]
        anchor_user_embed = torch.sum(torch.einsum('bij,bi->bij',Wsimu_embed, alpha_embed), dim=1)

        return anchor_user_embed#.to(torch.float64)
    
    def calculate_contrast_loss(self, rep1, rep2):
        #print(rep1.dtype, rep2.dtype)
        batch_size = rep1.shape[0]

        rep1_norm = rep1.norm(dim=-1, keepdim=True)
        rep2_norm = rep2.norm(dim=-1, keepdim=True)
        batch_self_11 = torch.einsum("ad,bd->ab", rep1, rep1) / (
                torch.einsum("ad,bd->ab", rep1_norm, rep1_norm) + 1e-6)  # [batch, batch]
        batch_cross_12 = torch.einsum("ad,bd->ab", rep1, rep2) / (
                torch.einsum("ad,bd->ab", rep1_norm, rep2_norm) + 1e-6)  # [batch, batch]
        batch_self_11 /= self.args.temperature
        batch_cross_12 /= self.args.temperature
        batch_first = torch.cat([batch_self_11, batch_cross_12], dim=-1)
        batch_arange = torch.arange(batch_size).to(torch.cuda.current_device())
        mask = F.one_hot(batch_arange, num_classes=batch_size * 2) * -1e10
        batch_first += mask
        batch_label1 = batch_arange + batch_size

        batch_self_22 = torch.einsum("ad,bd->ab", rep2, rep2) / (
                torch.einsum("ad,bd->ab", rep2_norm, rep2_norm) + 1e-6)  # [batch, batch]
        batch_cross_21 = torch.einsum("ad,bd->ab", rep2, rep1) / (
                torch.einsum("ad,bd->ab", rep2_norm, rep1_norm) + 1e-6)  # [batch, batch]
        batch_self_22 /= self.args.temperature
        batch_cross_21 /= self.args.temperature
        batch_second = torch.cat([batch_self_22, batch_cross_21], dim=-1)
        batch_second += mask
        batch_label2 = batch_arange + batch_size

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]
        contras_loss = self.cl_loss(batch_predict, batch_label)

        return contras_loss
