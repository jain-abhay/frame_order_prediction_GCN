import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import itertools 



class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        self.in_features = in_features       

        self.out_features = out_features
        self.dropout = dropout
       
        self.W = nn.Parameter(torch.empty(size=(1,in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        

    

    def forward(self, input, adj , h0 , lamda, alpha, l):
        # import 
        
        beta = math.log(lamda/l+1) 

        hi = torch.matmul(adj, input)       
        res = (1-alpha)*hi+alpha*h0

        # hi = (1-alpha)*input+alpha*h0
        # res = torch.matmul(adj, hi)

        output = beta*torch.matmul(res, self.weight)+(1-beta)*res 

        return output



class GCNOF(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nfinal, nadj, dropout, lamda, alpha, variant):
        super(GCNOF, self).__init__()
        self.norm_mode = 'PN'
        self.norm_scale = 1

        self.conv_enc = nn.ModuleList()
        for _ in range(nlayers):
            self.conv_enc.append(GraphConvolution(nhidden, nhidden, dropout))

        self.conv_dec = nn.ModuleList()
        for _ in range(nlayers):
            self.conv_dec.append(GraphConvolution(nhidden, nhidden, dropout))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nhidden))
        self.fcs.append(nn.Linear(nadj + 2*nfeat, nhidden))

        self.fcs.append(nn.Linear(nfinal*nlayers, nfinal))
        # for _ in range(nlayers):
        #     self.fcs.append(nn.Linear(nhidden, nfinal))
        self.out = nn.Linear(nhidden, nfinal)

        self.params1 = list(self.conv_enc.parameters())
        self.params2 = list(self.conv_dec.parameters())
        self.params3 = list(self.fcs.parameters())

        self.act_fn = nn.LeakyReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        # self.norm = PairNorm(self.norm_mode, self.norm_scale)
        
    def forward(self, init_feat, adj): #x.shape = [2,N,F] #adj = [1,N,N]
        _layers = []
        msof = []
        x = init_feat
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))        
        _layers.append(x)

        for i,con in enumerate(self.conv_enc):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn((con(x,adj,_layers[-1],self.lamda, self.alpha, i+1)))
                        
        x = F.dropout(x, self.dropout, training=self.training)
        x_final = self.fcs[1](x)


        return  x_final

class FOP(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nfinal, nadj, dropout, lamda, alpha, variant):
        super(FOP, self).__init__()
        self.norm_mode = 'PN'
        self.norm_scale = 1

        self.conv_dec = nn.ModuleList()
        for _ in range(nlayers):
            self.conv_dec.append(GraphConvolution(nhidden, nhidden, dropout))

        self.fcs = nn.ModuleList()        
        self.fcs.append(nn.Linear(2*nhidden, nhidden))
        
        self.out = nn.Linear(6*nhidden*nadj, nfinal)

        # self.params1 = list(self.conv_enc.parameters())
        self.params2 = list(self.conv_dec.parameters())
        self.params3 = list(self.fcs.parameters())

        self.act_fn = nn.LeakyReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.sig = nn.Sigmoid()
        
    def forward(self, in_feat, adj): #x.shape = [2,N,F] #adj = [1,N,N]
        _layers = []
        x_final = in_feat      
        idx = list(range(x_final.shape[0]))
        idx = list(itertools.combinations(idx,2))

        pair_feat = []
        for inds in idx:
            p_feat = torch.cat((x_final[inds[0]].unsqueeze(0), x_final[inds[1]].unsqueeze(0)), -1)
            pair_feat.append(p_feat)
        pair_feat = torch.cat(pair_feat,0)

        x = F.dropout(pair_feat, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))        
        _layers.append(x)

        for i,con in enumerate(self.conv_dec):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn((con(x,adj,_layers[-1],self.lamda, self.alpha, i+1)))

        x = torch.reshape(x, (1, -1))
        seq = self.sig(self.out(x))
       
        return  seq



class recon(nn.Module):
    def __init__(self, dropout, m_pos):
        super(recon, self).__init__()
        self.m = m_pos
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(8*(1+2*m_pos), 128))        
        self.fcs.append(nn.Linear(128, 256))
        self.fcs.append(nn.Linear(256, 128))
        self.fcs.append(nn.Linear(128, 64))
        self.fcs.append(nn.Linear(64, 3))
        self.params = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def positional_encoding(self,p,m):
        pi = torch.pi        
        p_final= p
        for i in range(0,m):
            p_mod= 2**i*p*pi
            p_final = torch.cat((p_final, torch.sin(p_mod), torch.cos(p_mod)), dim=-1)

        return p_final
        
    def forward(self, im, flow):
        im_feat = im[...,:-2]
        feat = torch.cat((im_feat, flow),dim=-1)
        feat = self.positional_encoding(feat,self.m)
        for i in range(len(self.fcs)-1):
            feat = self.act_fn(self.fcs[i](feat)) 
        feat = torch.tanh(self.fcs[len(self.fcs)-1](feat))                
        return feat
















# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.th = nn.Parameter(torch.empty(size=(1, 1)))
#         nn.init.xavier_uniform_(self.th.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)

#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)



#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'








# class GraphConvolution(nn.Module):

#     def __init__(self, in_features, out_features, residual=False, variant=False):
#         super(GraphConvolution, self).__init__() 
#         self.variant = variant
#         if self.variant:
#             self.in_features = 2*in_features 
#         else:
#             self.in_features = in_features

#         self.out_features = out_features
#         self.residual = residual
#         self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj , h0 , lamda, alpha, l):
#         theta = math.log(lamda/l+1)
#         hi = torch.spmm(adj, input)
#         if self.variant:
#             support = torch.cat([hi,h0],1)
#             r = (1-alpha)*hi+alpha*h0
#         else:
#             support = (1-alpha)*hi+alpha*h0
#             r = support
#         output = theta*torch.mm(support, self.weight)+(1-theta)*r
#         if self.residual:
#             output = output+input
#         return output











# class GCNII(nn.Module):
#     def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
#         super(GCNII, self).__init__()
#         self.convs = nn.ModuleList()
#         for _ in range(nlayers):
#             self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
#         self.fcs = nn.ModuleList()
#         self.fcs.append(nn.Linear(nfeat, nhidden))
#         self.fcs.append(nn.Linear(nhidden, nclass))
#         self.params1 = list(self.convs.parameters())
#         self.params2 = list(self.fcs.parameters())
#         self.act_fn = nn.ReLU()
#         self.dropout = dropout
#         self.alpha = alpha
#         self.lamda = lamda

#     def forward(self, x, adj):
#         _layers = []
#         x = F.dropout(x, self.dropout, training=self.training)
#         layer_inner = self.act_fn(self.fcs[0](x))
#         _layers.append(layer_inner)
#         for i,con in enumerate(self.convs):
#             layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#             layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
#         layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#         layer_inner = self.fcs[-1](layer_inner)
#         return F.log_softmax(layer_inner, dim=1)




# class GCNIIppi(nn.Module):
#     def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
#         super(GCNIIppi, self).__init__()
#         self.convs = nn.ModuleList()
#         for _ in range(nlayers):
#             self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
#         self.fcs = nn.ModuleList()
#         self.fcs.append(nn.Linear(nfeat, nhidden))
#         self.fcs.append(nn.Linear(nhidden, nclass))
#         self.act_fn = nn.ReLU()
#         self.sig = nn.Sigmoid()
#         self.dropout = dropout
#         self.alpha = alpha
#         self.lamda = lamda

#     def forward(self, x, adj):
#         _layers = []
#         x = F.dropout(x, self.dropout, training=self.training)
#         layer_inner = self.act_fn(self.fcs[0](x))
#         _layers.append(layer_inner)
#         for i,con in enumerate(self.convs):
#             layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#             layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
#         layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#         layer_inner = self.sig(self.fcs[-1](layer_inner))
#         return layer_inner


if __name__ == '__main__':
    pass






