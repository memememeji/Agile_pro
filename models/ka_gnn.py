import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
#import dgl
#import dgl.function as fn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.nn import global_mean_pool, global_max_pool,global_add_pool


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):     # gridsize:傅里叶级数的基函数个数（核心超参），越大拟合能力越强、计算量越大
        super(KAN_linear,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))  # 存储参数张量，并归一化
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))  # 一维张量reshape成为四维张量
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

    
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)  #重塑前张量的「总元素数」 = 重塑后目标形状outshape的「总元素数」
        return y
    



#class NaiveFourierKANLayer(nn.Module):           # 结合傅里叶KAN的信息传递机制
class NaiveFourierKANLayer(MessagePassing):
    def __init__(self, in_feats, out_feats, gridsize, addbias=True):
        # super(NaiveFourierKANLayer, self).__init__()
        super(NaiveFourierKANLayer, self).__init__(aggr='add')
        self.gridsize = gridsize
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.fouriercoeffs = nn.Parameter(torch.randn(2, out_feats, in_feats, gridsize) / 
                                          (np.sqrt(in_feats) * np.sqrt(gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))


    #def forward(self, g, x):         # g为dgl图对象，x为节点特征向量
    def forward(self, x, edge_index):
        out = self.propagate(edge_index,x=x)
        if self.addbias:
            out += self.bias
            
        return out
       
        '''
        with g.local_scope():
            g.ndata['h'] = x  # node data
            
            g.update_all(message_func=self.fourier_transform, reduce_func=fn.sum(msg='m', out='h'))
            # If there is a bias, add it after message passing
            if self.addbias:
                g.ndata['h'] += self.bias

            return g.ndata['h']
        '''

    def message(self, x_j):
        src_feat = x_j  # 对每条边的源节点特征

        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=src_feat.device), (1, 1, 1, self.gridsize)
        )
        src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)
        cos_kx = torch.cos(k * src_rshp)
        sin_kx = torch.sin(k * src_rshp)
        
        # Reshape for multiplication
        cos_kx = torch.reshape(cos_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
        sin_kx = torch.reshape(sin_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))

        # Perform Fourier transform using einsum
        m = torch.einsum("dbik,djik->bj", torch.concat([cos_kx, sin_kx], axis=0), self.fouriercoeffs)   # 对源节点特征应用傅里叶变换，然后einsum得到消息m

        return m
        

        '''
            def fourier_transform(self, edges):      # 信息函数
        src_feat = edges.src['h']  # Shape: (E, in_feats)

        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=src_feat.device), (1, 1, 1, self.gridsize))
        src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)
        cos_kx = torch.cos(k * src_rshp)
        sin_kx = torch.sin(k * src_rshp)
        
        # Reshape for multiplication
        cos_kx = torch.reshape(cos_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
        sin_kx = torch.reshape(sin_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))

        # Perform Fourier transform using einsum
        m = torch.einsum("dbik,djik->bj", torch.concat([cos_kx, sin_kx], axis=0), self.fouriercoeffs)   # 对源节点特征应用傅里叶变换，然后einsum得到消息m

        # Returning the message to be reduced
        return {'m': m}
        '''
    

'''
class KA_GNN_two(nn.Module):   # 简化版本，读出层更简单
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):  
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        #self.layers.append()
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        

        #self.layers.append(KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias))
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, out_feat, grid_feat, addbias=use_bias))


        #self.layers.append(NaiveFourierKANLayer(out_feat, out_feat, grid_feat, addbias=use_bias))
        self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
        #self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=True)
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

        
    def forward(self, g, h):
        h = self.kan_line(h)

        for i, layer in enumerate(self.layers):
            m = layer(g, h) 
            h = nn.functional.leaky_relu(torch.add(m, h))
        
        if self.pooling == 'avg':
            y = self.avgpool(g, h)

        elif self.pooling == 'max':
            y = self.maxpool(g, h)
            
        
        elif self.pooling == 'sum':
            y = self.sumpool(g, h)


        else:
            print('No pooling found!!!!')

        out = self.Readout(y)    
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
       
        return self.parameters()
    
'''


class KA_GNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
       
        self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        self.linear = KAN_linear(hidden_feat, out, grid_feat, addbias=use_bias)

        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        #nn.Sigmoid(),
                        self.leaky_relu,
                        self.linear_2,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

# Args:x: 节点特征 [num_nodes, in_feat]
# edge_index: 边索引 [2, num_edges]
# batch: 批次索引 [num_nodes] - 指示每个节点属于哪个图 
    def forward(self, x, edge_index, batch):

        h = self.kan_line(x)
        
        for i, layer in enumerate(self.layers):
            # PyG 的图卷积层接收 edge_index 而不是图对象
            h = layer(h, edge_index)
        
        # PyG 的全局池化
        if self.pooling == 'avg':
            y = global_mean_pool(h, batch)
        elif self.pooling == 'max':
            y = global_max_pool(h, batch)
        elif self.pooling == 'sum':
            y = global_add_pool(h, batch)
        else:
            raise ValueError(f'Unknown pooling: {self.pooling}')

        out = self.Readout(y)
        return out
    
    def get_grad_norm_weights(self):
        return self.parameters()



'''
class KA_GNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
       
        self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        self.linear = KAN_linear(hidden_feat, out, grid_feat, addbias=use_bias)

        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        #nn.Sigmoid(),
                        self.leaky_relu,
                        self.linear_2,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

    def forward(self, g, features):
        h = self.kan_line(features)
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                h = layer(g, h)  
                
            else:
                h = layer(h) 
        if self.pooling == 'avg':
            y = self.avgpool(g, h)
            #y1 = pool_subgraphs_node(out_1, g_graph)
            #y2 = pool_subgraphs_node(out_2, lg_graph)
            #y3 = pool_subgraphs_node(out_3, fg_graph)


        elif self.pooling == 'max':
            y = self.maxpool(g, h)
            
        elif self.pooling == 'sum':
            y = self.sumpool(g, h)

        else:
            print('No pooling found!!!!')

        out = self.Readout(y)     
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        return self.parameters()
'''
    
