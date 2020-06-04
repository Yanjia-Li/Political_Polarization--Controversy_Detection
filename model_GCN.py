
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops



class GCN_Network(nn.Module):

    def __init__(self, input_dim=512):
        super(GCN_Network, self).__init__()

        self.gcn1 = GraphConvolutionLayer(input_dim, 64)
        self.gcn2 = GraphConvolutionLayer(64, 16)
        #self.gcn3 = GraphConvolutionLayer(16, 1)
        self.lin = torch.nn.Linear(16, 1)
        

    def forward(self, adjacency, feature):

        h = F.relu(self.gcn1(adjacency, feature)) #input_size = 512 = num_of_features # output_size = 64 = h_size
        logits = self.gcn2(adjacency, h) #input_size = 64 = h_size # output_size = 16 = size_of_logits
        #logits = self.gcn3(adjacency, logits) # input_size = 16 = size_of_logits # output_size = 1
        logits = self.lin(logits)
        
        logits = torch.sigmoid(logits)
        return logits

class GraphConvolutionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolutionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):

        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        #print('hh, this is the shape ',self.weight.shape)

        if self.use_bias:
            output += self.bias
        return output









class SAGEConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
    
        super(SAGEConv, self).__init__(aggr='max') #  "Max" aggregation.
        
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]


        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding


