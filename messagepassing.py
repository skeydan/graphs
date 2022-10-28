import numpy as np

import torch
from torch.nn import Linear, Parameter, Sequential as Seq, Linear, ReLU
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GCNConv, global_add_pool, GraphConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree



#################################################################################################################
#################################################################################################################

x = torch.tensor([[1, 1], [2, 2], [3, 3], [11, 11], [12, 12]], dtype=torch.float)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [0, 2],
                           [2, 0],
                           [1, 2],
                           [2, 1],
                           [2, 3],
                           [3, 2],
                           [2, 4],
                           [4, 2],
                           [3, 4],
                           [4, 3]
                           ], dtype=torch.long)
y = torch.tensor([[0], [0], [0], [1], [1]], dtype=torch.float)
data = Data(x = x, edge_index = edge_index.t().contiguous(), y = y)
data.edge_index, data.edge_index.shape
data.x, data.x.shape
data.y


# incoming connections
data.edge_index[0]
# tensor([0, 1, 0, 2, 1, 2, 2, 3, 2, 4, 3, 4])
# i.e. coming from 0, coming from 1, coming from 0, ...
# this yields incoming nodes' (= neighbors') features 
x[data.edge_index[0]]
# tensor([[ 1.,  1.],
#         [ 2.,  2.],
#         [ 1.,  1.],
#         [ 3.,  3.],
#         [ 2.,  2.],
#         [ 3.,  3.],
#         [ 3.,  3.],
#         [11., 11.],
#         [ 3.,  3.],
#         [12., 12.],
#         [11., 11.],
#         [12., 12.]])
# i.e. from 0 get (1,1), from 1 get (2,2), ...
# this is available as x_j

# outgoing connections
data.edge_index[1]
# i.e., going to node 0, ...
# the features sent over those connections
x[data.edge_index[1]]  


#################################################################################################################
#################################################################################################################


class IAmMyOthers(MessagePassing):
    def __init__(self):
        super().__init__(aggr = "sum") # or use mean, max, min
    def forward(self, x, edge_index):
        print("in forward")
        out = self.propagate(edge_index, x = x)
        return out
    def message(self, x_j):
        print("in message, x_j is")
        # node features of node j for all edges coming into node i 
        # same as x[data.edge_index[0]]
        # __lift__ uses torch.index_select
        print(x_j)
        return x_j

module = IAmMyOthers()
out = module(data.x, data.edge_index)
print("result is:")
out


######################################

class IAmMyOthersAndMyselfAsWell(MessagePassing):
    def __init__(self):
        super().__init__(aggr = "sum")
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print("in forward, augmented edge index is")
        print(edge_index.shape)
        out = self.propagate(edge_index, x = x)
        return out
    def message(self, x_j):
        return x_j

module = IAmMyOthersAndMyselfAsWell()
out = module(data.x, data.edge_index)
print("result is:")
out


######################################

#https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html
from torch_scatter import scatter

class IAmJustTheOppositeReally(MessagePassing):
    def __init__(self):
        super().__init__() 
    def forward(self, x, edge_index):
        print("in forward")
        out = self.propagate(edge_index, x = x)
        return out
    def message(self, x_j):
        print("in message, x_j is")
        print(x_j)
        return x_j
    def aggregate(self, inputs, index):
        print("in aggregate, inputs is")
        # same as x_j (incoming node features)
        print(inputs)
        print("in aggregate, index is")
        # this is data.edge_index[1]
        print(index)
        return - scatter(inputs, index, dim = 0, reduce = "add") # default is -1
        
# read like so:
# inputs =
# tensor([[ 1.,  1.],
#         [ 2.,  2.],
#         [ 1.,  1.],
#         [ 3.,  3.],
#         [ 2.,  2.],
#         [ 3.,  3.],
#         [ 3.,  3.],
#         [11., 11.],
#         [ 3.,  3.],
#         [12., 12.],
#         [11., 11.],
#         [12., 12.]])
# index = 
# tensor([1, 0, 2, 0, 2, 1, 3, 2, 4, 2, 4, 3])
# (1,1) goes into 1, (2,2) goes into 0, ...

module = IAmJustTheOppositeReally()
out = module(data.x, data.edge_index)
print("result is:")
out


######################################

class IDoEvolveOverTime(MessagePassing):
    def __init__(self):
        super().__init__(aggr = "sum")
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x = x)
        return out
    def message(self, x_j):
        print("in message")
        return x_j
    def aggregate(self, inputs, index):
        print("in aggregate")
        # same as x_j (incoming node features)
        return(scatter(inputs, index, dim = 0, reduce = "add")) # default is -1
    def update(self, inputs, x):
        print("in update, inputs is")
        print(inputs)
        print("in update, x is")
        print(x)
        return (inputs + x)/2

module = IDoEvolveOverTime()
out = module(data.x, data.edge_index)
print("result is:")
out


######################################

class ILearnAndEvolve(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "sum")
        self.mlp = Seq(Linear(in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.mlp(x)
        out = self.propagate(edge_index = edge_index, x = x)
        return out
    def message(self, x_j):
        return x_j
    def update(self, inputs, x):
        return (inputs + x)/2

module = ILearnAndEvolve(2, 2)
out = module(data.x, data.edge_index)
print("result is:")
out

######################################


class ILearnAndEvolveDoubly(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "sum")
        self.mlp1 = Seq(Linear(in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
        self.mlp2 = Seq(Linear(2 * out_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.mlp1(x)
        out = self.propagate(edge_index = edge_index, x = x)
        return out
    def message(self, x_j, x_i):
        new_embeddings = torch.cat([x_i, x_j - x_i], dim = 1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp2(new_embeddings)
    def update(self, inputs, x):
        return (inputs + x)/2

module = ILearnAndEvolveDoubly(2, 2)
out = module(data.x, data.edge_index)
print("result is:")
out




#################################################################################################################
#################################################################################################################

class Network(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.conv = ILearnAndEvolveDoubly(in_channels, out_channels)
        self.classifier = Linear(out_channels, num_classes)

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        return self.classifier(x)

model = Network(2, 2, 1)
model(data.x, data.edge_index)  
      

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.binary_cross_entropy_with_logits(out, data.y)
    print(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    
preds = torch.sigmoid(out)
preds
