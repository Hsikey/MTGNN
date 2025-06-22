import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv, GCNConv, GraphMultisetTransformer,GATConv


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim):
        super(GNN, self).__init__()
        # self.conv1 = RGCNConv(g_dim, h1_dim, 3)
        self.conv1 = GCNConv(g_dim, h1_dim)
        # self.conv2 = GATConv(g_dim, h1_dim)
        # self.conv3 = GATConv(h1_dim, h2_dim)
        # self.conv4 = GCNConv(h1_dim, h2_dim)
        self.conv5 = TransformerConv(h1_dim, h2_dim, heads=4, concat=False)
        self.bn = nn.BatchNorm1d(h2_dim * 1)


    # def forward(self, node_features, edge_index, edge_weight):
    #     x = self.conv1(node_features, edge_index, edge_weight)
    #     x = nn.functional.leaky_relu(self.bn(self.conv4(x, edge_index, edge_weight)))

    def forward(self, node_features, edge_index, edge_weight):
        x = self.conv1(node_features, edge_index, edge_weight)
        x = nn.functional.leaky_relu(self.bn(self.conv5(x, edge_index)))


        return x
