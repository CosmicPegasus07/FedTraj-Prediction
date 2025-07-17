import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class SubGraph(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SubGraph, self).__init__()
        self.conv1 = GATv2Conv(in_channels, out_channels, heads=4, concat=True, dropout=0.6) # Output: out_channels * 4
        self.conv2 = GATv2Conv(out_channels * 4, out_channels, heads=4, concat=True, dropout=0.6) # Output: out_channels * 4

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x # Output shape: [num_nodes, out_channels * 4]

class VectorNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32):
        super(VectorNet, self).__init__()
        self.agent_subgraph = SubGraph(in_channels, hidden_dim) # Outputs hidden_dim * 4
        self.global_graph = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=True, dropout=0.6) # Outputs hidden_dim
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index):
        # Process through agent subgraph
        agent_features = self.agent_subgraph(x, edge_index) # Output: [num_nodes, hidden_dim * 4]

        # Process through global graph
        # The global_graph is also a GATv2Conv, so it takes node features and edge_index
        global_features = torch.relu(self.global_graph(agent_features, edge_index)) # Output: [num_nodes, hidden_dim]

        # Final linear layer for prediction
        output = self.fc(global_features) # Output: [num_nodes, out_channels]
        return output