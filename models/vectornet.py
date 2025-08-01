import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class SubGraph(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.6):
        super(SubGraph, self).__init__()
        self.conv1 = GATv2Conv(in_channels, out_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(out_channels * heads, out_channels, heads=heads, concat=True, dropout=dropout)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

class VectorNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=32, seq_len=30):
        super(VectorNet, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim # Store hidden_dim for use in forward
        self.agent_embedding = nn.Linear(in_channels * seq_len, hidden_dim)
        self.agent_subgraph = SubGraph(hidden_dim, hidden_dim)
        
        self.map_embedding = nn.Linear(2, hidden_dim)
        self.map_subgraph = SubGraph(hidden_dim, hidden_dim)

        # New: Centerline processor
        self.centerline_processor = nn.Linear(2, hidden_dim)

        self.global_graph = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=True, dropout=0.6)

        # Multi-layer prediction with map, agent, and centerline integration
        self.fc1 = nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim)  # agent + map + centerline
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, out_channels)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        # Agent processing
        agent_x = data.x.view(data.x.size(0), -1)
        agent_x = torch.relu(self.agent_embedding(agent_x))
        agent_features = self.agent_subgraph(agent_x, data.edge_index)

        # Map processing
        if hasattr(data, 'lane_x') and data.lane_x.numel() > 0:
            map_x = torch.relu(self.map_embedding(data.lane_x))
            if hasattr(data, 'lane_edge_index') and data.lane_edge_index.numel() > 0:
                map_features = self.map_subgraph(map_x, data.lane_edge_index)
            else:
                map_features = map_x
            # Global pooling to get single map representation
            map_global = global_mean_pool(map_features, torch.zeros(map_features.size(0), dtype=torch.long, device=map_features.device))
        else:
            map_global = torch.zeros(1, self.hidden_dim, device=agent_features.device)

        # Agent global processing
        global_features = torch.relu(self.global_graph(agent_features, data.edge_index))
        focal_global_features = global_features[data.focal_idx]

        # Expand map features to match focal features batch size
        map_global = map_global.expand(focal_global_features.size(0), -1)

        # Process centerline features
        processed_centerlines = []
        if hasattr(data, 'centerline'):
            for cl in data.centerline:
                if cl is not None and cl.numel() > 0:
                    mean_cl = torch.mean(cl, dim=0)
                    processed_cl = self.centerline_processor(mean_cl)
                    processed_centerlines.append(processed_cl)
        
        if not processed_centerlines:
            # If there are no valid centerlines, use a zero tensor for all items in the batch
            num_focal_agents = focal_global_features.size(0)
            processed_centerlines_tensor = torch.zeros(num_focal_agents, self.hidden_dim, device=focal_global_features.device)
        else:
            # Stack the processed centerlines to get a tensor of shape (num_valid_centerlines, hidden_dim)
            processed_centerlines_tensor = torch.stack(processed_centerlines, dim=0)
            # This assumes one centerline per focal agent, which might need adjustment
            if processed_centerlines_tensor.size(0) != focal_global_features.size(0):
                # This is a temporary fix. A better solution is needed.
                # For now, we'll just use the first processed centerline for all focal agents.
                processed_centerlines_tensor = processed_centerlines_tensor[0].unsqueeze(0).expand(focal_global_features.size(0), -1)

        # Concatenate agent, map, and centerline features
        combined_features = torch.cat([focal_global_features, map_global, processed_centerlines_tensor], dim=1)

        # Multi-layer prediction with regularization
        x = torch.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc_out(x)

        return output