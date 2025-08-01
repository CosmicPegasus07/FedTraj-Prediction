import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2TrajectoryPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, seq_len=30):
        super(GATv2TrajectoryPredictor, self).__init__()
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels # Store hidden_channels for use in forward
        self.agent_embedding = nn.Linear(in_channels * seq_len, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        
        # Map processing components
        self.map_embedding = nn.Linear(2, hidden_channels)
        self.map_conv = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=0.3)

        # Centerline processor
        self.centerline_processor = nn.Linear(2, hidden_channels)

        # Adjust fc input size to include map and centerline features
        self.fc1 = nn.Linear(hidden_channels + hidden_channels + hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc_out = nn.Linear(hidden_channels // 2, out_channels)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        agent_x = data.x.view(data.x.size(0), -1)
        agent_x = torch.relu(self.agent_embedding(agent_x))

        x = F.dropout(agent_x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, data.edge_index)
        
        focal_features = x[data.focal_idx]

        # Process map features
        if hasattr(data, 'lane_x') and data.lane_x.numel() > 0:
            map_x = torch.relu(self.map_embedding(data.lane_x))
            if hasattr(data, 'lane_edge_index') and data.lane_edge_index.numel() > 0:
                map_features = self.map_conv(map_x, data.lane_edge_index)
            else:
                map_features = map_x
            # Global pooling to get single map representation
            map_global = global_mean_pool(map_features, torch.zeros(map_features.size(0), dtype=torch.long, device=map_features.device))
            # Expand to match focal features batch size
            map_global = map_global.expand(focal_features.size(0), -1)
        else:
            map_global = torch.zeros(focal_features.size(0), self.hidden_channels, device=focal_features.device)

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
            num_focal_agents = focal_features.size(0)
            processed_centerlines_tensor = torch.zeros(num_focal_agents, self.hidden_channels, device=focal_features.device)
        else:
            # Stack the processed centerlines to get a tensor of shape (num_valid_centerlines, hidden_channels)
            processed_centerlines_tensor = torch.stack(processed_centerlines, dim=0)
            # This assumes one centerline per focal agent, which might need adjustment
            # If the number of centerlines doesn't match the number of focal agents, this will fail.
            # A more robust solution might involve padding or a different aggregation strategy.
            if processed_centerlines_tensor.size(0) != focal_features.size(0):
                # This is a temporary fix. A better solution is needed.
                # For now, we'll just use the first processed centerline for all focal agents.
                processed_centerlines_tensor = processed_centerlines_tensor[0].unsqueeze(0).expand(focal_features.size(0), -1)

        # Concatenate agent, map, and centerline features
        combined_features = torch.cat([focal_features, map_global, processed_centerlines_tensor], dim=1)

        # Multi-layer prediction with regularization
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)

        return x