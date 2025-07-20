import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2TrajectoryPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, seq_len=5):
        super(GATv2TrajectoryPredictor, self).__init__()
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels # Store hidden_channels for use in forward
        self.agent_embedding = nn.Linear(in_channels * seq_len, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        
        # New: Centerline processor
        self.centerline_processor = nn.Linear(2, hidden_channels)

        # Adjust fc input size to include processed centerline features
        self.fc = nn.Linear(hidden_channels + hidden_channels, out_channels)

    def forward(self, data):
        print(f"[DEBUG GNN] Input data.x shape: {data.x.shape}")
        agent_x = data.x.view(data.x.size(0), -1)
        agent_x = torch.relu(self.agent_embedding(agent_x))

        x = F.dropout(agent_x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, data.edge_index)
        
        focal_features = x[data.focal_idx]

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

        # Concatenate with focal_features
        combined_features = torch.cat([focal_features, processed_centerlines_tensor], dim=1)

        x = self.fc(combined_features)
        return x