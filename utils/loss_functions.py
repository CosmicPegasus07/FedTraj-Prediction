"""
Enhanced loss functions for trajectory prediction with map constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MapConstrainedLoss(nn.Module):
    """
    Loss function that combines trajectory prediction loss with map constraints
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super(MapConstrainedLoss, self).__init__()
        self.alpha = alpha  # Weight for trajectory loss
        self.beta = beta    # Weight for velocity consistency
        self.gamma = gamma  # Weight for smoothness
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets, centerlines=None):
        """
        Args:
            predictions: (batch_size, seq_len, 2) predicted trajectories
            targets: (batch_size, seq_len, 2) ground truth trajectories
            centerlines: Optional centerline information for map constraints
        """
        batch_size, seq_len, _ = predictions.shape
        
        # 1. Basic trajectory loss (MSE)
        traj_loss = self.mse_loss(predictions, targets)
        
        # 2. Velocity consistency loss
        pred_velocities = predictions[:, 1:] - predictions[:, :-1]
        target_velocities = targets[:, 1:] - targets[:, :-1]
        velocity_loss = self.mse_loss(pred_velocities, target_velocities)
        
        # 3. Smoothness loss (penalize sharp turns)
        if seq_len > 2:
            pred_accelerations = pred_velocities[:, 1:] - pred_velocities[:, :-1]
            smoothness_loss = torch.mean(torch.norm(pred_accelerations, dim=-1))
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # 4. Map constraint loss (if centerlines available)
        map_loss = torch.tensor(0.0, device=predictions.device)
        if centerlines is not None:
            # Penalize predictions that are too far from centerlines
            valid_centerlines = 0
            for i, centerline in enumerate(centerlines):
                if centerline is not None and centerline.numel() > 0 and i < predictions.shape[0]:
                    pred_traj = predictions[i]  # (seq_len, 2)

                    # Ensure centerline has the right shape
                    if centerline.dim() == 1:
                        centerline = centerline.view(-1, 2)

                    if centerline.shape[1] == 2 and centerline.shape[0] > 0:
                        # Find closest centerline points for each prediction point
                        distances = torch.cdist(pred_traj.unsqueeze(0), centerline.unsqueeze(0))  # (1, seq_len, centerline_len)
                        min_distances, _ = torch.min(distances.squeeze(0), dim=1)  # (seq_len,)

                        # Penalize if too far from centerline (threshold adjusted for normalized space)
                        threshold = 0.1  # Smaller threshold for normalized coordinates
                        map_penalty = torch.clamp(min_distances - threshold, min=0.0)
                        map_loss += torch.mean(map_penalty)
                        valid_centerlines += 1

            # Average over valid centerlines
            if valid_centerlines > 0:
                map_loss = map_loss / valid_centerlines
        
        # Combine all losses
        total_loss = (self.alpha * traj_loss + 
                     self.beta * velocity_loss + 
                     self.gamma * smoothness_loss + 
                     0.1 * map_loss)
        
        return total_loss, {
            'trajectory_loss': traj_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'map_loss': map_loss.item(),
            'total_loss': total_loss.item()
        }

class ScaledTrajectoryLoss(nn.Module):
    """
    Scaled trajectory loss function that produces reasonable loss values (around 0.2)
    """
    def __init__(self, scale_factor=100.0):
        super(ScaledTrajectoryLoss, self).__init__()
        self.scale_factor = scale_factor
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, seq_len, 2) or (batch_size, output_size)
            targets: (batch_size, seq_len, 2) or (batch_size, output_size)
        """
        # Calculate base MSE loss
        base_loss = self.mse_loss(predictions, targets)
        
        # Scale the loss to get reasonable values
        scaled_loss = base_loss * self.scale_factor
        
        return scaled_loss

class ImprovedTrajectoryLoss(nn.Module):
    """
    Improved trajectory loss with better normalization and stability
    """
    def __init__(self, reduction='mean'):
        super(ImprovedTrajectoryLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, seq_len * 2) or (batch_size, seq_len, 2)
            targets: (batch_size, seq_len * 2) or (batch_size, seq_len, 2)
        """
        # Ensure consistent shape
        if predictions.dim() == 2:
            batch_size, features = predictions.shape
            seq_len = features // 2
            predictions = predictions.view(batch_size, seq_len, 2)
            targets = targets.view(batch_size, seq_len, 2)
        
        # Calculate L2 distance for each time step
        distances = torch.norm(predictions - targets, dim=-1)  # (batch_size, seq_len)
        
        # Apply time-weighted loss (later predictions are more important)
        time_weights = torch.linspace(1.0, 2.0, distances.shape[1], device=distances.device)
        weighted_distances = distances * time_weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return torch.mean(weighted_distances)
        elif self.reduction == 'sum':
            return torch.sum(weighted_distances)
        else:
            return weighted_distances

def compute_trajectory_metrics(predictions, targets):
    """
    Compute standard trajectory prediction metrics
    
    Args:
        predictions: (batch_size, seq_len, 2) predicted trajectories
        targets: (batch_size, seq_len, 2) ground truth trajectories
    
    Returns:
        dict: Dictionary containing ADE, FDE, and other metrics
    """
    # Ensure consistent shape
    if predictions.dim() == 2:
        batch_size, features = predictions.shape
        seq_len = features // 2
        predictions = predictions.view(batch_size, seq_len, 2)
        targets = targets.view(batch_size, seq_len, 2)
    
    # Calculate distances at each time step
    distances = torch.norm(predictions - targets, dim=-1)  # (batch_size, seq_len)
    
    # Average Displacement Error (ADE)
    ade = torch.mean(distances)
    
    # Final Displacement Error (FDE)
    fde = torch.mean(distances[:, -1])
    
    # Miss Rate (percentage of predictions with FDE > 2 meters)
    # Note: Adjust threshold based on your coordinate system
    miss_threshold = 0.2  # Adjust based on normalization
    miss_rate = torch.mean((distances[:, -1] > miss_threshold).float())
    
    return {
        'ade': ade.item(),
        'fde': fde.item(),
        'miss_rate': miss_rate.item(),
        'mean_distance': torch.mean(distances).item(),
        'max_distance': torch.max(distances).item()
    }


