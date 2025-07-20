# data_utils.py

import torch
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
from pathlib import Path
import pandas as pd
import numpy as np
import random
import warnings
import json

def get_scenario_file_list(scenario_dir, num_scenarios=-1, selection_criteria="first", client_id=None, num_clients=None):
    scenario_dir = Path(scenario_dir)
    scenario_folders = [f for f in scenario_dir.iterdir() if f.is_dir()]
    print(f"[INFO] Found {len(scenario_folders)} scenario folders in {scenario_dir}.")

    if num_scenarios != -1:
        if selection_criteria == "first":
            selected_folders = scenario_folders[:num_scenarios]
        else:
            selected_folders = random.sample(scenario_folders, min(num_scenarios, len(scenario_folders)))
    else:
        selected_folders = scenario_folders

    if client_id is not None and num_clients is not None:
        files_per_client = len(selected_folders) // num_clients
        start_index = client_id * files_per_client
        end_index = start_index + files_per_client if client_id < num_clients - 1 else len(selected_folders)
        selected_folders = selected_folders[start_index:end_index]

    scenario_files = []
    for folder in selected_folders:
        parquet_files = list(folder.glob("scenario_*.parquet"))
        if parquet_files:
            scenario_files.append(parquet_files[0])
    
    print(f"[INFO] Using {len(scenario_files)} scenario parquet files for loading.")
    return scenario_files

class ArgoversePyGDataset(PyGDataset):
    def __init__(self, scenario_paths, transform=None, pre_transform=None, is_test=False, seq_len=5):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.is_test = is_test
        self.seq_len = seq_len
        
        self.processed_data = []
        self.successful_paths = []

        for path in scenario_paths:
            data = self.process_scenario(path)
            if data is not None:
                self.processed_data.append(data)
                self.successful_paths.append(path)

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

    def process_scenario(self, scenario_path):
        try:
            df = pd.read_parquet(scenario_path)
        except Exception as e:
            warnings.warn(f"Could not read {scenario_path}: {e}")
            return None

        past_df = df[df['observed'] == True].copy()
        future_df = df[df['observed'] == False].copy()

        focal_track_id = df['focal_track_id'].iloc[0]
        focal_future_df = future_df[future_df['track_id'] == focal_track_id]

        if past_df.empty:
            return None
        
        if not self.is_test:
            if focal_future_df.empty:
                return None
            # Pad focal_future_df if it's shorter than seq_len
            if len(focal_future_df) < self.seq_len:
                last_row = focal_future_df.iloc[-1]
                padding_rows = pd.DataFrame([last_row] * (self.seq_len - len(focal_future_df)))
                focal_future_df = pd.concat([focal_future_df, padding_rows], ignore_index=True)

        agent_features = []
        agent_ids = past_df['track_id'].unique()
        for agent_id in agent_ids:
            agent_df = past_df[past_df['track_id'] == agent_id].copy()
            agent_df = agent_df.sort_values(by='observed')
            
            if len(agent_df) < self.seq_len:
                agent_df = pd.concat([pd.DataFrame([agent_df.iloc[0]] * (self.seq_len - len(agent_df))), agent_df], ignore_index=True)

            # Normalize features
            features_to_normalize = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading']
            # Placeholder for mean and std. In a real scenario, these should be pre-calculated from the entire training dataset.
            # For demonstration, using arbitrary values or values derived from a small sample.
            # Example: mean_vals = {'position_x': 0.0, 'position_y': 0.0, 'velocity_x': 0.0, 'velocity_y': 0.0, 'heading': 0.0}
            # Example: std_vals = {'position_x': 100.0, 'position_y': 100.0, 'velocity_x': 10.0, 'velocity_y': 10.0, 'heading': 1.0}
            # For now, a simple standardization assuming roughly centered data.
            for col in features_to_normalize:
                mean_val = agent_df[col].mean() # This is not ideal, should be global mean
                std_val = agent_df[col].std() # This is not ideal, should be global std
                if std_val == 0: # Avoid division by zero
                    agent_df[col] = 0.0
                else:
                    agent_df[col] = (agent_df[col] - mean_val) / std_val

            agent_features.append(torch.tensor(
                agent_df[features_to_normalize].values[-self.seq_len:],
                dtype=torch.float32
            ))
        
        x = torch.stack(agent_features)

        num_agents = len(agent_ids)
        if num_agents > 1:
            senders = torch.arange(num_agents).repeat_interleave(num_agents - 1)
            receivers = torch.cat([torch.cat([torch.arange(i), torch.arange(i + 1, num_agents)]) for i in range(num_agents)])
            edge_index = torch.stack([senders, receivers], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        if not self.is_test:
            y = torch.tensor(
                focal_future_df[["position_x", "position_y"]].values[:self.seq_len], 
                dtype=torch.float32
            )
        else:
            y = torch.zeros(self.seq_len, 2, dtype=torch.float32)

        focal_idx = torch.tensor(np.where(agent_ids == focal_track_id)[0][0], dtype=torch.long)
        
        static_map_path = next(scenario_path.parent.glob("log_map_archive_*.json"), None)
        if static_map_path is None:
            warnings.warn(f"Could not find map for {scenario_path}. Skipping.")
            return None
        
        with open(static_map_path, 'r') as f:
            map_data = json.load(f)
        
        lane_segments = map_data.get('lane_segments', {})
        lane_edge_index = []
        lane_features = []
        lane_id_map = {lane_id: i for i, lane_id in enumerate(lane_segments.keys())}
        
        for lane_id, segment in lane_segments.items():
            if 'polygon_boundary' in segment and segment['polygon_boundary']:
                lane_features.append(torch.tensor(np.array(segment['polygon_boundary']), dtype=torch.float32).mean(axis=0))
            for successor in segment.get('successors', []):
                if successor in lane_id_map:
                    lane_edge_index.append([lane_id_map[lane_id], lane_id_map[successor]])
            for predecessor in segment.get('predecessors', []):
                if predecessor in lane_id_map:
                    lane_edge_index.append([lane_id_map[predecessor], lane_id_map[lane_id]])

        lane_x = torch.stack(lane_features) if lane_features else torch.empty(0, 2, dtype=torch.float32)
        lane_edge_index = torch.tensor(lane_edge_index, dtype=torch.long).t().contiguous() if lane_edge_index else torch.empty(2, 0, dtype=torch.long)

        centerline = focal_future_df['centerline'].values[0] if not self.is_test and 'centerline' in focal_future_df else None

        data = Data(x=x, edge_index=edge_index, y=y, focal_idx=focal_idx)
        data.lane_x = lane_x
        data.lane_edge_index = lane_edge_index
        data.centerline = torch.tensor(centerline, dtype=torch.float32) if centerline is not None and centerline.size > 0 else torch.empty(0, 2, dtype=torch.float32)
        data.scenario_path = str(scenario_path)
        data.static_map_path = str(static_map_path)
        
        return data

def custom_collate(batch):
    scenario_paths = [item.scenario_path for item in batch]
    static_map_paths = [item.static_map_path for item in batch]
    centerlines = [item.centerline for item in batch] # Collect centerlines
    
    # Temporarily remove custom attributes for default collation
    for item in batch:
        delattr(item, 'scenario_path')
        delattr(item, 'static_map_path')
        # Only delete centerline if it exists to avoid AttributeError
        if hasattr(item, 'centerline'):
            delattr(item, 'centerline')
    
    collated_batch = Batch.from_data_list(batch)
    
    # Add the lists of paths and centerlines back to the collated batch
    collated_batch.scenario_path = scenario_paths
    collated_batch.static_map_path = static_map_paths
    collated_batch.centerline = centerlines # Assign directly as an attribute
    
    return collated_batch

def get_pyg_data_loader(scenario_dir, batch_size=32, num_scenarios=-1, selection_criteria="first", shuffle=True, mode='train', client_id=None, num_clients=None, seq_len=5):
    scenario_files = get_scenario_file_list(scenario_dir, num_scenarios, selection_criteria, client_id, num_clients)
    
    is_test = (mode == 'test')
    dataset = ArgoversePyGDataset(scenario_files, is_test=is_test, seq_len=seq_len)
    
    if len(dataset) == 0:
        print("[WARNING] Created an empty dataset. No valid scenarios found.")
        return None

    print(f"[INFO] Created PyG dataset with {len(dataset)} valid samples.")
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)    
    return loader