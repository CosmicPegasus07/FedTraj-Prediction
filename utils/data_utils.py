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

def get_scenario_file_list(scenario_dir, num_scenarios=-1, selection_criteria="first", client_id=None, num_clients=None):
    """
    Gets a list of scenario parquet files from a directory.
    - num_scenarios: Number of scenarios to load. -1 means all.
    """
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
    """
    PyTorch Geometric dataset for Argoverse scenarios.
    """
    def __init__(self, scenario_paths, transform=None, pre_transform=None, is_test=False):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.is_test = is_test
        
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
        """
        Processes a single parquet file into a PyG Data object.
        """
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
        
        if not self.is_test and focal_future_df.empty:
            return None

        last_step_df = past_df.groupby('track_id').last().reset_index()
        num_agents = len(last_step_df)
        
        x = torch.tensor(
            last_step_df[["position_x", "position_y", "velocity_x", "velocity_y", "heading"]].values,
            dtype=torch.float32
        )

        if num_agents > 1:
            senders = torch.arange(num_agents).repeat_interleave(num_agents - 1)
            receivers = torch.cat([torch.cat([torch.arange(i), torch.arange(i + 1, num_agents)]) for i in range(num_agents)])
            edge_index = torch.stack([senders, receivers], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        if not self.is_test:
            y = torch.tensor(
                focal_future_df[["position_x", "position_y"]].values[0], 
                dtype=torch.float32
            )
        else:
            y = torch.zeros(2, dtype=torch.float32)

        focal_agent_idx_list = last_step_df.index[last_step_df['track_id'] == focal_track_id].tolist()
        if not focal_agent_idx_list:
            return None

        focal_idx = torch.tensor(focal_agent_idx_list[0], dtype=torch.long)
        
        # Find the static map path
        static_map_path = next(scenario_path.parent.glob("log_map_archive_*.json"), None)
        if static_map_path is None:
            warnings.warn(f"Could not find map for {scenario_path}. Skipping.")
            return None

        data = Data(x=x, edge_index=edge_index, y=y, focal_idx=focal_idx)
        data.scenario_path = str(scenario_path)
        data.static_map_path = str(static_map_path)
        
        return data

def custom_collate(batch):
    """
    Custom collate function to handle scenario and map paths.
    """
    scenario_paths = [item.scenario_path for item in batch]
    static_map_paths = [item.static_map_path for item in batch]
    
    # Temporarily remove custom attributes for default collation
    for item in batch:
        delattr(item, 'scenario_path')
        delattr(item, 'static_map_path')
    
    collated_batch = Batch.from_data_list(batch)
    
    # Add the lists of paths back to the collated batch
    collated_batch.scenario_path = scenario_paths
    collated_batch.static_map_path = static_map_paths
    
    return collated_batch

def get_pyg_data_loader(scenario_dir, batch_size=32, num_scenarios=-1, selection_criteria="first", shuffle=True, mode='train', client_id=None, num_clients=None):
    """
    Creates a PyG DataLoader for the Argoverse dataset.
    """
    scenario_files = get_scenario_file_list(scenario_dir, num_scenarios, selection_criteria, client_id, num_clients)
    
    is_test = (mode == 'test')
    dataset = ArgoversePyGDataset(scenario_files, is_test=is_test)
    
    if len(dataset) == 0:
        print("[WARNING] Created an empty dataset. No valid scenarios found.")
        return None

    print(f"[INFO] Created PyG dataset with {len(dataset)} valid samples.")
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)    
    return loader
