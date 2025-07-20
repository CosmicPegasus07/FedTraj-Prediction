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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os
from tqdm import tqdm
import gc
import time

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

def get_cache_path(scenario_path, seq_len, is_test=False):
    """Generate cache file path for processed PyG data"""
    cache_dir = Path("cache") / "pyg_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenario_name = Path(scenario_path).stem
    test_suffix = "_test" if is_test else ""
    cache_file = cache_dir / f"{scenario_name}_seq{seq_len}{test_suffix}.pkl"
    return cache_file

def process_scenario_worker(args):
    """Worker function for multiprocessing scenario processing with caching"""
    scenario_path, is_test, seq_len, global_stats = args

    try:
        # Check cache first for processed PyG data
        cache_path = get_cache_path(scenario_path, seq_len, is_test)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify cache is compatible with current approach
                    if cached_data is not None and hasattr(cached_data, 'x') and cached_data.x.shape[-1] == 5:
                        return cached_data
            except Exception as e:
                # If cache is corrupted or incompatible, continue with processing
                print(f"[WARNING] Cache corrupted for {scenario_path}: {e}")

        # Process scenario
        data = process_single_scenario(scenario_path, is_test, seq_len, global_stats)

        # Cache the result if processing was successful
        if data is not None:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                # If caching fails, continue without caching
                print(f"[WARNING] Failed to cache {scenario_path}: {e}")

        return data
    except Exception as e:
        warnings.warn(f"Error processing {scenario_path}: {e}")
        return None

def get_relative_normalization_stats():
    """Get fixed normalization stats for relative coordinate approach"""
    # Fixed normalization for relative coordinates
    # No need for complex statistics computation
    return {
        'position_scale': 10.0,  # 10 meters for relative positions
        'velocity_scale': 5.0,   # 5 m/s for velocities
        'heading_scale': np.pi,  # pi radians for heading
        'method': 'fixed_relative',
        'description': 'Fixed normalization for relative coordinate approach'
    }

def process_single_scenario(scenario_path, is_test, seq_len, global_stats):
    """Process a single scenario file into PyG Data object"""
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

    if not is_test:
        if focal_future_df.empty:
            return None
        # Pad focal_future_df if it's shorter than seq_len
        if len(focal_future_df) < seq_len:
            last_row = focal_future_df.iloc[-1]
            padding_rows = pd.DataFrame([last_row] * (seq_len - len(focal_future_df)))
            focal_future_df = pd.concat([focal_future_df, padding_rows], ignore_index=True)

    agent_features = []
    agent_ids = past_df['track_id'].unique()
    features_to_normalize = ['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading']

    for agent_id in agent_ids:
        agent_df = past_df[past_df['track_id'] == agent_id].copy()
        agent_df = agent_df.sort_values(by='observed')

        if len(agent_df) < seq_len:
            agent_df = pd.concat([pd.DataFrame([agent_df.iloc[0]] * (seq_len - len(agent_df))), agent_df], ignore_index=True)

        # Convert absolute positions to relative positions (displacements from first position)
        first_pos_x = agent_df['position_x'].iloc[0]
        first_pos_y = agent_df['position_y'].iloc[0]
        
        # Create relative positions (displacements from the first observed position)
        agent_df['relative_x'] = agent_df['position_x'] - first_pos_x
        agent_df['relative_y'] = agent_df['position_y'] - first_pos_y
        
        # Use relative positions and velocities for features
        features = ['relative_x', 'relative_y', 'velocity_x', 'velocity_y', 'heading']
        
        # Simple normalization for relative coordinates using fixed scales
        agent_df['relative_x'] = agent_df['relative_x'] / global_stats['position_scale']
        agent_df['relative_y'] = agent_df['relative_y'] / global_stats['position_scale']
        agent_df['velocity_x'] = agent_df['velocity_x'] / global_stats['velocity_scale']
        agent_df['velocity_y'] = agent_df['velocity_y'] / global_stats['velocity_scale']
        agent_df['heading'] = agent_df['heading'] / global_stats['heading_scale']

        agent_features.append(torch.tensor(
            agent_df[features].values[-seq_len:],
            dtype=torch.float32
        ))

    x = torch.stack(agent_features)

    # Optimized edge creation - only create edges for nearby agents
    num_agents = len(agent_ids)
    if num_agents > 1:
        # For large numbers of agents, create a more sparse connectivity
        if num_agents <= 10:
            # Full connectivity for small graphs
            senders = torch.arange(num_agents).repeat_interleave(num_agents - 1)
            receivers = torch.cat([torch.cat([torch.arange(i), torch.arange(i + 1, num_agents)]) for i in range(num_agents)])
        else:
            # Sparse connectivity for large graphs - connect each agent to k nearest neighbors
            k = min(5, num_agents - 1)  # Connect to at most 5 neighbors
            positions = x[:, -1, :2]  # Use last timestep positions
            distances = torch.cdist(positions, positions)
            _, nearest_indices = torch.topk(distances, k + 1, largest=False, dim=1)  # +1 to exclude self
            nearest_indices = nearest_indices[:, 1:]  # Remove self

            senders = torch.arange(num_agents).repeat_interleave(k)
            receivers = nearest_indices.flatten()

        edge_index = torch.stack([senders, receivers], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    if not is_test:
        # CRITICAL FIX: Use relative targets (displacements from last observed position)
        last_observed_pos_x = focal_future_df['position_x'].iloc[0] if len(focal_future_df) > 0 else 0
        last_observed_pos_y = focal_future_df['position_y'].iloc[0] if len(focal_future_df) > 0 else 0
        
        # Get the last observed position of the focal agent
        focal_past_df = past_df[past_df['track_id'] == focal_track_id]
        if not focal_past_df.empty:
            last_observed_pos_x = focal_past_df['position_x'].iloc[-1]
            last_observed_pos_y = focal_past_df['position_y'].iloc[-1]
        
        # Compute relative targets (displacements from last observed position)
        target_positions = focal_future_df[["position_x", "position_y"]].values[:seq_len]
        relative_targets = target_positions - np.array([last_observed_pos_x, last_observed_pos_y])
        
        # Normalize relative targets using the same scale as inputs
        relative_targets = relative_targets / global_stats['position_scale']

        y = torch.tensor(relative_targets, dtype=torch.float32)
    else:
        y = torch.zeros(seq_len, 2, dtype=torch.float32)

    focal_idx = torch.tensor(np.where(agent_ids == focal_track_id)[0][0], dtype=torch.long)

    # Simplified map processing - cache map data
    # The map file is in the same directory as the parquet file
    scenario_dir = scenario_path.parent
    static_map_path = next(scenario_dir.glob("log_map_archive_*.json"), None)

    if static_map_path is None:
        lane_x = torch.empty(0, 2, dtype=torch.float32)
        lane_edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        lane_x, lane_edge_index = process_map_data(static_map_path)

    centerline = focal_future_df['centerline'].values[0] if not is_test and 'centerline' in focal_future_df else None

    data = Data(x=x, edge_index=edge_index, y=y, focal_idx=focal_idx)
    data.lane_x = lane_x
    data.lane_edge_index = lane_edge_index
    data.centerline = torch.tensor(centerline, dtype=torch.float32) if centerline is not None and centerline.size > 0 else torch.empty(0, 2, dtype=torch.float32)
    data.scenario_path = str(scenario_path)
    data.static_map_path = str(static_map_path) if static_map_path else ""

    return data

def process_map_data(static_map_path):
    """Process map data with proper normalization to match trajectory data"""
    try:
        # Load normalization statistics for relative coordinates
        global_stats = get_relative_normalization_stats()

        with open(static_map_path, 'r') as f:
            map_data = json.load(f)

        lane_segments = map_data.get('lane_segments', {})
        lane_edge_index = []
        lane_features = []
        lane_id_map = {lane_id: i for i, lane_id in enumerate(lane_segments.keys())}



        for lane_id, segment in lane_segments.items():
            # Try to get centerline first, then fall back to polygon boundary
            lane_points = None

            if 'centerline' in segment and segment['centerline']:
                # Extract x, y coordinates from centerline objects
                centerline_coords = []
                for point in segment['centerline']:
                    if isinstance(point, dict) and 'x' in point and 'y' in point:
                        centerline_coords.append([point['x'], point['y']])
                if centerline_coords:
                    lane_points = np.array(centerline_coords)
            elif 'polygon_boundary' in segment and segment['polygon_boundary']:
                # Extract x, y coordinates from polygon boundary objects
                boundary_coords = []
                for point in segment['polygon_boundary']:
                    if isinstance(point, dict) and 'x' in point and 'y' in point:
                        boundary_coords.append([point['x'], point['y']])
                if boundary_coords:
                    lane_points = np.array(boundary_coords)

            if lane_points is not None:
                # Normalize map coordinates using the same scales as trajectory data
                if global_stats is not None:
                    # Apply same normalization as trajectory positions (relative coordinates)
                    normalized_x = lane_points[:, 0] / global_stats['position_scale']
                    normalized_y = lane_points[:, 1] / global_stats['position_scale']
                    normalized_points = np.column_stack([normalized_x, normalized_y])
                else:
                    normalized_points = lane_points

                # Use mean of lane points as lane feature (could also use start/end points)
                lane_features.append(torch.tensor(normalized_points.mean(axis=0), dtype=torch.float32))

            for successor in segment.get('successors', []):
                if successor in lane_id_map:
                    lane_edge_index.append([lane_id_map[lane_id], lane_id_map[successor]])
            for predecessor in segment.get('predecessors', []):
                if predecessor in lane_id_map:
                    lane_edge_index.append([lane_id_map[predecessor], lane_id_map[lane_id]])

        lane_x = torch.stack(lane_features) if lane_features else torch.empty(0, 2, dtype=torch.float32)
        lane_edge_index = torch.tensor(lane_edge_index, dtype=torch.long).t().contiguous() if lane_edge_index else torch.empty(2, 0, dtype=torch.long)

        return lane_x, lane_edge_index
    except Exception as e:
        print(f"Warning: Map processing failed for {static_map_path}: {e}")
        # Return empty tensors if map processing fails
        lane_x = torch.empty(0, 2, dtype=torch.float32)
        lane_edge_index = torch.empty(2, 0, dtype=torch.long)
        return lane_x, lane_edge_index

class ArgoversePyGDataset(PyGDataset):
    def __init__(self, scenario_paths, transform=None, pre_transform=None, is_test=False, seq_len=30, num_workers=None, use_multiprocessing=True):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.is_test = is_test
        self.seq_len = seq_len

        # CRITICAL FIX: Use fixed normalization for relative coordinates
        global_stats = get_relative_normalization_stats()
        print(f"[INFO] Using fixed normalization for relative coordinates: position_scale={global_stats['position_scale']}, velocity_scale={global_stats['velocity_scale']}")

        self.processed_data = []
        self.successful_paths = []

        # Try multiprocessing first, fall back to sequential if it fails
        if use_multiprocessing and len(scenario_paths) > 50:
            success = self._process_with_multiprocessing(scenario_paths, global_stats, num_workers)
            if not success:
                print("[WARNING] Multiprocessing failed, falling back to sequential processing...")
                self._process_sequentially(scenario_paths, global_stats)
        else:
            print(f"[INFO] Processing {len(scenario_paths)} scenarios sequentially...")
            self._process_sequentially(scenario_paths, global_stats)

        print(f"[INFO] Successfully processed {len(self.processed_data)} out of {len(scenario_paths)} scenarios")

        # Force garbage collection
        gc.collect()

    def _process_with_multiprocessing(self, scenario_paths, global_stats, num_workers):
        """Try to process scenarios with multiprocessing"""
        try:
            if num_workers is None:
                num_workers = min(mp.cpu_count() // 2, 4)  # More conservative worker count

            print(f"[INFO] Processing {len(scenario_paths)} scenarios with {num_workers} workers...")

            # Prepare arguments for multiprocessing
            args_list = [(path, self.is_test, self.seq_len, global_stats) for path in scenario_paths]

            # Use a more robust approach with timeout
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_path = {executor.submit(process_scenario_worker, args): args[0] for args in args_list}

                # Collect results with progress bar and timeout
                completed_count = 0
                for future in tqdm(as_completed(future_to_path, timeout=300), total=len(scenario_paths), desc="Loading PyG data"):
                    path = future_to_path[future]
                    try:
                        data = future.result(timeout=30)  # 30 second timeout per scenario
                        if data is not None:
                            self.processed_data.append(data)
                            self.successful_paths.append(path)
                        completed_count += 1
                    except Exception as e:
                        warnings.warn(f"Error processing {path}: {e}")
                        completed_count += 1
                        continue

                return True

        except Exception as e:
            print(f"[WARNING] Multiprocessing failed with error: {e}")
            return False

    def _process_sequentially(self, scenario_paths, global_stats):
        """Process scenarios sequentially as fallback"""
        for path in tqdm(scenario_paths, desc="Loading PyG data (sequential)"):
            try:
                # Check cache first
                cache_path = get_cache_path(path, self.seq_len, self.is_test)
                if cache_path.exists():
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                        # Verify cache is compatible with current approach
                        if cached_data is not None and hasattr(cached_data, 'x') and cached_data.x.shape[-1] == 5:
                            self.processed_data.append(cached_data)
                            self.successful_paths.append(path)
                            continue
                    except Exception as e:
                        # If cache is corrupted or incompatible, continue with processing
                        print(f"[WARNING] Cache corrupted for {path}: {e}")

                # Process scenario
                data = process_single_scenario(path, self.is_test, self.seq_len, global_stats)

                if data is not None:
                    # Cache the result
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        # If caching fails, continue without caching
                        print(f"[WARNING] Failed to cache {path}: {e}")

                    self.processed_data.append(data)
                    self.successful_paths.append(path)

            except Exception as e:
                warnings.warn(f"Error processing {path}: {e}")
                continue

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]



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

def get_pyg_data_loader(scenario_dir, batch_size=32, num_scenarios=-1, selection_criteria="first", shuffle=True, mode='train', client_id=None, num_clients=None, seq_len=30, num_workers=None, use_multiprocessing=None):
    """
    Create optimized PyG data loader with multiprocessing and caching

    Args:
        scenario_dir: Directory containing scenario files
        batch_size: Batch size for data loader
        num_scenarios: Number of scenarios to load (-1 for all)
        selection_criteria: How to select scenarios ("first" or "random")
        shuffle: Whether to shuffle the data
        mode: Mode of operation ('train', 'val', 'test')
        client_id: Client ID for federated learning
        num_clients: Total number of clients for federated learning
        seq_len: Sequence length for trajectory prediction
        num_workers: Number of worker processes for data loading
        use_multiprocessing: Whether to use multiprocessing (None for auto-detect)
    """
    start_time = time.time()

    scenario_files = get_scenario_file_list(scenario_dir, num_scenarios, selection_criteria, client_id, num_clients)

    if not scenario_files:
        print("[WARNING] No scenario files found.")
        return None

    is_test = (mode == 'test')

    # Auto-detect multiprocessing usage
    if use_multiprocessing is None:
        # Disable multiprocessing for small datasets or if we've had issues
        use_multiprocessing = len(scenario_files) > 50

    # Determine number of workers (more conservative)
    if num_workers is None:
        if len(scenario_files) < 100:
            num_workers = 1  # Sequential for very small datasets
        elif len(scenario_files) < 500:
            num_workers = 2
        else:
            num_workers = min(4, mp.cpu_count() // 2)  # More conservative

    dataset = ArgoversePyGDataset(
        scenario_files,
        is_test=is_test,
        seq_len=seq_len,
        num_workers=num_workers,
        use_multiprocessing=use_multiprocessing
    )

    if len(dataset) == 0:
        print("[WARNING] Created an empty dataset. No valid scenarios found.")
        return None

    loading_time = time.time() - start_time
    print(f"[INFO] Created PyG dataset with {len(dataset)} valid samples in {loading_time:.2f} seconds.")
    if len(dataset) > 0:
        print(f"[INFO] Average processing time: {loading_time/len(dataset)*1000:.2f} ms per scenario")

    # Use optimized data loader settings
    loader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=0,  # Disable DataLoader multiprocessing since we already processed in parallel
        pin_memory=torch.cuda.is_available(),  # Pin memory if CUDA is available
        persistent_workers=False
    )

    return loader

def clear_cache():
    """Clear the PyG data cache"""
    cache_dir = Path("cache")
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print("[INFO] Cache cleared successfully.")
    else:
        print("[INFO] No cache found to clear.")

def get_cache_info():
    """Get information about the current cache"""
    cache_dir = Path("cache") / "pyg_data"
    if not cache_dir.exists():
        return {"cached_files": 0, "cache_size_mb": 0}

    cached_files = list(cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cached_files)

    return {
        "cached_files": len(cached_files),
        "cache_size_mb": total_size / (1024 * 1024)
    }