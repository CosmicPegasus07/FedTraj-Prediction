import pandas as pd
import json
import torch
from utils.data_utils import get_scenario_file_list

# Get a single scenario file from the small training set
scenario_files = get_scenario_file_list("dataset/train_small", num_scenarios=1)

if scenario_files:
    # --- Inspect Parquet File ---
    scenario_path = scenario_files[0]
    print(f"--- Inspecting Parquet File: {scenario_path} ---")
    df = pd.read_parquet(scenario_path)
    print("\nDataFrame Columns:")
    print(df.columns)
    print("\nDataFrame Head:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()

    # Inspect focal_future_df and y tensor
    focal_track_id = df['focal_track_id'].iloc[0]
    future_df = df[df['observed'] == False].copy()
    focal_future_df = future_df[future_df['track_id'] == focal_track_id]
    print("\nFocal Future DataFrame (Ground Truth):")
    print(focal_future_df)

    if not focal_future_df.empty:
        y_tensor = torch.tensor(
            focal_future_df[["position_x", "position_y"]].values[0], 
            dtype=torch.float32
        )
        print("\nExtracted Y tensor (Ground Truth):")
        print(y_tensor)
    else:
        print("\nNo focal future data found for ground truth.")

    # --- Inspect JSON Map File ---
    static_map_path = next(scenario_path.parent.glob("log_map_archive_*.json"), None)
    if static_map_path:
        print(f"\n--- Inspecting JSON Map File: {static_map_path} ---")
        with open(static_map_path, 'r') as f:
            map_data = json.load(f)
        print("Map Data Keys:")
        print(map_data.keys())
        # Print some sample data from the map
        if 'lane_segments' in map_data:
            print("\nSample Lane Segment:")
            # Get the first lane segment that has a polygon_boundary
            sample_segment = None
            for segment_id, segment_data in map_data['lane_segments'].items():
                if 'polygon_boundary' in segment_data and segment_data['polygon_boundary']:
                    sample_segment = segment_data
                    break
            if sample_segment:
                print(sample_segment)
            else:
                print("No lane segments with polygon_boundary found.")
    else:
        print("\nNo map file found for this scenario.")
else:
    print("No scenario files found.")