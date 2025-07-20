import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

def constrain_predictions_to_map(pred_traj_abs, static_map):
    """
    Constrain predictions to be within the map boundaries
    """
    if pred_traj_abs is None or static_map is None:
        return pred_traj_abs
    
    # Get map boundaries from drivable areas
    map_coords = []
    for drivable_area in static_map.vector_drivable_areas.values():
        area_coords = np.array([[point.x, point.y] for point in drivable_area.area_boundary])
        map_coords.append(area_coords)
    
    if not map_coords:
        return pred_traj_abs
    
    # Calculate map bounds
    all_map_coords = np.vstack(map_coords)
    map_min_x, map_min_y = np.min(all_map_coords, axis=0)
    map_max_x, map_max_y = np.max(all_map_coords, axis=0)
    
    # Add some padding to map bounds
    padding = 50  # meters
    map_min_x -= padding
    map_min_y -= padding
    map_max_x += padding
    map_max_y += padding
    
    # Constrain predictions
    constrained_pred = pred_traj_abs.copy()
    constrained_pred[:, 0] = np.clip(constrained_pred[:, 0], map_min_x, map_max_x)
    constrained_pred[:, 1] = np.clip(constrained_pred[:, 1], map_min_y, map_max_y)
    
    return constrained_pred

def plot_argoverse2_with_prediction(
    scenario_path,
    static_map_path,
    pred_traj_abs,
    gt_traj_abs=None,
    title="Trajectory Prediction on Argoverse2 Map",
    save_path=None
):
    """
    Plot an Argoverse2 scenario with map, agents, and overlay your prediction.
    """
    try:
        scenario_path = Path(scenario_path)
        static_map_path = Path(static_map_path)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
    except Exception as e:
        print(f"Error loading scenario or map: {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    all_coords = []

    for drivable_area in static_map.vector_drivable_areas.values():
        polygon_points = np.array([[point.x, point.y] for point in drivable_area.area_boundary])
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], color='#e0e0e0', alpha=0.5, zorder=0)
        all_coords.append(polygon_points)

    for lane_segment in static_map.vector_lane_segments.values():
        left_boundary_xyz = lane_segment.left_lane_boundary.xyz
        right_boundary_xyz = lane_segment.right_lane_boundary.xyz
        ax.plot(left_boundary_xyz[:, 0], left_boundary_xyz[:, 1], '--', color='gray', alpha=0.7, linewidth=1, zorder=1)
        ax.plot(right_boundary_xyz[:, 0], right_boundary_xyz[:, 1], '--', color='gray', alpha=0.7, linewidth=1, zorder=1)
        all_coords.append(left_boundary_xyz[:, :2])
        all_coords.append(right_boundary_xyz[:, :2])

    focal_track_id = scenario.focal_track_id
    for track in scenario.tracks:
        observed_traj_points = [
            [object_state.position[0], object_state.position[1]]
            for object_state in track.object_states if object_state.observed
        ]
        
        if not observed_traj_points:
            continue
            
        observed_traj = np.array(observed_traj_points)
        all_coords.append(observed_traj)
        
        is_focal = track.track_id == focal_track_id
        color = 'blue' if is_focal else 'purple'
        lw = 2 if is_focal else 1
        alpha = 1.0 if is_focal else 0.7

        ax.plot(observed_traj[:, 0], observed_traj[:, 1], color=color, linewidth=lw, alpha=alpha, marker='o', markersize=2)
        if is_focal and len(observed_traj) > 0:
            ax.plot(observed_traj[-1, 0], observed_traj[-1, 1], 'o', color='darkblue', markersize=8, zorder=6)

    if pred_traj_abs is not None:
        # Constrain predictions to map boundaries
        constrained_pred = constrain_predictions_to_map(pred_traj_abs, static_map)
        # Plot prediction line
        ax.plot(constrained_pred[:, 0], constrained_pred[:, 1], color='green', linewidth=2, zorder=10)
        # Plot prediction points with smaller markers
        ax.scatter(constrained_pred[:, 0], constrained_pred[:, 1], color='green', s=15, zorder=11, alpha=0.8, marker='o')
        # Add a larger marker for the final prediction
        if len(constrained_pred) > 0:
            ax.scatter(constrained_pred[-1, 0], constrained_pred[-1, 1], color='green', s=80, zorder=12, alpha=1.0, marker='*', edgecolors='black', linewidth=1)
        all_coords.append(constrained_pred[:, :2])

    if gt_traj_abs is not None:
        ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], color='red', linewidth=2.5, marker='s', markersize=8, zorder=11)
        all_coords.append(gt_traj_abs[:, :2])

    if all_coords:
        all_coords_np = np.vstack(all_coords)
        min_x, min_y = np.min(all_coords_np, axis=0)
        max_x, max_y = np.max(all_coords_np, axis=0)



        padding_x = (max_x - min_x) * 0.15
        padding_y = (max_y - min_y) * 0.15

        if padding_x < 10: padding_x = 10
        if padding_y < 10: padding_y = 10

        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)


    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    
    legend_elements = [
        mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Focal Agent Observed'),
        mlines.Line2D([], [], color='darkblue', marker='o', markersize=8, label='Focal Agent Last Observed', linestyle='None'),
        mlines.Line2D([], [], color='purple', marker='o', markersize=5, label='Other Agents Observed'),
        mlines.Line2D([], [], color='green', marker='*', markersize=10, label='Prediction', linestyle='None'),
    ]
    if gt_traj_abs is not None:
        legend_elements.append(mlines.Line2D([], [], color='red', marker='s', markersize=8, label='Ground Truth', linestyle='None'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[INFO] Saved plot to: {save_path}")
    else:
        plt.close()  # Always close figure to prevent memory leaks


def create_animation_for_scenario(
    scenario_path,
    static_map_path,
    pred_traj_abs,
    gt_traj_abs=None,
    title="Trajectory Animation on Argoverse2 Map",
    save_path=None
):
    """
    Create an animation of the scenario with predicted and ground truth trajectories.
    """
    try:
        scenario_path = Path(scenario_path)
        static_map_path = Path(static_map_path)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
    except Exception as e:
        print(f"Error loading scenario or map for animation: {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    for drivable_area in static_map.vector_drivable_areas.values():
        polygon_points = np.array([[point.x, point.y] for point in drivable_area.area_boundary])
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], color='#e0e0e0', alpha=0.5, zorder=0)

    for lane_segment in static_map.vector_lane_segments.values():
        left_boundary_xyz = lane_segment.left_lane_boundary.xyz
        right_boundary_xyz = lane_segment.right_lane_boundary.xyz
        ax.plot(left_boundary_xyz[:, 0], left_boundary_xyz[:, 1], '--', color='gray', alpha=0.7, linewidth=1, zorder=1)
        ax.plot(right_boundary_xyz[:, 0], right_boundary_xyz[:, 1], '--', color='gray', alpha=0.7, linewidth=1, zorder=1)

    all_coords = []
    for drivable_area in static_map.vector_drivable_areas.values():
        all_coords.append(np.array([[point.x, point.y] for point in drivable_area.area_boundary]))
    for lane_segment in static_map.vector_lane_segments.values():
        all_coords.append(lane_segment.left_lane_boundary.xyz[:, :2])
        all_coords.append(lane_segment.right_lane_boundary.xyz[:, :2])

    agent_plots = {}
    focal_track_id = scenario.focal_track_id
    for track in scenario.tracks:
        is_focal = track.track_id == focal_track_id
        color = 'blue' if is_focal else 'purple'
        lw = 2 if is_focal else 1
        alpha = 1.0 if is_focal else 0.7
        line, = ax.plot([], [], color=color, linewidth=lw, alpha=alpha, marker='o', markersize=2)
        agent_plots[track.track_id] = line
        all_coords.append(np.array([[os.position[0], os.position[1]] for os in track.object_states]))

    # Constrain predictions to map boundaries
    constrained_pred = constrain_predictions_to_map(pred_traj_abs, static_map)
    # Plot prediction line and points
    pred_line, = ax.plot(constrained_pred[:, 0], constrained_pred[:, 1], color='green', linewidth=2, zorder=10)
    pred_points = ax.scatter(constrained_pred[:, 0], constrained_pred[:, 1], color='green', s=15, zorder=11, alpha=0.8, marker='o')
    # Add final prediction marker
    if len(constrained_pred) > 0:
        final_pred = ax.scatter([constrained_pred[-1, 0]], [constrained_pred[-1, 1]], color='green', s=80, zorder=12, alpha=1.0, marker='*', edgecolors='black', linewidth=1)
    focal_last_observed_plot, = ax.plot([], [], 'o', color='darkblue', markersize=8, zorder=7)
    
    gt_line = None
    if gt_traj_abs is not None:
        ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], color='red', linewidth=2.5, marker='s', markersize=8, zorder=11, alpha=0.3)
        gt_line, = ax.plot([], [], color='red', linewidth=2.5, marker='s', markersize=8, zorder=11)

    all_coords.append(pred_traj_abs[:, :2])
    if gt_traj_abs is not None: all_coords.append(gt_traj_abs[:, :2])

    if all_coords:
        all_coords_np = np.vstack(all_coords)
        min_x, min_y = np.min(all_coords_np, axis=0)
        max_x, max_y = np.max(all_coords_np, axis=0)

        padding_x = (max_x - min_x) * 0.15
        padding_y = (max_y - min_y) * 0.15
        if padding_x < 10: padding_x = 10
        if padding_y < 10: padding_y = 10

        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    
    legend_elements = [
        mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Focal Agent'),
        mlines.Line2D([], [], color='darkblue', marker='o', markersize=8, label='Focal Agent Last Observed', linestyle='None'),
        mlines.Line2D([], [], color='purple', marker='o', markersize=5, label='Other Agents'),
        mlines.Line2D([], [], color='green', marker='*', markersize=10, label='Prediction', linestyle='None'),
    ]
    if gt_traj_abs is not None:
        legend_elements.append(mlines.Line2D([], [], color='red', marker='s', markersize=8, label='Ground Truth', linestyle='None'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        num_observed_steps = frame + 1
        for track in scenario.tracks:
            observed_states = [os for os in track.object_states if os.timestep < num_observed_steps]
            if observed_states:
                traj_points = np.array([[os.position[0], os.position[1]] for os in observed_states])
                agent_plots[track.track_id].set_data(traj_points[:, 0], traj_points[:, 1])
                if track.track_id == focal_track_id:
                    focal_last_observed_plot.set_data([traj_points[-1, 0]], [traj_points[-1, 1]])
            else:
                agent_plots[track.track_id].set_data([], [])
                if track.track_id == focal_track_id:
                    focal_last_observed_plot.set_data([], [])
        
        if gt_line and gt_traj_abs is not None:
            gt_line.set_data(gt_traj_abs[:frame+1, 0], gt_traj_abs[:frame+1, 1])

        return list(agent_plots.values()) + [pred_line, focal_last_observed_plot] + ([gt_line] if gt_line else [])

    max_timestep = 0
    for track in scenario.tracks:
        if track.object_states:
            max_timestep = max(max_timestep, max(os.timestep for os in track.object_states))
    
    num_frames = max_timestep + 1

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow', dpi=100)
        plt.close(fig)
        print(f"[INFO] Saved animation to: {save_path}")
    else:
        plt.close(fig)  # Always close figure to prevent memory leaks


def visualize_predictions(batch, model, device, save_dir, prefix="", is_test_mode=False, seq_len=30):
    """
    Enhanced prediction visualization with consolidated JSON output.
    This function assumes batch contains scenario_path and static_map_path attributes.
    """
    import os
    import json
    from datetime import datetime

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        batch = batch.to(device)
        outputs = model(batch)
        outputs = outputs.view(-1, seq_len, 2)
        pred = outputs.cpu().numpy()

        # CRITICAL FIX: Convert relative predictions back to absolute coordinates for visualization
        # The model predicts relative displacements, so we need to convert them back to absolute coordinates
        
        scenario_path = getattr(batch, 'scenario_path', [None])[0]
        static_map_path = getattr(batch, 'static_map_path', [None])[0]

        if scenario_path is None or static_map_path is None:
            print("[WARN] Batch missing scenario_path/static_map_path. Skipping visualization.")
            return
        
        # Get the last observed position of the focal agent from the scenario
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            focal_track = next(track for track in scenario.tracks if track.track_id == scenario.focal_track_id)
            
            # Find the last observed position
            last_observed_pos = None
            for object_state in focal_track.object_states:
                if object_state.observed:
                    last_observed_pos = object_state.position
            
            if last_observed_pos is not None:
                # Convert relative predictions to absolute coordinates
                # pred contains relative displacements (normalized by 10.0)
                relative_displacements = pred[0] * 10.0  # Denormalize
                
                # Convert to absolute coordinates
                absolute_pred = np.zeros_like(relative_displacements)
                current_pos = np.array([last_observed_pos[0], last_observed_pos[1]])
                
                for i in range(len(relative_displacements)):
                    absolute_pred[i] = current_pos + relative_displacements[i]
                    current_pos = absolute_pred[i]  # Update for next step
                
                pred[0] = absolute_pred
                print(f"[INFO] Converted relative predictions to absolute coordinates. Range: [{absolute_pred.min():.2f}, {absolute_pred.max():.2f}]")
            else:
                print("[WARNING] Could not find last observed position for focal agent")
                
        except Exception as e:
            print(f"[WARNING] Could not convert relative predictions to absolute coordinates: {e}")
            # Fallback: use predictions as-is
            pass

        # Extract sample ID from prefix
        sample_id = 0
        try:
            import re
            match = re.search(r'sample_(\d+)', prefix)
            if match:
                sample_id = int(match.group(1))
        except:
            pass

        # Create comprehensive prediction data
        prediction_data = {
            'sample_id': sample_id,
            'scenario_path': str(scenario_path),
            'static_map_path': str(static_map_path),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': pred[0].tolist(),  # First prediction in batch
            'sequence_length': seq_len,
            'prediction_horizon': len(pred[0]),
            'model_confidence': float(np.mean(np.linalg.norm(pred[0], axis=1))),  # Simple confidence metric
        }

        gt = None
        if not is_test_mode:
            try:
                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                focal_track = next(track for track in scenario.tracks if track.track_id == scenario.focal_track_id)

                gt_traj_points = [
                    object_state.position for object_state in focal_track.object_states
                    if not object_state.observed
                ]
                if gt_traj_points:
                    gt = np.array(gt_traj_points)
                    prediction_data['ground_truth'] = gt.tolist()

                    # Calculate prediction error metrics
                    if len(gt) >= len(pred[0]):
                        gt_truncated = gt[:len(pred[0])]
                        ade = np.mean(np.linalg.norm(pred[0] - gt_truncated, axis=1))
                        fde = np.linalg.norm(pred[0][-1] - gt_truncated[-1])
                        prediction_data['metrics'] = {
                            'ade': float(ade),
                            'fde': float(fde),
                            'prediction_length': len(pred[0]),
                            'gt_length': len(gt)
                        }

            except Exception as e:
                print(f"[WARN] Could not load ground truth for visualization from {scenario_path}: {e}")
                prediction_data['ground_truth'] = None
                prediction_data['metrics'] = None

        # Save individual prediction JSON (will be consolidated later)
        pred_save_path = os.path.join(save_dir, f"{prefix}predictions.json")
        with open(pred_save_path, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        print(f"[INFO] Saved prediction data to: {pred_save_path}")

        # Generate visualization
        save_path = os.path.join(save_dir, f"{prefix}prediction.png")
        plot_argoverse2_with_prediction(
            scenario_path=scenario_path,
            static_map_path=static_map_path,
            pred_traj_abs=pred[0],
            gt_traj_abs=gt,
            title=f"Trajectory Prediction - Sample {sample_id}",
            save_path=save_path
        )

        # Generate animation for each sample (with proper naming)
        anim_save_path = os.path.join(save_dir, f"{prefix}animation.gif")
        create_animation_for_scenario(
            scenario_path=scenario_path,
            static_map_path=static_map_path,
            pred_traj_abs=pred[0],
            gt_traj_abs=gt,
            title=f"Trajectory Animation - Sample {sample_id}",
            save_path=anim_save_path
        )


        return prediction_data