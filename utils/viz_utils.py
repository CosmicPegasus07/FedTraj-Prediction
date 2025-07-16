import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.animation import FuncAnimation # Added for animation

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

    # Collect all coordinates to set dynamic plot limits
    all_coords = []

    # Plot map elements
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

    # Plot agent trajectories
    focal_track_id = scenario.focal_track_id
    other_agents_plotted = False # Flag to ensure 'Other Agents Observed' is only in legend once
    for track in scenario.tracks:
        observed_traj_points = []
        for object_state in track.object_states:
            if object_state.observed:
                observed_traj_points.append([object_state.position[0], object_state.position[1]])
        
        if not observed_traj_points:
            continue
            
        observed_traj = np.array(observed_traj_points)
        all_coords.append(observed_traj)
        
        is_focal = track.track_id == focal_track_id
        color = 'blue' if is_focal else 'purple'
        lw = 2 if is_focal else 1
        alpha = 1.0 if is_focal else 0.7
        label = 'Focal Agent Observed' if is_focal else None

        if not is_focal and not other_agents_plotted:
            label = 'Other Agents Observed'
            other_agents_plotted = True

        ax.plot(observed_traj[:, 0], observed_traj[:, 1], color=color, linewidth=lw, alpha=alpha, marker='o', markersize=2, label=label)
        if is_focal and len(observed_traj) > 0:
            ax.plot(observed_traj[-1, 0], observed_traj[-1, 1], 'o', color='darkblue', markersize=8, zorder=6, label='Focal Agent Last Observed')

    # Plot predictions and ground truth
    if pred_traj_abs is not None:
        ax.plot(pred_traj_abs[:, 0], pred_traj_abs[:, 1], color='green', linewidth=2.5, marker='*', markersize=10, label='Prediction', zorder=10)
        all_coords.append(pred_traj_abs[:, :2])

    if gt_traj_abs is not None:
        ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], color='red', linewidth=2.5, marker='s', markersize=8, label='Ground Truth', zorder=11)
        all_coords.append(gt_traj_abs[:, :2])

    # Set dynamic plot limits
    if all_coords:
        all_coords_np = np.vstack(all_coords)
        min_x, min_y = np.min(all_coords_np, axis=0)
        max_x, max_y = np.max(all_coords_np, axis=0)

        # Add padding
        padding_x = (max_x - min_x) * 0.15 # 15% padding
        padding_y = (max_y - min_y) * 0.15 # 15% padding
        
        # Ensure a minimum padding if the range is very small
        if padding_x < 10: padding_x = 10
        if padding_y < 10: padding_y = 10

        ax.set_xlim(min_x - padding_x, max_x + padding_x)
        ax.set_ylim(min_y - padding_y, max_y + padding_y)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

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

    # Plot static map elements
    for drivable_area in static_map.vector_drivable_areas.values():
        polygon_points = np.array([[point.x, point.y] for point in drivable_area.area_boundary])
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], color='#e0e0e0', alpha=0.5, zorder=0)

    for lane_segment in static_map.vector_lane_segments.values():
        left_boundary_xyz = lane_segment.left_lane_boundary.xyz
        right_boundary_xyz = lane_segment.right_lane_boundary.xyz
        ax.plot(left_boundary_xyz[:, 0], left_boundary_xyz[:, 1], '--', color='gray', alpha=0.7, linewidth=1, zorder=1)
        ax.plot(right_boundary_xyz[:, 0], right_boundary_xyz[:, 1], '--', color='gray', alpha=0.7, linewidth=1, zorder=1)

    # Collect all coordinates for dynamic plot limits
    all_coords = []
    for drivable_area in static_map.vector_drivable_areas.values():
        all_coords.append(np.array([[point.x, point.y] for point in drivable_area.area_boundary]))
    for lane_segment in static_map.vector_lane_segments.values():
        all_coords.append(lane_segment.left_lane_boundary.xyz[:, :2])
        all_coords.append(lane_segment.right_lane_boundary.xyz[:, :2])

    # Initialize agent plots
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

    # Prediction and Ground Truth plots (static for animation)
    pred_line, = ax.plot(pred_traj_abs[:, 0], pred_traj_abs[:, 1], color='green', linewidth=2.5, marker='*', markersize=10, label='Prediction', zorder=10)
    if gt_traj_abs is not None:
        gt_line, = ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], color='red', linewidth=2.5, marker='s', markersize=8, label='Ground Truth', zorder=11)
    else:
        gt_line = None

    all_coords.append(pred_traj_abs[:, :2])
    if gt_traj_abs is not None: all_coords.append(gt_traj_abs[:, :2])

    # Set dynamic plot limits
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
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        for track in scenario.tracks:
            # Only plot observed states up to the current frame
            current_states = [os for os in track.object_states if os.timestep <= frame and os.observed]
            if current_states:
                traj_points = np.array([[os.position[0], os.position[1]] for os in current_states])
                agent_plots[track.track_id].set_data(traj_points[:, 0], traj_points[:, 1])
            else:
                agent_plots[track.track_id].set_data([], []) # Clear if no states yet
        return list(agent_plots.values()) + [pred_line] + ([gt_line] if gt_line else [])

    # Determine the number of frames (timesteps) for the animation
    max_timestep = 0
    for track in scenario.tracks:
        for object_state in track.object_states:
            if object_state.timestep > max_timestep:
                max_timestep = object_state.timestep
    num_frames = max_timestep + 1 # Assuming timesteps start from 0

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow', dpi=100) # Lower DPI for GIF to save size
        plt.close(fig)
    else:
        plt.show()

def visualize_predictions(batch, model, device, save_dir, prefix="", is_test_mode=False):
    """
    Generate and save a prediction visualization for a batch.
    This function assumes batch contains scenario_path and static_map_path attributes.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index)
        # Get focal agent prediction (assume batch_size=1 for visualization)
        focal_global_idx = batch.ptr[:-1] + batch.focal_idx
        pred = outputs[focal_global_idx].cpu().numpy()
        gt = batch.y.view(-1, 2).cpu().numpy() if not is_test_mode else None
        
        # Get scenario/map paths from batch. These are lists, so we take the first element.
        scenario_path = getattr(batch, 'scenario_path', [None])[0]
        static_map_path = getattr(batch, 'static_map_path', [None])[0]

        if scenario_path is None or static_map_path is None:
            print("[WARN] Batch missing scenario_path/static_map_path. Skipping visualization.")
            return
            
        save_path = os.path.join(save_dir, f"{prefix}prediction.png")
        plot_argoverse2_with_prediction(
            scenario_path=scenario_path,
            static_map_path=static_map_path,
            pred_traj_abs=pred,
            gt_traj_abs=gt,
            title="Trajectory Prediction",
            save_path=save_path
        )
        print(f"[INFO] Saved visualization: {save_path}")

        # Also create an animation
        anim_save_path = os.path.join(save_dir, f"{prefix}animation.gif")
        create_animation_for_scenario(
            scenario_path=scenario_path,
            static_map_path=static_map_path,
            pred_traj_abs=pred,
            gt_traj_abs=gt,
            title="Trajectory Animation",
            save_path=anim_save_path
        )
        print(f"[INFO] Saved animation: {anim_save_path}")