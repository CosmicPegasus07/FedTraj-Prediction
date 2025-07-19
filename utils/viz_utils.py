import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines

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
        ax.plot(pred_traj_abs[:, 0], pred_traj_abs[:, 1], color='green', linewidth=2.5, marker='*', markersize=10, zorder=10)
        all_coords.append(pred_traj_abs[:, :2])

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

    pred_line, = ax.plot(pred_traj_abs[:, 0], pred_traj_abs[:, 1], color='green', linewidth=2.5, marker='*', markersize=10, zorder=10)
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
        focal_global_idx = batch.ptr[:-1] + batch.focal_idx
        pred = outputs[focal_global_idx].cpu().numpy()
        
        scenario_path = getattr(batch, 'scenario_path', [None])[0]
        static_map_path = getattr(batch, 'static_map_path', [None])[0]

        if scenario_path is None or static_map_path is None:
            print("[WARN] Batch missing scenario_path/static_map_path. Skipping visualization.")
            return

        gt = None
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            focal_track = next(track for track in scenario.tracks if track.track_id == scenario.focal_track_id)
            
            gt_traj_points = [
                object_state.position for object_state in focal_track.object_states
                if not object_state.observed
            ]
            if gt_traj_points:
                gt = np.array(gt_traj_points)
        except Exception as e:
            print(f"[WARN] Could not load ground truth for visualization from {scenario_path}: {e}")

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