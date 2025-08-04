import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch, Polygon
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.transforms as transforms

def get_predicted_action_text(pred_traj, last_observed_state):
    """
    Infer a human-readable action from the predicted trajectory.
    """
    if pred_traj is None or last_observed_state is None:
        return "Unknown"

    # Get start and end points of the prediction
    start_pos = last_observed_state.position
    start_heading = last_observed_state.heading
    end_pos = pred_traj[-1]

    # Calculate displacement vector
    displacement = end_pos - start_pos
    
    # Rotate displacement vector to be relative to the agent's start heading
    # This aligns the coordinate system with the agent's perspective
    cos_h = np.cos(-start_heading)
    sin_h = np.sin(-start_heading)
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h, cos_h]
    ])
    
    # Apply rotation
    local_displacement = rotation_matrix.dot(displacement)
    
    # Lateral displacement (left/right)
    lateral_dist = local_displacement[1]
    
    # Longitudinal displacement (forward/backward)
    longitudinal_dist = local_displacement[0]

    # Calculate final heading from the last few points of the trajectory
    if len(pred_traj) > 5:
        final_heading_vec = pred_traj[-1] - pred_traj[-6]
        final_heading = np.arctan2(final_heading_vec[1], final_heading_vec[0])
    else:
        final_heading = start_heading # Assume same heading if trajectory is too short

    # Calculate change in heading
    heading_change = final_heading - start_heading
    # Normalize heading change to be within [-pi, pi]
    heading_change = (heading_change + np.pi) % (2 * np.pi) - np.pi

    # Define thresholds for classification
    lat_thresh_lane_change = 2.0  # meters
    turn_thresh = np.pi / 6  # 30 degrees
    
    # Classify action based on displacement and heading change
    if abs(heading_change) > turn_thresh:
        if heading_change > 0:
            return "Turn Left"
        else:
            return "Turn Right"
    elif lateral_dist > lat_thresh_lane_change:
        return "Lane Change Left"
    elif lateral_dist < -lat_thresh_lane_change:
        return "Lane Change Right"
    elif longitudinal_dist < 2.0: # If the vehicle has barely moved forward
        return "Slowing Down / Stop"
    else:
        return "Following Lane"

def create_car_patch(x, y, heading, length=4.5, width=2.0, color='blue', alpha=1.0, is_focal=False):
    """
    Create a car-shaped patch with proper orientation
    Args:
        x, y: position coordinates
        heading: heading angle in radians
        length: car length in meters
        width: car width in meters  
        color: car color
        alpha: transparency
        is_focal: if True, make it larger and more prominent
    """
    if is_focal:
        length *= 1.3
        width *= 1.3
    
    # Create car shape as a rectangle with rounded corners
    car_patch = FancyBboxPatch(
        (-length/2, -width/2), length, width,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5 if is_focal else 1.0,
        alpha=alpha,
        zorder=10 if is_focal else 8
    )
    
    # Apply rotation and translation
    t = transforms.Affine2D().rotate(heading).translate(x, y)
    car_patch.set_transform(t + plt.gca().transData)
    
    return car_patch

def create_direction_arrow(x, y, heading, length=3.0, color='white', alpha=0.8):
    """
    Create a direction arrow to show vehicle orientation
    """
    arrow_length = length
    dx = arrow_length * np.cos(heading)
    dy = arrow_length * np.sin(heading)
    
    # Create arrow pointing in direction of heading
    arrow = plt.annotate('', xy=(x + dx/2, y + dy/2), xytext=(x - dx/2, y - dy/2),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=alpha),
                        zorder=11)
    return arrow

def get_optimal_view_bounds(all_coords, zoom_factor=0.8):
    """
    Calculate optimal view bounds for better visualization
    Args:
        all_coords: array of all coordinate points
        zoom_factor: factor to zoom in (0.8 = 20% closer than default)
    """
    if len(all_coords) == 0:
        return None
        
    all_coords_np = np.vstack(all_coords)
    min_x, min_y = np.min(all_coords_np, axis=0)
    max_x, max_y = np.max(all_coords_np, axis=0)
    
    # Calculate center and range
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    range_x = (max_x - min_x) * zoom_factor
    range_y = (max_y - min_y) * zoom_factor
    
    # Ensure minimum range for readability
    min_range = 30  # meters
    range_x = max(range_x, min_range)
    range_y = max(range_y, min_range)
    
    # Add some padding
    padding_factor = 0.15
    padding_x = range_x * padding_factor
    padding_y = range_y * padding_factor
    
    return {
        'xlim': (center_x - range_x/2 - padding_x, center_x + range_x/2 + padding_x),
        'ylim': (center_y - range_y/2 - padding_y, center_y + range_y/2 + padding_y)
    }

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

def plot_argoverse2_with_prediction_enhanced(
    scenario_path,
    static_map_path,
    pred_traj_abs,
    gt_traj_abs=None,
    title="Trajectory Prediction on Argoverse2 Map",
    save_path=None,
    use_enhanced_visuals=True,
    zoom_factor=0.7,
    predicted_action_text=None
):
    """
    Enhanced plot with car representations and better view
    """
    try:
        scenario_path = Path(scenario_path)
        static_map_path = Path(static_map_path)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
    except Exception as e:
        print(f"Error loading scenario or map: {e}")
        return

    fig, ax = plt.subplots(figsize=(14, 14))  # Larger figure for better detail

    all_coords = []

    # Draw map with better styling
    for drivable_area in static_map.vector_drivable_areas.values():
        polygon_points = np.array([[point.x, point.y] for point in drivable_area.area_boundary])
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], 
                color='#f0f0f0', alpha=0.6, zorder=0, edgecolor='#d0d0d0', linewidth=0.5)
        all_coords.append(polygon_points)

    # Draw lanes with better styling
    for lane_segment in static_map.vector_lane_segments.values():
        left_boundary_xyz = lane_segment.left_lane_boundary.xyz
        right_boundary_xyz = lane_segment.right_lane_boundary.xyz
        ax.plot(left_boundary_xyz[:, 0], left_boundary_xyz[:, 1], 
                '--', color='#808080', alpha=0.8, linewidth=1.2, zorder=1)
        ax.plot(right_boundary_xyz[:, 0], right_boundary_xyz[:, 1], 
                '--', color='#808080', alpha=0.8, linewidth=1.2, zorder=1)
        all_coords.append(left_boundary_xyz[:, :2])
        all_coords.append(right_boundary_xyz[:, :2])

    focal_track_id = scenario.focal_track_id
    
    # Process each track with enhanced visuals
    for track in scenario.tracks:
        observed_states = [os for os in track.object_states if os.observed]
        
        if not observed_states:
            continue
            
        observed_traj = np.array([[os.position[0], os.position[1]] for os in observed_states])
        all_coords.append(observed_traj)
        
        is_focal = track.track_id == focal_track_id
        
        if use_enhanced_visuals:
            # Draw trajectory path
            path_color = '#1f77b4' if is_focal else '#9467bd'  # Better colors
            path_alpha = 0.8 if is_focal else 0.6
            ax.plot(observed_traj[:, 0], observed_traj[:, 1], 
                   color=path_color, linewidth=2.5 if is_focal else 1.5, 
                   alpha=path_alpha, zorder=3)
            
            # Add car representations at key points
            num_cars_to_show = min(5, len(observed_states))  # Show max 5 cars to avoid clutter
            indices = np.linspace(0, len(observed_states)-1, num_cars_to_show, dtype=int)
            
            for i, idx in enumerate(indices):
                os = observed_states[idx]
                car_color = '#1f77b4' if is_focal else '#9467bd'
                car_alpha = 1.0 if i == len(indices)-1 else 0.7  # Last car more prominent
                
                # Create car patch
                car_patch = create_car_patch(
                    os.position[0], os.position[1], os.heading,
                    color=car_color, alpha=car_alpha, is_focal=is_focal
                )
                ax.add_patch(car_patch)
                
                # Add direction arrow for the last observed position
                if i == len(indices)-1:
                    create_direction_arrow(
                        os.position[0], os.position[1], os.heading,
                        color='white' if is_focal else 'lightgray'
                    )
        else:
            # Fallback to original simple visualization
            color = 'blue' if is_focal else 'purple'
            lw = 2 if is_focal else 1
            alpha = 1.0 if is_focal else 0.7
            ax.plot(observed_traj[:, 0], observed_traj[:, 1], 
                   color=color, linewidth=lw, alpha=alpha, marker='o', markersize=2)
            if is_focal and len(observed_traj) > 0:
                ax.plot(observed_traj[-1, 0], observed_traj[-1, 1], 
                       'o', color='darkblue', markersize=8, zorder=6)

    # Enhanced prediction visualization
    if pred_traj_abs is not None:
        constrained_pred = constrain_predictions_to_map(pred_traj_abs, static_map)
        
        if use_enhanced_visuals:
            # Draw prediction path with gradient effect
            ax.plot(constrained_pred[:, 0], constrained_pred[:, 1], 
                   color='#2ca02c', linewidth=3, zorder=10, alpha=0.9,
                   linestyle='-', marker='o', markersize=4, markerfacecolor='#2ca02c',
                   markeredgecolor='green', markeredgewidth=1)
            
            # Highlight final prediction with a special marker
            if len(constrained_pred) > 0:
                ax.scatter(constrained_pred[-1, 0], constrained_pred[-1, 1], 
                          color='#ff7f0e', s=120, zorder=12, alpha=1.0, 
                          marker='*', edgecolors='black', linewidth=2,
                          label='Final Prediction')
        else:
            # Fallback to original visualization
            ax.plot(constrained_pred[:, 0], constrained_pred[:, 1], 
                   color='green', linewidth=2, zorder=10)
            ax.scatter(constrained_pred[:, 0], constrained_pred[:, 1], 
                      color='green', s=15, zorder=11, alpha=0.8, marker='o')
            if len(constrained_pred) > 0:
                ax.scatter(constrained_pred[-1, 0], constrained_pred[-1, 1], 
                          color='green', s=80, zorder=12, alpha=1.0, 
                          marker='*', edgecolors='black', linewidth=1)
        
        all_coords.append(constrained_pred[:, :2])

    # Enhanced ground truth visualization
    if gt_traj_abs is not None:
        if use_enhanced_visuals:
            ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], 
                   color='#d62728', linewidth=3, alpha=0.8, zorder=11,
                   linestyle='--', marker='s', markersize=6,
                   markerfacecolor='#d62728', markeredgecolor='white', 
                   markeredgewidth=1, label='Ground Truth')
        else:
            ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], 
                   color='red', linewidth=2.5, marker='s', markersize=8, zorder=11)
        all_coords.append(gt_traj_abs[:, :2])

    # Use optimal view bounds for better zoom
    if all_coords:
        view_bounds = get_optimal_view_bounds(all_coords, zoom_factor)
        if view_bounds:
            ax.set_xlim(view_bounds['xlim'])
            ax.set_ylim(view_bounds['ylim'])

    # Enhanced styling
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Enhanced legend
    if use_enhanced_visuals:
        legend_elements = [
            mlines.Line2D([], [], color='#1f77b4', linewidth=3, label='Focal Agent Path'),
            mlines.Line2D([], [], color='#9467bd', linewidth=2, label='Other Agents Path'),
            mlines.Line2D([], [], color='#2ca02c', linewidth=3, label='Prediction'),
        ]
        if gt_traj_abs is not None:
            legend_elements.append(mlines.Line2D([], [], color='#d62728', linewidth=3, 
                                               linestyle='--', label='Ground Truth'))
    else:
        # Original legend for backward compatibility
        legend_elements = [
            mlines.Line2D([], [], color='blue', marker='o', markersize=5, label='Focal Agent Observed'),
            mlines.Line2D([], [], color='darkblue', marker='o', markersize=8, label='Focal Agent Last Observed', linestyle='None'),
            mlines.Line2D([], [], color='purple', marker='o', markersize=5, label='Other Agents Observed'),
            mlines.Line2D([], [], color='green', marker='*', markersize=10, label='Prediction', linestyle='None'),
        ]
        if gt_traj_abs is not None:
            legend_elements.append(mlines.Line2D([], [], color='red', marker='s', markersize=8, label='Ground Truth', linestyle='None'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add predicted action text
    if predicted_action_text:
        ax.text(0.5, 0.02, f"Predicted Action: {predicted_action_text}", 
                ha='center', va='bottom', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6))

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)  # Higher DPI for better quality
        plt.close()
        print(f"[INFO] Saved enhanced plot to: {save_path}")
    else:
        plt.close()

def plot_argoverse2_with_prediction(
    scenario_path,
    static_map_path,
    pred_traj_abs,
    gt_traj_abs=None,
    title="Trajectory Prediction on Argoverse2 Map",
    save_path=None,
    use_enhanced_visuals=True,
    predicted_action_text=None
):
    """
    Plot an Argoverse2 scenario with map, agents, and overlay your prediction.
    Now defaults to enhanced visuals but maintains backward compatibility.
    """
    if use_enhanced_visuals:
        # Use the enhanced version by default
        return plot_argoverse2_with_prediction_enhanced(
            scenario_path, static_map_path, pred_traj_abs, gt_traj_abs, 
            title, save_path, use_enhanced_visuals=True, predicted_action_text=predicted_action_text
        )
    
    # Original implementation for backward compatibility
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
    
    # Add predicted action text
    if predicted_action_text:
        ax.text(0.5, 0.02, f"Predicted Action: {predicted_action_text}", 
                ha='center', va='bottom', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[INFO] Saved plot to: {save_path}")
    else:
        plt.close()  # Always close figure to prevent memory leaks


def create_animation_for_scenario_enhanced(
    scenario_path,
    static_map_path,
    pred_traj_abs,
    gt_traj_abs=None,
    title="Trajectory Animation on Argoverse2 Map", 
    save_path=None,
    use_enhanced_visuals=True,
    zoom_factor=0.7,
    predicted_action_text=None
):
    """
    Enhanced animation with car representations and better visuals
    """
    try:
        scenario_path = Path(scenario_path)
        static_map_path = Path(static_map_path)
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
    except Exception as e:
        print(f"Error loading scenario or map for animation: {e}")
        return

    fig, ax = plt.subplots(figsize=(14, 14))

    # Draw static map elements with enhanced styling
    for drivable_area in static_map.vector_drivable_areas.values():
        polygon_points = np.array([[point.x, point.y] for point in drivable_area.area_boundary])
        ax.fill(polygon_points[:, 0], polygon_points[:, 1], 
                color='#f0f0f0', alpha=0.6, zorder=0, edgecolor='#d0d0d0', linewidth=0.5)

    for lane_segment in static_map.vector_lane_segments.values():
        left_boundary_xyz = lane_segment.left_lane_boundary.xyz
        right_boundary_xyz = lane_segment.right_lane_boundary.xyz
        ax.plot(left_boundary_xyz[:, 0], left_boundary_xyz[:, 1], 
                '--', color='#808080', alpha=0.8, linewidth=1.2, zorder=1)
        ax.plot(right_boundary_xyz[:, 0], right_boundary_xyz[:, 1], 
                '--', color='#808080', alpha=0.8, linewidth=1.2, zorder=1)

    all_coords = []
    for drivable_area in static_map.vector_drivable_areas.values():
        all_coords.append(np.array([[point.x, point.y] for point in drivable_area.area_boundary]))
    for lane_segment in static_map.vector_lane_segments.values():
        all_coords.append(lane_segment.left_lane_boundary.xyz[:, :2])
        all_coords.append(lane_segment.right_lane_boundary.xyz[:, :2])

    # Prepare agent animation elements
    agent_paths = {}
    agent_cars = {}
    agent_arrows = {}
    focal_track_id = scenario.focal_track_id
    
    for track in scenario.tracks:
        is_focal = track.track_id == focal_track_id
        color = '#1f77b4' if is_focal else '#9467bd'
        lw = 2.5 if is_focal else 1.5
        alpha = 0.8 if is_focal else 0.6
        
        # Create path line for animation
        line, = ax.plot([], [], color=color, linewidth=lw, alpha=alpha, zorder=3)
        agent_paths[track.track_id] = line
        
        # Store agent car patches and arrows (will be created/updated during animation)
        agent_cars[track.track_id] = None
        agent_arrows[track.track_id] = None
        
        all_coords.append(np.array([[os.position[0], os.position[1]] for os in track.object_states]))

    # Draw static prediction and ground truth
    constrained_pred = constrain_predictions_to_map(pred_traj_abs, static_map)
    if use_enhanced_visuals:
        ax.plot(constrained_pred[:, 0], constrained_pred[:, 1], 
               color='#2ca02c', linewidth=3, zorder=10, alpha=0.9,
               linestyle='-', marker='o', markersize=4, markerfacecolor='#2ca02c',
               markeredgecolor='green', markeredgewidth=1)
        if len(constrained_pred) > 0:
            ax.scatter(constrained_pred[-1, 0], constrained_pred[-1, 1], 
                      color='#ff7f0e', s=120, zorder=12, alpha=1.0, 
                      marker='*', edgecolors='black', linewidth=2)
    else:
        ax.plot(constrained_pred[:, 0], constrained_pred[:, 1], color='green', linewidth=2, zorder=10)
        ax.scatter(constrained_pred[:, 0], constrained_pred[:, 1], color='green', s=15, zorder=11, alpha=0.8, marker='o')
        if len(constrained_pred) > 0:
            ax.scatter(constrained_pred[-1, 0], constrained_pred[-1, 1], color='green', s=80, zorder=12, alpha=1.0, marker='*', edgecolors='black', linewidth=1)
    
    if gt_traj_abs is not None:
        if use_enhanced_visuals:
            ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], 
                   color='#d62728', linewidth=3, alpha=0.4, zorder=11,
                   linestyle='--', marker='s', markersize=4)
        else:
            ax.plot(gt_traj_abs[:, 0], gt_traj_abs[:, 1], color='red', linewidth=2.5, marker='s', markersize=8, zorder=11, alpha=0.3)

    all_coords.append(pred_traj_abs[:, :2])
    if gt_traj_abs is not None: 
        all_coords.append(gt_traj_abs[:, :2])

    # Set optimal view bounds
    if all_coords:
        view_bounds = get_optimal_view_bounds(all_coords, zoom_factor)
        if view_bounds:
            ax.set_xlim(view_bounds['xlim'])
            ax.set_ylim(view_bounds['ylim'])

    # Enhanced styling
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Enhanced legend
    legend_elements = [
        mlines.Line2D([], [], color='#1f77b4', linewidth=3, label='Focal Agent'),
        mlines.Line2D([], [], color='#9467bd', linewidth=2, label='Other Agents'),
        mlines.Line2D([], [], color='#2ca02c', linewidth=3, label='Prediction'),
    ]
    if gt_traj_abs is not None:
        legend_elements.append(mlines.Line2D([], [], color='#d62728', linewidth=3, linestyle='--', label='Ground Truth'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add predicted action text
    if predicted_action_text:
        ax.text(0.5, 0.02, f"Predicted Action: {predicted_action_text}", 
                ha='center', va='bottom', transform=ax.transAxes, 
                fontsize=14, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6))

    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        num_observed_steps = frame + 1
        
        for track in scenario.tracks:
            observed_states = [os for os in track.object_states if os.timestep < num_observed_steps]
            is_focal = track.track_id == focal_track_id
            
            if observed_states:
                traj_points = np.array([[os.position[0], os.position[1]] for os in observed_states])
                agent_paths[track.track_id].set_data(traj_points[:, 0], traj_points[:, 1])
                
                # Update car representation for current position
                if use_enhanced_visuals and len(observed_states) > 0:
                    current_state = observed_states[-1]
                    
                    # Remove old car and arrow if they exist
                    if agent_cars[track.track_id] is not None:
                        agent_cars[track.track_id].remove()
                    if agent_arrows[track.track_id] is not None:
                        agent_arrows[track.track_id].remove()
                    
                    # Create new car patch for current position
                    car_color = '#1f77b4' if is_focal else '#9467bd'
                    agent_cars[track.track_id] = create_car_patch(
                        current_state.position[0], current_state.position[1], current_state.heading,
                        color=car_color, alpha=1.0, is_focal=is_focal
                    )
                    ax.add_patch(agent_cars[track.track_id])
                    
                    # Create direction arrow
                    agent_arrows[track.track_id] = create_direction_arrow(
                        current_state.position[0], current_state.position[1], current_state.heading,
                        color='white' if is_focal else 'lightgray'
                    )
            else:
                agent_paths[track.track_id].set_data([], [])
                # Remove car and arrow if no observed states
                if agent_cars[track.track_id] is not None:
                    agent_cars[track.track_id].remove()
                    agent_cars[track.track_id] = None
                if agent_arrows[track.track_id] is not None:
                    agent_arrows[track.track_id].remove()
                    agent_arrows[track.track_id] = None

        return list(agent_paths.values()) + [p for p in agent_cars.values() if p is not None]

    # Calculate animation frames
    max_timestep = 0
    for track in scenario.tracks:
        if track.object_states:
            max_timestep = max(max_timestep, max(os.timestep for os in track.object_states))
    
    num_frames = max_timestep + 1
    anim = FuncAnimation(fig, update, frames=num_frames, interval=150, blit=False)  # Slightly slower for better viewing

    if save_path:
        anim.save(save_path, writer='pillow', dpi=150)  # Higher DPI for better quality
        plt.close(fig)
        print(f"[INFO] Saved enhanced animation to: {save_path}")
    else:
        plt.close(fig)

def create_animation_for_scenario(
    scenario_path,
    static_map_path,
    pred_traj_abs,
    gt_traj_abs=None,
    title="Trajectory Animation on Argoverse2 Map",
    save_path=None,
    use_enhanced_visuals=True,
    predicted_action_text=None
):
    """
    Create an animation of the scenario with predicted and ground truth trajectories.
    Now defaults to enhanced visuals but maintains backward compatibility.
    """
    if use_enhanced_visuals:
        # Use the enhanced version by default
        return create_animation_for_scenario_enhanced(
            scenario_path, static_map_path, pred_traj_abs, gt_traj_abs, 
            title, save_path, use_enhanced_visuals=True, predicted_action_text=predicted_action_text
        )
    
    # Original implementation for backward compatibility 
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
    
    # Add predicted action text
    if predicted_action_text:
        ax.text(0.5, 0.02, f"Predicted Action: {predicted_action_text}", 
                ha='center', va='bottom', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

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
        
        last_observed_state = None
        # Get the last observed position of the focal agent from the scenario
        try:
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            focal_track = next(track for track in scenario.tracks if track.track_id == scenario.focal_track_id)
            
            # Find the last observed position
            last_observed_pos = None
            for object_state in focal_track.object_states:
                if object_state.observed:
                    last_observed_pos = object_state.position
                    last_observed_state = object_state
            
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

        # Get predicted action text
        predicted_action_text = get_predicted_action_text(pred[0], last_observed_state)

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
            'predicted_action': predicted_action_text
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
            save_path=save_path,
            predicted_action_text=predicted_action_text
        )

        # Generate animation for each sample (with proper naming)
        anim_save_path = os.path.join(save_dir, f"{prefix}animation.gif")
        create_animation_for_scenario(
            scenario_path=scenario_path,
            static_map_path=static_map_path,
            pred_traj_abs=pred[0],
            gt_traj_abs=gt,
            title=f"Trajectory Animation - Sample {sample_id}",
            save_path=anim_save_path,
            predicted_action_text=predicted_action_text
        )


        return prediction_data

def consolidate_predictions(predictions_dir, output_file):
    """

    Consolidate all individual prediction JSON files into a single file.
    """
    import json
    import glob

    all_predictions = []
    json_files = glob.glob(os.path.join(predictions_dir, "*_predictions.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_predictions.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    # Sort by sample_id for consistent output
    all_predictions.sort(key=lambda x: x.get('sample_id', 0))
    
    # Write to consolidated file
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
        
    print(f"[INFO] Consolidated {len(all_predictions)} predictions into {output_file}")
    
    # Clean up individual JSON files
    for file_path in json_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
            
    print("[INFO] Cleaned up individual prediction JSON files.")
