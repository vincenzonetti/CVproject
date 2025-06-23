"""
3D interactive visualization for basketball tracking.

This module creates interactive 3D plots using Plotly to visualize
player trajectories on the basketball court.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PLOTLY_COLORS, AXIS_LABELS
from tracking.metrics import extract_trajectories
from utils.court import create_court_grid


def create_interactive_3d_plot(tracking_3d_results: Dict, metrics: Dict, 
                               court_corners: Optional[np.ndarray], 
                               save_path: str) -> None:
    """
    Create an interactive 3D plot with player trajectories.
    
    Args:
        tracking_3d_results: Dictionary with frame-wise 3D positions
        metrics: Dictionary of calculated metrics for each player
        court_corners: Optional array of court corner coordinates
        save_path: Path to save the HTML file
    """
    # Extract trajectories
    trajectories = extract_trajectories(tracking_3d_results)
    
    # Debug information
    print_trajectory_debug_info(trajectories)
    
    # Create figure
    fig = go.Figure()
    
    # Add court visualization if available
    if court_corners is not None:
        add_court_to_figure(fig, court_corners)
    
    # Keep track of court traces
    num_court_traces = len(fig.data)
    
    # Add player trajectories
    player_traces_added = add_player_trajectories(fig, trajectories, metrics)
    
    print(f"\nAdded traces for {len(player_traces_added)} players: {player_traces_added}")
    
    # Configure layout
    configure_3d_layout(fig, num_court_traces)
    
    # Save the plot
    fig.write_html(save_path)
    print(f"Interactive 3D plot saved to {save_path}")


def print_trajectory_debug_info(trajectories: Dict) -> None:
    """Print debug information about trajectories."""
    print("\n=== Trajectory Debug Information ===")
    for obj_name, traj in trajectories.items():
        print(f"{obj_name}: {len(traj['frames'])} frames")
        if len(traj['positions']) > 0:
            pos = np.array(traj['positions'])
            print(f"  Position range: X[{pos[:,0].min():.2f}, {pos[:,0].max():.2f}], "
                  f"Y[{pos[:,1].min():.2f}, {pos[:,1].max():.2f}], "
                  f"Z[{pos[:,2].min():.2f}, {pos[:,2].max():.2f}]")


def add_court_to_figure(fig: go.Figure, court_corners: np.ndarray) -> None:
    """Add court boundary and surface to the figure."""
    # Convert court corners to basketball convention
    court_x = court_corners[:, 0]  # X stays X
    court_y = np.zeros_like(court_corners[:, 0])  # Y = 0 (ground level)
    court_z = court_corners[:, 1]  # Original Y becomes Z (depth)
    
    # Create court outline
    fig.add_trace(go.Scatter3d(
        x=court_x, y=court_y, z=court_z,
        mode='lines+markers',
        name='Court Boundary',
        line=dict(color='gray', width=2),
        marker=dict(size=6, color='black'),
        showlegend=True,
        visible=True  # Court always visible
    ))
    
    # Add court surface
    xx, yy, zz = create_court_grid(court_corners)
    
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='oranges',
        opacity=0.3,
        showscale=False,
        name='Court Surface',
        showlegend=False,
        visible=True  # Court always visible
    ))


def add_player_trajectories(fig: go.Figure, trajectories: Dict, 
                           metrics: Dict) -> List[str]:
    """Add player trajectory traces to the figure."""
    player_traces_added = []
    
    for i, (obj_name, traj) in enumerate(trajectories.items()):
        if len(traj['positions']) < 2:
            print(f"Skipping {obj_name}: only {len(traj['positions'])} position(s)")
            continue
        
        positions = np.array(traj['positions'])
        frames = traj['frames']
        
        # Create hover text
        hover_text = create_hover_text(obj_name, frames, positions, metrics)
        
        # Add main trajectory
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],  # X (width)
            y=positions[:, 1],  # Y (height)
            z=positions[:, 2],  # Z (depth)
            mode='lines+markers',
            name=f'{obj_name}',
            line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=4),
            marker=dict(size=3),
            text=hover_text,
            hoverinfo='text',
            legendgroup=f'player{obj_name}',
            visible=False  # Start with all players hidden
        ))
        
        # Add start marker
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            name=f'Start {obj_name}',
            marker=dict(size=10, color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], 
                       symbol='circle'),
            showlegend=False,
            legendgroup=f'player{obj_name}',
            visible=False
        ))
        
        # Add end marker
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode='markers',
            name=f'End {obj_name}',
            marker=dict(size=10, color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], 
                       symbol='square'),
            showlegend=False,
            legendgroup=f'player{obj_name}',
            visible=False
        ))
        
        player_traces_added.append(obj_name)
    
    return player_traces_added


def create_hover_text(obj_name: str, frames: List[int], 
                      positions: np.ndarray, metrics: Dict) -> List[str]:
    """Create hover text for trajectory points."""
    hover_text = []
    
    for j, frame in enumerate(frames):
        text = f"Player: {obj_name}<br>"
        text += f"Frame: {frame}<br>"
        text += f"Position: ({positions[j, 0]:.2f}, {positions[j, 1]:.2f}, "
        text += f"{positions[j, 2]:.2f}) m<br>"
        
        if obj_name in metrics:
            text += f"Avg Speed: {metrics[obj_name]['avg_ground_speed_ms']:.2f} m/s<br>"
            text += f"Total Distance: {metrics[obj_name]['total_ground_distance_m']:.2f} m"
        
        hover_text.append(text)
    
    return hover_text


def configure_3d_layout(fig: go.Figure, num_court_traces: int) -> None:
    """Configure the 3D plot layout and controls."""
    fig.update_layout(
        title={
            'text': 'Basketball Player 3D Trajectories<br>'
                    '<sub>Click on player names to show/hide trajectories</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis=dict(
                title=AXIS_LABELS['x'],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title=AXIS_LABELS['y'],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=[0, 6]  # Limit height range
            ),
            zaxis=dict(
                title=AXIS_LABELS['z'],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.5, z=1),  # Basketball court proportions
            camera=dict(
                eye=dict(x=0, y=1.5, z=-2.5),  # Side view
                center=dict(x=0, y=0, z=0)
            )
        ),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            x=1.02,
            y=0.5,
            yanchor='middle',
            title=dict(text='Players (click to show/hide)')
        ),
        width=1200,
        height=800
    )
    
    # Add show/hide all buttons
    add_visibility_controls(fig, num_court_traces)
    
def add_visibility_controls(fig: go.Figure, num_court_traces: int) -> None:
    """
    Adds 'Show All' and 'Hide All' buttons to the figure layout.

    Args:
        fig: The Plotly Figure object to update.
        num_court_traces: The number of initial traces that represent the
                          court, which should always remain visible.
    """
    total_traces = len(fig.data)
    num_player_traces = total_traces - num_court_traces

    # Create the 'Show All' button configuration
    show_all_button = dict(
        label='Show All Players',
        method='update',
        args=[{'visible': [True] * num_court_traces + [True] * num_player_traces},
              {'title.text': '<b>Basketball Player 3D Trajectories</b><br><sub>Showing all player trajectories</sub>'}]
    )

    # Create the 'Hide All' button configuration
    hide_all_button = dict(
        label='Hide All Players',
        method='update',
        args=[{'visible': [True] * num_court_traces + [False] * num_player_traces},
             {'title.text': '<b>Basketball Player 3D Trajectories</b><br><sub>Click on player names in the legend to toggle individual trajectories</sub>'}]
    )

    # Update the figure's layout with the new buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[show_all_button, hide_all_button],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )