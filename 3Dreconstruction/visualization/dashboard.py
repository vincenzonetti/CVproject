"""
Metrics dashboard visualization for basketball tracking.

This module creates comprehensive dashboards showing player
movement statistics and comparisons.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_metrics_dashboard(metrics: Dict, trajectories: Dict, save_path: str) -> None:
    """
    Create a comprehensive metrics dashboard.
    
    Args:
        metrics: Dictionary of calculated metrics for each player
        trajectories: Dictionary of player trajectories
        save_path: Path to save the HTML dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Speed Distribution', 'Distance Traveled', 
                       'Coverage Area', 'Movement Intensity'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Extract data for plotting
    plot_data = extract_plot_data(metrics)
    
    # Add speed distribution plot
    add_speed_plot(fig, plot_data, row=1, col=1)
    
    # Add distance plot
    add_distance_plot(fig, plot_data, row=1, col=2)
    
    # Add coverage area plot
    add_coverage_plot(fig, plot_data, row=2, col=1)
    
    # Add movement intensity plot
    add_movement_intensity_plot(fig, plot_data, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Player Movement Metrics Dashboard",
        height=800,
        showlegend=True
    )
    
    # Save dashboard
    fig.write_html(save_path)
    print(f"Metrics dashboard saved to {save_path}")
    
    # Save detailed metrics to CSV
    save_metrics_csv(metrics, save_path.replace('.html', '.csv'))


def extract_plot_data(metrics: Dict) -> Dict:
    """Extract and organize data for plotting."""
    players = list(metrics.keys())
    
    return {
        'players': players,
        'avg_speeds': [metrics[p]['avg_ground_speed_ms'] for p in players],
        'max_speeds': [metrics[p]['max_ground_speed_ms'] for p in players],
        'distances': [metrics[p]['total_ground_distance_m'] for p in players],
        'areas': [metrics[p]['coverage_area_m2'] for p in players],
        'accelerations': [metrics[p]['avg_acceleration_ms2'] for p in players],
        'direction_changes': [metrics[p]['direction_changes'] for p in players],
        'max_heights': [metrics[p]['max_height_m'] for p in players],
        'time_tracked': [metrics[p]['time_tracked_s'] for p in players]
    }


def add_speed_plot(fig: go.Figure, data: Dict, row: int, col: int) -> None:
    """Add speed distribution bar plot."""
    fig.add_trace(
        go.Bar(name='Avg Speed', x=data['players'], y=data['avg_speeds'], 
               marker_color='lightblue'),
        row=row, col=col
    )
    fig.add_trace(
        go.Bar(name='Max Speed', x=data['players'], y=data['max_speeds'], 
               marker_color='darkblue'),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Player", row=row, col=col)
    fig.update_yaxes(title_text="Speed (m/s)", row=row, col=col)


def add_distance_plot(fig: go.Figure, data: Dict, row: int, col: int) -> None:
    """Add distance traveled bar plot."""
    fig.add_trace(
        go.Bar(x=data['players'], y=data['distances'], 
               marker_color='green', showlegend=False),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Player", row=row, col=col)
    fig.update_yaxes(title_text="Distance (m)", row=row, col=col)


def add_coverage_plot(fig: go.Figure, data: Dict, row: int, col: int) -> None:
    """Add coverage area bar plot."""
    fig.add_trace(
        go.Bar(x=data['players'], y=data['areas'], 
               marker_color='orange', showlegend=False),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Player", row=row, col=col)
    fig.update_yaxes(title_text="Area (m²)", row=row, col=col)


def add_movement_intensity_plot(fig: go.Figure, data: Dict, 
                               row: int, col: int) -> None:
    """Add movement intensity scatter plot."""
    # Create hover text with player names and additional info
    hover_text = []
    for i, player in enumerate(data['players']):
        text = f"{player}<br>"
        text += f"Acceleration: {data['accelerations'][i]:.2f} m/s²<br>"
        text += f"Direction Changes: {data['direction_changes'][i]}<br>"
        text += f"Max Height: {data['max_heights'][i]:.2f} m"
        hover_text.append(text)
    
    fig.add_trace(
        go.Scatter(
            x=data['accelerations'], 
            y=data['direction_changes'],
            mode='markers+text',
            text=data['players'],
            textposition="top center",
            marker=dict(
                size=10, 
                color=data['time_tracked'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time<br>Tracked (s)")
            ),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Avg Acceleration (m/s²)", row=row, col=col)
    fig.update_yaxes(title_text="Direction Changes", row=row, col=col)


def save_metrics_csv(metrics: Dict, csv_path: str) -> None:
    """Save detailed metrics to CSV file."""
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'Player'
    metrics_df.to_csv(csv_path)
    print(f"Detailed metrics saved to {csv_path}")


def create_summary_statistics(metrics: Dict) -> pd.DataFrame:
    """Create summary statistics across all players."""
    df = pd.DataFrame(metrics).T
    
    summary = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'Min': df.min(),
        'Max': df.max()
    })
    
    return summary.T