import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import DOWNSAMPLE_STEP

TEMPERATURE_COLUMNS = ["temperature", "tempc", "temp_c", "t"]
HUMIDITY_COLUMNS = ["humidity", "humidity_percent", "rh"]

def _find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_plot_columns(df: pd.DataFrame):
    temp_col = _find_column(df, TEMPERATURE_COLUMNS)
    hum_col = _find_column(df, HUMIDITY_COLUMNS)
    return temp_col, hum_col

def get_temperature_plot(temp_col: str, values: tuple, city: str, time_values: tuple = None):
    if temp_col is None or not values:
        return None
    if time_values:
        fig = px.line(x=time_values, y=values, title=f"Temperature Trend Over Time - {city.title()}")
        fig.update_xaxes(title_text="Date/Time")
    else:
        fig = px.line(y=values, title=f"Temperature Trend - {city.title()}")
    fig.update_yaxes(title_text="Temperature (°C)")
    fig.update_traces(line=dict(width=2), marker=dict(size=2))
    return fig

def get_humidity_plot(hum_col: str, values: tuple, city: str, time_values: tuple = None):
    if hum_col is None or not values:
        return None
    if time_values:
        fig = px.line(x=time_values, y=values, title=f"Humidity Trend Over Time - {city.title()}")
        fig.update_xaxes(title_text="Date/Time")
    else:
        fig = px.line(y=values, title=f"Humidity Trend - {city.title()}")
    fig.update_yaxes(title_text="Humidity (%)")
    fig.update_traces(line=dict(width=2), marker=dict(size=2))
    return fig

def get_scatter_plot(temp_col: str, hum_col: str, x_values: tuple, y_values: tuple, city: str, anomaly_values: tuple = None):
    if temp_col is None or hum_col is None or not x_values or not y_values:
        return None

    fig = px.scatter(
        x=x_values,
        y=y_values,
        opacity=0.8,
        title=f"Temperature vs Humidity Relationship - {city.title()}"
    )

    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Humidity (%)")
    fig.update_traces(marker=dict(size=7, color="#00b3e6"), selector=dict(mode="markers"))
    fig.update_layout(transition_duration=0)
    return fig

def get_anomaly_scatter_plot(temp_col: str, hum_col: str, x_values: tuple, y_values: tuple, anomaly_values: tuple, city: str):
    if temp_col is None or hum_col is None or not x_values or not y_values or not anomaly_values:
        return None

    # Split by label for correct hover and visual clarity
    normal_points = [(x, y) for x, y, a in zip(x_values, y_values, anomaly_values) if a == 'Normal']
    anomaly_points = [(x, y) for x, y, a in zip(x_values, y_values, anomaly_values) if a == 'Anomaly']

    fig = go.Figure()
    if normal_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in normal_points],
            y=[p[1] for p in normal_points],
            mode='markers',
            marker=dict(color='#999999', size=7, opacity=0.55),
            name='Normal',
            hovertemplate='Type: Normal<br>Temp: %{x}<br>Humidity: %{y}<extra></extra>'
        ))

    if anomaly_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in anomaly_points],
            y=[p[1] for p in anomaly_points],
            mode='markers',
            marker=dict(color='#ff4136', size=9, symbol='diamond', opacity=0.85),
            name='Anomaly',
            hovertemplate='Type: Anomaly<br>Temp: %{x}<br>Humidity: %{y}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Weather Anomalies Detection - {city.title()}",
        showlegend=False,
        transition_duration=0,
    )
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Humidity (%)")
    return fig

def get_anomaly_percentage_pie_chart(normal_counts: int, anomaly_counts: int, city: str):
    """Create a donut chart showing the percentage distribution of normal vs anomaly data points."""
    if normal_counts == 0 and anomaly_counts == 0:
        return None
    
    total = normal_counts + anomaly_counts
    normal_pct = (normal_counts / total * 100) if total > 0 else 0
    anomaly_pct = (anomaly_counts / total * 100) if total > 0 else 0
    
    labels = ['Normal', 'Anomaly']
    values = [normal_counts, anomaly_counts]
    colors = ['#2ecc71', '#ff4136']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{customdata:.2f}%<extra></extra>',
        customdata=[normal_pct, anomaly_pct],
        textposition='auto',
        textinfo='label+percent',
        hole=0.4
    )])
    
    fig.update_layout(
        title=f"Data Distribution - {city.title()}",
        showlegend=True,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def get_cluster_scatter_plot(temp_col: str, hum_col: str, x_values: tuple, y_values: tuple, cluster_values: tuple, city: str):
    """Create a scatter plot colored by cluster."""
    if temp_col is None or hum_col is None or not x_values or not y_values or not cluster_values:
        return None

    # Create color map for clusters
    unique_clusters = list(set(cluster_values))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}

    fig = go.Figure()

    for cluster in unique_clusters:
        cluster_points = [(x, y) for x, y, c in zip(x_values, y_values, cluster_values) if c == cluster]
        if cluster_points:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in cluster_points],
                y=[p[1] for p in cluster_points],
                mode='markers',
                marker=dict(color=color_map[cluster], size=7, opacity=0.7),
                name=f'Cluster {cluster}',
                hovertemplate=f'Cluster: {cluster}<br>Temp: %{{x}}<br>Humidity: %{{y}}<extra></extra>'
            ))

    fig.update_layout(
        title=f"Cluster Visualization - {city.title()}",
        showlegend=True,
        transition_duration=0,
    )
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Humidity (%)")
    return fig

def get_final_anomaly_plot(temp_col: str, hum_col: str, x_values: tuple, y_values: tuple, final_anomaly_values: tuple, city: str):
    """Create a scatter plot highlighting final hybrid anomalies."""
    if temp_col is None or hum_col is None or not x_values or not y_values or not final_anomaly_values:
        return None

    normal_points = [(x, y) for x, y, a in zip(x_values, y_values, final_anomaly_values) if a == 1]
    anomaly_points = [(x, y) for x, y, a in zip(x_values, y_values, final_anomaly_values) if a == -1]

    fig = go.Figure()
    if normal_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in normal_points],
            y=[p[1] for p in normal_points],
            mode='markers',
            marker=dict(color='#999999', size=7, opacity=0.55),
            name='Normal',
            hovertemplate='Type: Normal<br>Temp: %{x}<br>Humidity: %{y}<extra></extra>'
        ))

    if anomaly_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in anomaly_points],
            y=[p[1] for p in anomaly_points],
            mode='markers',
            marker=dict(color='#ff4136', size=9, symbol='diamond', opacity=0.85),
            name='Hybrid Anomaly',
            hovertemplate='Type: Hybrid Anomaly<br>Temp: %{x}<br>Humidity: %{y}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Hybrid Anomaly Detection - {city.title()}",
        showlegend=True,
        transition_duration=0,
    )
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Humidity (%)")
    return fig
