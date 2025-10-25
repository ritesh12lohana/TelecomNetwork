import solara
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from typing import List, Dict
import json

# Data Generation Functions
def generate_ran_data(num_records: int = 1000, days: int = 30) -> pd.DataFrame:
    """Generate simulated RAN (Radio Access Network) data"""
    np.random.seed(42)
    
    # Time series
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_records)]
    
    # Cell IDs and locations
    cell_ids = np.random.choice([f'CELL_{i:03d}' for i in range(1, 51)], num_records)
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_records)
    
    # Network KPIs with realistic patterns
    base_latency = 20  # ms
    latency = base_latency + np.random.normal(0, 5, num_records) + \
              np.sin(np.linspace(0, 4*np.pi, num_records)) * 3
    latency = np.clip(latency, 5, 100)
    
    base_throughput = 50  # Mbps
    throughput = base_throughput + np.random.normal(0, 10, num_records) + \
                 np.cos(np.linspace(0, 4*np.pi, num_records)) * 15
    throughput = np.clip(throughput, 10, 100)
    
    base_signal = -80  # dBm
    signal_strength = base_signal + np.random.normal(0, 5, num_records) + \
                     np.sin(np.linspace(0, 3*np.pi, num_records)) * 8
    signal_strength = np.clip(signal_strength, -110, -50)
    
    # Additional metrics
    packet_loss = np.random.exponential(0.5, num_records)
    packet_loss = np.clip(packet_loss, 0, 5)
    
    active_users = np.random.poisson(50, num_records)
    active_users = np.clip(active_users, 10, 150)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cell_id': cell_ids,
        'region': regions,
        'latency_ms': latency,
        'throughput_mbps': throughput,
        'signal_strength_dbm': signal_strength,
        'packet_loss_pct': packet_loss,
        'active_users': active_users
    })
    
    return df

def predict_trends(df: pd.DataFrame, metric: str, forecast_hours: int = 24) -> tuple:
    """Predict future trends using linear regression"""
    df_sorted = df.sort_values('timestamp')
    
    # Prepare data for prediction
    X = np.arange(len(df_sorted)).reshape(-1, 1)
    y = df_sorted[metric].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    last_idx = len(df_sorted)
    future_X = np.arange(last_idx, last_idx + forecast_hours).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Generate future timestamps
    last_time = df_sorted['timestamp'].iloc[-1]
    future_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
    
    return future_times, predictions, model.score(X, y)

def calculate_kpis(df: pd.DataFrame) -> Dict:
    """Calculate overall network KPIs"""
    return {
        'avg_latency': df['latency_ms'].mean(),
        'avg_throughput': df['throughput_mbps'].mean(),
        'avg_signal': df['signal_strength_dbm'].mean(),
        'avg_packet_loss': df['packet_loss_pct'].mean(),
        'total_cells': df['cell_id'].nunique(),
        'total_records': len(df)
    }

# Initialize data
initial_data = generate_ran_data(1000, 30)

# Reactive state
df_data = solara.reactive(initial_data)
selected_metric = solara.reactive("latency_ms")
selected_region = solara.reactive("All")
forecast_enabled = solara.reactive(True)

@solara.component
def MetricCard(title: str, value: float, unit: str, color: str = "#1f77b4"):
    """Display a KPI metric card"""
    with solara.Card(style=f"background: linear-gradient(135deg, {color}22 0%, {color}44 100%); padding: 20px; margin: 10px;"):
        solara.HTML(tag="div", unsafe_innerHTML=f"""
            <div style="text-align: center;">
                <div style="font-size: 14px; color: #666; margin-bottom: 8px;">{title}</div>
                <div style="font-size: 32px; font-weight: bold; color: {color};">{value:.2f}</div>
                <div style="font-size: 12px; color: #888;">{unit}</div>
            </div>
        """)

@solara.component
def TimeSeriesChart(df: pd.DataFrame, metric: str, forecast: bool = True):
    """Create interactive time series chart with predictions"""
    df_sorted = df.sort_values('timestamp')
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted[metric],
        mode='lines',
        name='Historical Data',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add predictions if enabled
    if forecast:
        future_times, predictions, r2_score = predict_trends(df_sorted, metric, 24)
        
        fig.add_trace(go.Scatter(
            x=future_times,
            y=predictions,
            mode='lines',
            name=f'Forecast (R²={r2_score:.3f})',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    metric_names = {
        'latency_ms': 'Latency (ms)',
        'throughput_mbps': 'Throughput (Mbps)',
        'signal_strength_dbm': 'Signal Strength (dBm)',
        'packet_loss_pct': 'Packet Loss (%)',
        'active_users': 'Active Users'
    }
    
    fig.update_layout(
        title=f'{metric_names.get(metric, metric)} Over Time',
        xaxis_title='Time',
        yaxis_title=metric_names.get(metric, metric),
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    solara.FigurePlotly(fig)

@solara.component
def RegionalDistribution(df: pd.DataFrame, metric: str):
    """Create regional comparison chart"""
    regional_stats = df.groupby('region')[metric].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=regional_stats['region'],
        y=regional_stats['mean'],
        error_y=dict(type='data', array=regional_stats['std']),
        marker_color='#2ca02c',
        name='Average'
    ))
    
    metric_names = {
        'latency_ms': 'Latency (ms)',
        'throughput_mbps': 'Throughput (Mbps)',
        'signal_strength_dbm': 'Signal Strength (dBm)',
        'packet_loss_pct': 'Packet Loss (%)'
    }
    
    fig.update_layout(
        title=f'Regional Comparison - {metric_names.get(metric, metric)}',
        xaxis_title='Region',
        yaxis_title=metric_names.get(metric, metric),
        height=400,
        template='plotly_white'
    )
    
    solara.FigurePlotly(fig)

@solara.component
def CorrelationHeatmap(df: pd.DataFrame):
    """Create correlation heatmap for metrics"""
    metrics = ['latency_ms', 'throughput_mbps', 'signal_strength_dbm', 
               'packet_loss_pct', 'active_users']
    
    corr_matrix = df[metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Latency', 'Throughput', 'Signal', 'Packet Loss', 'Users'],
        y=['Latency', 'Throughput', 'Signal', 'Packet Loss', 'Users'],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Metric Correlation Matrix',
        height=400,
        template='plotly_white'
    )
    
    solara.FigurePlotly(fig)

@solara.component
def Page():
    """Main dashboard page"""
    
    # Filter data based on region
    if selected_region.value != "All":
        filtered_df = df_data.value[df_data.value['region'] == selected_region.value]
    else:
        filtered_df = df_data.value
    
    kpis = calculate_kpis(filtered_df)
    
    with solara.Column(style={"padding": "20px", "max-width": "1400px", "margin": "0 auto"}):
        # Header
        solara.HTML(tag="div", unsafe_innerHTML="""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #1f77b4; margin-bottom: 10px;">
                    📡 Network Data Insights Dashboard
                </h1>
                <p style="color: #666; font-size: 16px;">
                    Real-time RAN Performance Analysis & Prediction
                </p>
            </div>
        """)
        
        # Controls
        with solara.Card(style="padding: 20px; margin-bottom: 20px;"):
            with solara.Row(justify="space-around"):
                with solara.Column(style="flex: 1; margin: 0 10px;"):
                    solara.Select(
                        label="Select Metric",
                        value=selected_metric,
                        values=[
                            "latency_ms",
                            "throughput_mbps",
                            "signal_strength_dbm",
                            "packet_loss_pct",
                            "active_users"
                        ],
                        dense=True
                    )
                
                with solara.Column(style="flex: 1; margin: 0 10px;"):
                    solara.Select(
                        label="Select Region",
                        value=selected_region,
                        values=["All", "North", "South", "East", "West", "Central"],
                        dense=True
                    )
                
                with solara.Column(style="flex: 1; margin: 0 10px;"):
                    solara.Checkbox(
                        label="Enable Forecasting",
                        value=forecast_enabled
                    )
                
                with solara.Column(style="flex: 1; margin: 0 10px;"):
                    if solara.Button("🔄 Regenerate Data", color="primary"):
                        df_data.value = generate_ran_data(1000, 30)
        
        # KPI Cards
        with solara.Row():
            with solara.Column(style="flex: 1;"):
                MetricCard("Avg Latency", kpis['avg_latency'], "ms", "#e74c3c")
            with solara.Column(style="flex: 1;"):
                MetricCard("Avg Throughput", kpis['avg_throughput'], "Mbps", "#2ecc71")
            with solara.Column(style="flex: 1;"):
                MetricCard("Avg Signal", kpis['avg_signal'], "dBm", "#3498db")
            with solara.Column(style="flex: 1;"):
                MetricCard("Packet Loss", kpis['avg_packet_loss'], "%", "#f39c12")
            with solara.Column(style="flex: 1;"):
                MetricCard("Total Cells", kpis['total_cells'], "cells", "#9b59b6")
        
        # Main Charts
        with solara.Row():
            with solara.Column(style="flex: 2;"):
                with solara.Card():
                    TimeSeriesChart(filtered_df, selected_metric.value, forecast_enabled.value)
            
            with solara.Column(style="flex: 1;"):
                with solara.Card():
                    RegionalDistribution(filtered_df, selected_metric.value)
        
        # Correlation Matrix
        with solara.Card():
            CorrelationHeatmap(filtered_df)
        
        # Data Table Preview
        with solara.Card(style="margin-top: 20px;"):
            solara.Markdown("### 📊 Data Preview (Last 10 Records)")
            solara.DataFrame(
                filtered_df.sort_values('timestamp', ascending=False).head(10),
                items_per_page=10
            )
        
        # Footer
        solara.HTML(tag="div", unsafe_innerHTML="""
            <div style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
                <p>Built with Python, Pandas, Plotly & Solara | Network Performance Analytics</p>
                <p style="font-size: 12px;">Data updates in real-time • ML-powered forecasting • Export ready</p>
            </div>
        """)

# Run the app
Page()
