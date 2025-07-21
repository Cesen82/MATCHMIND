"""
Advanced charting module for ProFootballAI
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from config import UI_CONFIG


# Color schemes
COLORS = {
    'primary': UI_CONFIG['primary_color'],
    'secondary': UI_CONFIG['secondary_color'],
    'success': UI_CONFIG['success_color'],
    'warning': UI_CONFIG['warning_color'],
    'danger': UI_CONFIG['danger_color'],
    'background': '#0d1117',
    'surface': '#21262d',
    'text': '#f0f6fc',
    'text_secondary': '#8b949e'
}

# Chart templates
DARK_TEMPLATE = {
    'layout': {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['surface'],
        'font': {
            'color': COLORS['text'],
            'family': 'Inter, sans-serif'
        },
        'title': {
            'font': {
                'size': 20,
                'color': COLORS['text']
            }
        },
        'xaxis': {
            'gridcolor': '#30363d',
            'linecolor': '#30363d',
            'tickfont': {'color': COLORS['text_secondary']}
        },
        'yaxis': {
            'gridcolor': '#30363d',
            'linecolor': '#30363d',
            'tickfont': {'color': COLORS['text_secondary']}
        },
        'hoverlabel': {
            'bgcolor': COLORS['surface'],
            'bordercolor': COLORS['primary'],
            'font': {'color': COLORS['text']}
        }
    }
}


def create_performance_chart(data: pd.DataFrame) -> go.Figure:
    """Create model performance chart over time"""
    
    fig = go.Figure()
    
    # Accuracy line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(
            color=COLORS['primary'],
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=8,
            color=COLORS['primary'],
            line=dict(color=COLORS['background'], width=2)
        ),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                      '<b>Accuracy:</b> %{y:.1%}<br>' +
                      '<extra></extra>'
    ))
    
    # Add moving average
    if len(data) > 7:
        ma7 = data['accuracy'].rolling(window=7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=ma7,
            mode='lines',
            name='7-day MA',
            line=dict(
                color=COLORS['secondary'],
                width=2,
                dash='dash'
            ),
            hovertemplate='<b>7-day MA:</b> %{y:.1%}<extra></extra>'
        ))
    
    # Add threshold line
    fig.add_hline(
        y=0.75,
        line_dash="dot",
        line_color=COLORS['warning'],
        annotation_text="Target: 75%"
    )
    
    # Update layout
    fig.update_layout(
        title="Model Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis_tickformat='.0%',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(33, 38, 45, 0.8)',
            bordercolor=COLORS['primary'],
            borderwidth=1
        ),
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_distribution_chart(data: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Create distribution histogram"""
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data[column],
        nbinsx=20,
        name='Distribution',
        marker=dict(
            color=COLORS['primary'],
            line=dict(
                color=COLORS['background'],
                width=1
            )
        ),
        opacity=0.8,
        hovertemplate='<b>Range:</b> %{x}<br>' +
                      '<b>Count:</b> %{y}<br>' +
                      '<extra></extra>'
    ))
    
    # Add normal distribution overlay
    mean = data[column].mean()
    std = data[column].std()
    x_range = np.linspace(data[column].min(), data[column].max(), 100)
    normal_dist = ((1 / (std * np.sqrt(2 * np.pi))) * 
                   np.exp(-0.5 * ((x_range - mean) / std) ** 2))
    
    # Scale to match histogram
    hist, bins = np.histogram(data[column], bins=20)
    bin_width = bins[1] - bins[0]
    normal_dist = normal_dist * len(data) * bin_width
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist,
        mode='lines',
        name='Normal Dist',
        line=dict(
            color=COLORS['secondary'],
            width=3,
            dash='dash'
        ),
        hovertemplate='<b>Normal Distribution</b><extra></extra>'
    ))
    
    # Add vertical lines for mean and median
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color=COLORS['success'],
        annotation_text=f"Mean: {mean:.2f}"
    )
    
    median = data[column].median()
    if abs(mean - median) > 0.01:
        fig.add_vline(
            x=median,
            line_dash="dot",
            line_color=COLORS['warning'],
            annotation_text=f"Median: {median:.2f}"
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Frequency",
        bargap=0.1,
        showlegend=True,
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_roi_chart(betting_history: pd.DataFrame) -> go.Figure:
    """Create ROI progression chart"""
    
    # Calculate cumulative ROI
    betting_history['cumulative_roi'] = (
        betting_history['returns'].cumsum() / 
        betting_history['stakes'].cumsum() - 1
    ) * 100
    
    fig = go.Figure()
    
    # ROI line
    fig.add_trace(go.Scatter(
        x=betting_history['date'],
        y=betting_history['cumulative_roi'],
        mode='lines+markers',
        name='Cumulative ROI',
        line=dict(
            color=COLORS['primary'],
            width=3
        ),
        fill='tozeroy',
        fillcolor=f"rgba(0, 212, 170, 0.1)",
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                      '<b>ROI:</b> %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_color=COLORS['text_secondary'],
        line_width=1
    )
    
    # Color negative areas
    fig.add_hrect(
        y0=betting_history['cumulative_roi'].min(),
        y1=0,
        fillcolor=COLORS['danger'],
        opacity=0.1,
        line_width=0
    )
    
    # Update layout
    fig.update_layout(
        title="Return on Investment (ROI) Progression",
        xaxis_title="Date",
        yaxis_title="Cumulative ROI (%)",
        yaxis_ticksuffix="%",
        hovermode='x unified',
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_league_comparison_chart(league_data: pd.DataFrame) -> go.Figure:
    """Create league comparison radar chart"""
    
    categories = ['Over 2.5 Rate', 'Avg Goals', 'Predictability', 'ROI', 'Volume']
    
    fig = go.Figure()
    
    for _, league in league_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                league['over25_rate'],
                league['avg_goals'] / 4 * 100,  # Normalize to 0-100
                league['predictability'] * 100,
                league['roi'] + 50,  # Shift to positive
                league['volume'] / league_data['volume'].max() * 100
            ],
            theta=categories,
            fill='toself',
            name=league['name'],
            opacity=0.6,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          '%{theta}: %{r:.1f}<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#30363d',
                linecolor='#30363d'
            ),
            angularaxis=dict(
                gridcolor='#30363d',
                linecolor='#30363d'
            ),
            bgcolor=COLORS['surface']
        ),
        title="League Comparison",
        showlegend=True,
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_prediction_confidence_chart(predictions: pd.DataFrame) -> go.Figure:
    """Create prediction confidence vs accuracy chart"""
    
    # Group by confidence level
    confidence_groups = predictions.groupby('confidence').agg({
        'is_correct': ['count', 'mean']
    }).reset_index()
    
    confidence_groups.columns = ['confidence', 'count', 'accuracy']
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    
    # Bar chart for count
    fig.add_trace(
        go.Bar(
            x=confidence_groups['confidence'],
            y=confidence_groups['count'],
            name='Predictions Count',
            marker_color=COLORS['primary'],
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>' +
                          'Count: %{y}<br>' +
                          '<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Line chart for accuracy
    fig.add_trace(
        go.Scatter(
            x=confidence_groups['confidence'],
            y=confidence_groups['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(
                color=COLORS['secondary'],
                width=3
            ),
            marker=dict(size=10),
            hovertemplate='<b>%{x}</b><br>' +
                          'Accuracy: %{y:.1%}<br>' +
                          '<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Confidence Level")
    fig.update_yaxes(title_text="Number of Predictions", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", tickformat='.0%', secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title="Prediction Confidence vs Accuracy",
        hovermode='x unified',
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_heatmap_chart(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str) -> go.Figure:
    """Create heatmap visualization"""
    
    # Pivot data for heatmap
    heatmap_data = data.pivot_table(
        index=y_col,
        columns=x_col,
        values=z_col,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[
            [0, COLORS['danger']],
            [0.5, COLORS['warning']],
            [1, COLORS['success']]
        ],
        hoverongaps=False,
        hovertemplate='<b>%{x}</b><br>' +
                      '<b>%{y}</b><br>' +
                      'Value: %{z:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_feature_importance_chart(feature_importance: Dict[str, float]) -> go.Figure:
    """Create feature importance horizontal bar chart"""
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    # Create gradient colors
    colors = [f"rgba(0, 212, 170, {0.4 + 0.6 * (i / len(features))})" for i in range(len(features))]
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(
                color=COLORS['primary'],
                width=1
            )
        ),
        hovertemplate='<b>%{y}</b><br>' +
                      'Importance: %{x:.3f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="Top 15 Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        **DARK_TEMPLATE['layout']
    )
    
    return fig


def create_betting_portfolio_chart(portfolio: List[Dict[str, Any]]) -> go.Figure:
    """Create betting portfolio allocation chart"""
    
    # Prepare data
    df = pd.DataFrame(portfolio)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Stake Allocation', 'Risk Distribution', 
                       'Expected Returns', 'Bet Type Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Stake allocation pie
    fig.add_trace(
        go.Pie(
            labels=df['match'],
            values=df['stake'],
            hole=0.4,
            marker_colors=px.colors.sequential.Tealgrn
        ),
        row=1, col=1
    )
    
    # Risk distribution pie
    risk_counts = df['risk_level'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=[COLORS['success'], COLORS['warning'], COLORS['danger']]
        ),
        row=1, col=2
    )
    
    # Expected returns bar
    fig.add_trace(
        go.Bar(
            x=df['match'],
            y=df['expected_return'],
            marker_color=COLORS['primary']
        ),
        row=2, col=1
    )
    
    # Bet type distribution
    bet_type_counts = df['bet_type'].value_counts()
    fig.add_trace(
        go.Bar(
            x=bet_type_counts.index,
            y=bet_type_counts.values,
            marker_color=COLORS['secondary']
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Betting Portfolio Analysis",
        showlegend=False,
        height=800,
        **DARK_TEMPLATE['layout']
    )
    
    return fig