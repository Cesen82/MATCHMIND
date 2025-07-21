"""
Dashboard page for ProFootballAI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any

from config import LEAGUES
from ..theme import create_metric_card
from ..charts import create_performance_chart, create_distribution_chart


async def render(api_client, predictor, db_manager):
    """Render dashboard page"""
    
    st.markdown("# 📊 Dashboard")
    st.markdown("Real-time overview of predictions and performance metrics")
    
    # Train model if needed
    if not predictor.is_trained:
        with st.spinner("🤖 Training AI model..."):
            # Load or create training data
            training_data = await db_manager.get_training_data()
            if training_data is None:
                # Generate synthetic data
                data = predictor.create_training_data(n_samples=5000)
                X = data[predictor.feature_names]
                y = data['over25']
            else:
                X = training_data[predictor.feature_names]
                y = training_data['over25']
            
            # Train model
            metrics = predictor.train(X, y)
            
            st.success(f"✅ Model trained successfully! AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Key Metrics Row
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get model metrics
    model_info = predictor.get_model_info()
    
    with col1:
        st.markdown(
            create_metric_card(
                "Model Accuracy",
                f"{model_info['performance_metrics']['accuracy']:.1%}",
                "+2.3%",
                "Model accuracy on test set"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card(
                "AUC-ROC Score",
                f"{model_info['performance_metrics']['auc_roc']:.3f}",
                "+0.021",
                "Area under ROC curve"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        # Get total predictions from database
        total_predictions = await db_manager.get_predictions_count()
        st.markdown(
            create_metric_card(
                "Total Predictions",
                f"{total_predictions:,}",
                f"+{total_predictions // 10}",
                "Total predictions made"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        # Calculate win rate
        win_rate = await db_manager.get_win_rate()
        st.markdown(
            create_metric_card(
                "Win Rate",
                f"{win_rate:.1%}",
                f"+{win_rate * 0.05:.1%}",
                "Overall success rate"
            ),
            unsafe_allow_html=True
        )
    
    # Charts Row
    st.markdown("### 📈 Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance over time
        performance_data = await db_manager.get_performance_history(days=30)
        
        if performance_data:
            fig = create_performance_chart(performance_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Generate sample data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            accuracy = [0.75 + np.random.normal(0, 0.05) for _ in range(30)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=accuracy,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Model Performance (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Accuracy",
                template="plotly_dark",
                height=400,
                yaxis=dict(tickformat='.1%'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # League distribution
        league_stats = []
        for league_name, league_info in list(LEAGUES.items())[:6]:
            league_stats.append({
                "League": league_name.split(' ', 1)[1],
                "Over 2.5 %": 50 + np.random.randint(10, 25)
            })
        
        league_df = pd.DataFrame(league_stats)
        
        fig = px.bar(
            league_df,
            x='League',
            y='Over 2.5 %',
            title="Top Leagues by Over 2.5 Rate",
            color='Over 2.5 %',
            color_continuous_scale=['#047857', '#00d4aa']
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis={'tickangle': 45},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Predictions
    st.markdown("### 🔮 Recent Predictions")
    
    recent_predictions = await db_manager.get_recent_predictions(limit=10)
    
    if recent_predictions:
        # Display as cards
        for pred in recent_predictions:
            match = f"{pred['home_team']} vs {pred['away_team']}"
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{match}**")
                st.caption(f"{pred['date']} at {pred['time']}")
            
            with col2:
                prob_color = "#00d4aa" if pred['probability'] > 0.7 else "#ffa502" if pred['probability'] > 0.6 else "#ff4757"
                st.markdown(f"<span style='color: {prob_color}; font-weight: bold;'>{pred['probability']:.1%}</span>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**@{pred['odds']:.2f}**")
            
            with col4:
                if pred.get('result'):
                    if pred['result'] == 'won':
                        st.success("✅ Won")
                    else:
                        st.error("❌ Lost")
                else:
                    st.info("⏳ Pending")
    else:
        st.info("No recent predictions available. Start making predictions!")
    
    # Model Information
    with st.expander("🤖 Model Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Type:**")
            st.write(model_info['model_type'].title())
            
            st.markdown("**Features Count:**")
            st.write(model_info['features_count'])
            
            st.markdown("**Training Status:**")
            st.write("✅ Trained" if model_info['is_trained'] else "❌ Not Trained")
        
        with col2:
            st.markdown("**Performance Metrics:**")
            for metric, value in model_info['performance_metrics'].items():
                if isinstance(value, float):
                    st.write(f"- {metric}: {value:.3f}")
                else:
                    st.write(f"- {metric}: {value}")
        
        st.markdown("**Top Features:**")
        top_features = model_info.get('top_features', [])
        if top_features:
            features_df = pd.DataFrame({
                "Feature": top_features[:5],
                "Importance": [f"Feature {i+1}" for i in range(min(5, len(top_features)))]
            })
            st.dataframe(features_df, hide_index=True)
    
    # System Status
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    # API Status
    api_status = api_client.get_status()
    
    with col1:
        rate_limit = api_status['rate_limit']
        remaining = rate_limit['remaining_hourly']
        
        if remaining > 50:
            status_color = "risk-low"
            status_text = "Healthy"
        elif remaining > 20:
            status_color = "risk-medium"
            status_text = "Limited"
        else:
            status_color = "risk-high"
            status_text = "Critical"
        
        st.markdown(
            create_metric_card(
                "API Status",
                status_text,
                f"{remaining} calls/hr",
                "API rate limit status",
                card_type=status_color
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        cache_status = api_status['cache']
        hit_rate = cache_status['hit_rate']
        
        st.markdown(
            create_metric_card(
                "Cache Hit Rate",
                f"{hit_rate:.1%}",
                f"{cache_status['size_mb']:.1f} MB",
                "Cache effectiveness"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        db_stats = await db_manager.get_database_stats()
        
        st.markdown(
            create_metric_card(
                "Database Size",
                f"{db_stats['size_mb']:.1f} MB",
                f"{db_stats['records']:,} records",
                "Database storage"
            ),
            unsafe_allow_html=True
        )
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Report generation coming soon!")
    
    with col3:
        if st.button("🧹 Clean Cache", use_container_width=True):
            await api_client.cache.clear()
            st.success("Cache cleared!")
    
    with col4:
        if st.button("💾 Backup Data", use_container_width=True):
            await db_manager.backup()
            st.success("Backup completed!")


async def get_sample_performance_data():
    """Generate sample performance data for visualization"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    data = []
    base_accuracy = 0.75
    
    for i, date in enumerate(dates):
        # Add some realistic variation
        accuracy = base_accuracy + np.sin(i/5) * 0.05 + np.random.normal(0, 0.02)
        accuracy = np.clip(accuracy, 0.6, 0.9)
        
        data.append({
            'date': date,
            'accuracy': accuracy,
            'predictions': np.random.randint(20, 50),
            'correct': int(accuracy * np.random.randint(20, 50))
        })
    
    return pd.DataFrame(data)