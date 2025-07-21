"""
Analytics page for ProFootballAI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import LEAGUES, get_league_season
from ..theme import create_metric_card
from ..charts import (
    create_league_comparison_chart,
    create_heatmap_chart,
    create_distribution_chart,
    create_roi_chart
)
from ...utils.exceptions import handle_exception

logger = logging.getLogger("profootball.ui.analytics")


async def render(api_client, db_manager):
    """Render analytics page"""
    
    st.markdown("# 📈 Advanced Analytics")
    st.markdown("In-depth analysis of leagues, teams, and betting performance")
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=[
            "League Overview",
            "Team Performance",
            "Over 2.5 Trends",
            "Betting Performance",
            "Model Analytics"
        ],
        index=0
    )
    
    if analysis_type == "League Overview":
        await render_league_overview(api_client, db_manager)
    elif analysis_type == "Team Performance":
        await render_team_performance(api_client, db_manager)
    elif analysis_type == "Over 2.5 Trends":
        await render_over25_trends(api_client, db_manager)
    elif analysis_type == "Betting Performance":
        await render_betting_performance(db_manager)
    elif analysis_type == "Model Analytics":
        await render_model_analytics(db_manager)


async def render_league_overview(api_client, db_manager):
    """Render league overview analytics"""
    
    st.markdown("## 🌍 League Overview")
    
    # League selector
    selected_leagues = st.multiselect(
        "Select Leagues to Compare",
        options=list(LEAGUES.keys()),
        default=list(LEAGUES.keys())[:5],
        help="Choose leagues for comparison"
    )
    
    if not selected_leagues:
        st.warning("Please select at least one league")
        return
    
    # Time period
    col1, col2 = st.columns([2, 1])
    
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=90), datetime.now()),
            max_value=datetime.now()
        )
        
    with col2:
        refresh = st.button("🔄 Refresh Data", use_container_width=True)
    
    # Fetch league statistics
    league_stats = []
    
    with st.spinner("Loading league statistics..."):
        for league_name in selected_leagues:
            try:
                league_info = LEAGUES[league_name]
                season = get_league_season(league_name)
                
                # Get team statistics
                team_stats = await api_client.get_team_stats(league_info['id'], season)
                
                if team_stats:
                    # Calculate league metrics
                    total_matches = sum(s.matches_played for s in team_stats)
                    total_over25 = sum(s.over25_count for s in team_stats)
                    avg_goals = np.mean([s.goals_per_match for s in team_stats])
                    
                    # Get from database for advanced metrics
                    db_stats = await db_manager.get_team_stats(league_info['id'], season)
                    
                    # Calculate predictability (std deviation of over 2.5 rates)
                    if not db_stats.empty:
                        predictability = 1 - db_stats['over25_rate'].std()
                    else:
                        predictability = 0.7
                    
                    league_stats.append({
                        'name': league_name.split(' ', 1)[1],
                        'country': league_info['country'],
                        'over25_rate': (total_over25 / max(total_matches, 1)) * 100,
                        'avg_goals': avg_goals,
                        'predictability': predictability,
                        'roi': np.random.uniform(-5, 15),  # Would calculate from betting history
                        'volume': total_matches,
                        'teams': len(team_stats)
                    })
                    
            except Exception as e:
                logger.error(f"Error loading {league_name}: {e}")
                continue
    
    if not league_stats:
        st.error("Failed to load league statistics")
        return
    
    league_df = pd.DataFrame(league_stats)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_over25 = league_df['over25_rate'].mean()
        st.metric("Avg Over 2.5 Rate", f"{avg_over25:.1f}%")
        
    with col2:
        avg_goals = league_df['avg_goals'].mean()
        st.metric("Avg Goals/Match", f"{avg_goals:.2f}")
        
    with col3:
        best_roi = league_df.nlargest(1, 'roi')['name'].iloc[0]
        st.metric("Best ROI", best_roi)
        
    with col4:
        most_predictable = league_df.nlargest(1, 'predictability')['name'].iloc[0]
        st.metric("Most Predictable", most_predictable)
    
    # League comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of Over 2.5 rates
        fig = px.bar(
            league_df.sort_values('over25_rate', ascending=True),
            x='over25_rate',
            y='name',
            orientation='h',
            title="Over 2.5 Goals Rate by League",
            color='over25_rate',
            color_continuous_scale=['#ff4757', '#ffa502', '#00d4aa']
        )
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Over 2.5 Rate (%)",
            yaxis_title="League",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Radar chart comparison
        fig = create_league_comparison_chart(league_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed league table
    st.markdown("### 📊 Detailed Statistics")
    
    # Format dataframe for display
    display_df = league_df.copy()
    display_df['over25_rate'] = display_df['over25_rate'].round(1).astype(str) + '%'
    display_df['avg_goals'] = display_df['avg_goals'].round(2)
    display_df['predictability'] = (display_df['predictability'] * 100).round(1).astype(str) + '%'
    display_df['roi'] = display_df['roi'].round(1).astype(str) + '%'
    
    st.dataframe(
        display_df[['name', 'country', 'teams', 'volume', 'over25_rate', 'avg_goals', 'predictability', 'roi']],
        hide_index=True,
        use_container_width=True
    )


async def render_team_performance(api_client, db_manager):
    """Render team performance analytics"""
    
    st.markdown("## 👥 Team Performance Analysis")
    
    # League and team selector
    col1, col2 = st.columns(2)
    
    with col1:
        selected_league = st.selectbox(
            "Select League",
            options=list(LEAGUES.keys()),
            index=0
        )
        
    with col2:
        league_info = LEAGUES[selected_league]
        season = get_league_season(selected_league)
        
        # Get teams
        with st.spinner("Loading teams..."):
            team_stats = await api_client.get_team_stats(league_info['id'], season)
            
        if team_stats:
            team_names = [s.team_name for s in team_stats]
            selected_teams = st.multiselect(
                "Select Teams",
                options=team_names,
                default=team_names[:6],
                help="Choose teams to analyze"
            )
        else:
            st.error("No teams found")
            return
    
    if not selected_teams:
        st.warning("Please select at least one team")
        return
    
    # Filter team stats
    selected_stats = [s for s in team_stats if s.team_name in selected_teams]
    
    # Team comparison metrics
    st.markdown("### 📊 Team Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    
    for stat in selected_stats:
        comparison_data.append({
            'Team': stat.team_name,
            'Matches': stat.matches_played,
            'Goals For': stat.goals_for,
            'Goals Against': stat.goals_against,
            'Over 2.5 Rate': stat.over25_rate,
            'Goals/Match': stat.goals_per_match,
            'Points': stat.points,
            'Form': np.random.choice(['WWWDL', 'WDWLW', 'LLWDW', 'WWWWW', 'DDDLL'])
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Goals comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Goals For',
            x=comp_df['Team'],
            y=comp_df['Goals For'],
            marker_color='#00d4aa'
        ))
        
        fig.add_trace(go.Bar(
            name='Goals Against',
            x=comp_df['Team'],
            y=comp_df['Goals Against'],
            marker_color='#ff4757'
        ))
        
        fig.update_layout(
            title="Goals Comparison",
            barmode='group',
            template="plotly_dark",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Over 2.5 rate comparison
        fig = px.scatter(
            comp_df,
            x='Goals/Match',
            y='Over 2.5 Rate',
            size='Matches',
            color='Points',
            hover_data=['Team'],
            title="Goals vs Over 2.5 Rate",
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            template="plotly_dark",
            yaxis_tickformat='.0%'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Team details table
    st.markdown("### 📋 Detailed Team Statistics")
    
    # Format for display
    display_df = comp_df.copy()
    display_df['Over 2.5 Rate'] = (display_df['Over 2.5 Rate'] * 100).round(1).astype(str) + '%'
    display_df['Goals/Match'] = display_df['Goals/Match'].round(2)
    
    # Add styling based on performance
    def highlight_performance(row):
        styles = []
        for col in row.index:
            if col == 'Over 2.5 Rate':
                value = float(row[col].rstrip('%'))
                if value > 70:
                    styles.append('background-color: rgba(0, 212, 170, 0.3)')
                elif value > 60:
                    styles.append('background-color: rgba(255, 165, 2, 0.3)')
                else:
                    styles.append('background-color: rgba(255, 71, 87, 0.3)')
            else:
                styles.append('')
        return styles
    
    styled_df = display_df.style.apply(highlight_performance, axis=1)
    st.dataframe(styled_df, hide_index=True, use_container_width=True)


async def render_over25_trends(api_client, db_manager):
    """Render Over 2.5 trends analysis"""
    
    st.markdown("## 📊 Over 2.5 Goals Trends")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_league = st.selectbox(
            "Select League",
            options=list(LEAGUES.keys()),
            index=0
        )
        
    with col2:
        time_period = st.selectbox(
            "Time Period",
            options=["Last 30 days", "Last 90 days", "This Season", "Last Season"],
            index=0
        )
        
    with col3:
        granularity = st.selectbox(
            "Granularity",
            options=["Daily", "Weekly", "Monthly"],
            index=1
        )
    
    # Generate trend data (would be from database in production)
    trend_data = generate_trend_data(time_period, granularity)
    
    # Main trend chart
    fig = go.Figure()
    
    # Over 2.5 percentage line
    fig.add_trace(go.Scatter(
        x=trend_data['date'],
        y=trend_data['over25_rate'],
        mode='lines+markers',
        name='Over 2.5 Rate',
        line=dict(color='#00d4aa', width=3),
        yaxis='y'
    ))
    
    # Average goals line
    fig.add_trace(go.Scatter(
        x=trend_data['date'],
        y=trend_data['avg_goals'],
        mode='lines',
        name='Avg Goals',
        line=dict(color='#ffa502', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    # Add trend line
    z = np.polyfit(range(len(trend_data)), trend_data['over25_rate'], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=trend_data['date'],
        y=p(range(len(trend_data))),
        mode='lines',
        name='Trend',
        line=dict(color='#ff4757', width=2, dash='dot')
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title="Over 2.5 Goals Trend Analysis",
        xaxis_title="Date",
        yaxis=dict(
            title="Over 2.5 Rate (%)",
            tickformat='.1f',
            side='left'
        ),
        yaxis2=dict(
            title="Average Goals",
            overlaying='y',
            side='right'
        ),
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week analysis
        st.markdown("### 📅 By Day of Week")
        
        dow_data = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Over 2.5 %': [58, 61, 63, 62, 65, 68, 67]
        })
        
        fig = px.bar(
            dow_data,
            x='Day',
            y='Over 2.5 %',
            title="Over 2.5 Rate by Day of Week",
            color='Over 2.5 %',
            color_continuous_scale=['#ff4757', '#00d4aa']
        )
        
        fig.update_layout(
            template="plotly_dark",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Time of day analysis
        st.markdown("### 🕐 By Kick-off Time")
        
        time_data = pd.DataFrame({
            'Time': ['12:00-15:00', '15:00-18:00', '18:00-21:00', '21:00+'],
            'Over 2.5 %': [60, 64, 66, 62]
        })
        
        fig = px.bar(
            time_data,
            x='Time',
            y='Over 2.5 %',
            title="Over 2.5 Rate by Kick-off Time",
            color='Over 2.5 %',
            color_continuous_scale=['#ff4757', '#00d4aa']
        )
        
        fig.update_layout(
            template="plotly_dark",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    # Generate correlation data
    factors = ['Home Form', 'Away Form', 'League Position', 'Rest Days', 'H2H History']
    correlations = [0.72, 0.68, -0.45, 0.23, 0.56]
    
    fig = go.Figure(go.Bar(
        x=correlations,
        y=factors,
        orientation='h',
        marker=dict(
            color=correlations,
            colorscale=['#ff4757', '#ffa502', '#00d4aa'],
            showscale=True,
            colorbar=dict(title="Correlation")
        )
    ))
    
    fig.update_layout(
        title="Factors Correlated with Over 2.5 Goals",
        xaxis_title="Correlation Coefficient",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)


async def render_betting_performance(db_manager):
    """Render betting performance analytics"""
    
    st.markdown("## 💰 Betting Performance")
    
    # Get betting history from database
    betting_history = await db_manager.get_betting_history()
    
    if betting_history.empty:
        # Generate sample data
        betting_history = generate_sample_betting_history()
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_bets = len(betting_history)
    winning_bets = len(betting_history[betting_history['profit'] > 0])
    total_profit = betting_history['profit'].sum()
    roi = (total_profit / betting_history['stake'].sum() * 100) if not betting_history.empty else 0
    
    with col1:
        st.metric("Total Bets", f"{total_bets:,}")
        
    with col2:
        st.metric("Win Rate", f"{winning_bets/max(total_bets, 1):.1%}")
        
    with col3:
        st.metric("Total Profit", f"€{total_profit:.2f}")
        
    with col4:
        st.metric("ROI", f"{roi:.1f}%")
    
    # ROI progression chart
    betting_history['cumulative_profit'] = betting_history['profit'].cumsum()
    betting_history['cumulative_stake'] = betting_history['stake'].cumsum()
    betting_history['cumulative_roi'] = (
        betting_history['cumulative_profit'] / 
        betting_history['cumulative_stake'] * 100
    )
    
    fig = create_roi_chart(betting_history)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance by bet type
    col1, col2 = st.columns(2)
    
    with col1:
        # By bet type
        by_type = betting_history.groupby('bet_type').agg({
            'stake': 'sum',
            'profit': 'sum',
            'bet_id': 'count'
        }).reset_index()
        
        by_type['roi'] = (by_type['profit'] / by_type['stake'] * 100).round(1)
        by_type.columns = ['Type', 'Total Stake', 'Total Profit', 'Count', 'ROI %']
        
        fig = px.bar(
            by_type,
            x='Type',
            y='ROI %',
            title="ROI by Bet Type",
            color='ROI %',
            color_continuous_scale=['#ff4757', '#ffa502', '#00d4aa']
        )
        
        fig.update_layout(
            template="plotly_dark",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # By confidence level
        by_confidence = betting_history.groupby('confidence').agg({
            'stake': 'sum',
            'profit': 'sum',
            'bet_id': 'count'
        }).reset_index()
        
        by_confidence['win_rate'] = (
            betting_history.groupby('confidence')['is_won'].mean() * 100
        ).values
        
        fig = px.scatter(
            by_confidence,
            x='confidence',
            y='win_rate',
            size='count',
            title="Win Rate by Confidence Level",
            color='win_rate',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Confidence Level",
            yaxis_title="Win Rate (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly breakdown
    st.markdown("### 📅 Monthly Performance")
    
    monthly = betting_history.groupby(pd.Grouper(key='date', freq='M')).agg({
        'stake': 'sum',
        'profit': 'sum',
        'bet_id': 'count'
    }).reset_index()
    
    monthly['roi'] = (monthly['profit'] / monthly['stake'] * 100).fillna(0)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Profit/Loss', 'Monthly ROI'),
        row_heights=[0.6, 0.4]
    )
    
    # Profit/Loss bars
    colors = ['#00d4aa' if x > 0 else '#ff4757' for x in monthly['profit']]
    
    fig.add_trace(
        go.Bar(
            x=monthly['date'],
            y=monthly['profit'],
            name='Profit/Loss',
            marker_color=colors
        ),
        row=1, col=1
    )
    
    # ROI line
    fig.add_trace(
        go.Scatter(
            x=monthly['date'],
            y=monthly['roi'],
            mode='lines+markers',
            name='ROI %',
            line=dict(color='#ffa502', width=3)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


async def render_model_analytics(db_manager):
    """Render model performance analytics"""
    
    st.markdown("## 🤖 Model Performance Analytics")
    
    # Get model performance history
    performance_history = await db_manager.get_performance_history(days=90)
    
    if performance_history.empty:
        performance_history = generate_sample_performance_data()
    
    # Current model metrics
    latest = performance_history.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Accuracy",
            f"{latest['accuracy']:.1%}",
            f"{(latest['accuracy'] - performance_history['accuracy'].mean()):.1%}"
        )
        
    with col2:
        st.metric(
            "Predictions Today",
            f"{latest['predictions']:,}"
        )
        
    with col3:
        st.metric(
            "Avg Confidence",
            f"{np.random.uniform(0.7, 0.8):.1%}"
        )
        
    with col4:
        st.metric(
            "Model Version",
            "v2.1.3",
            "Updated 2h ago"
        )
    
    # Performance over time
    fig = go.Figure()
    
    # Accuracy line
    fig.add_trace(go.Scatter(
        x=performance_history['date'],
        y=performance_history['accuracy'],
        mode='lines',
        name='Accuracy',
        line=dict(color='#00d4aa', width=3)
    ))
    
    # Add rolling average
    performance_history['ma7'] = performance_history['accuracy'].rolling(7).mean()
    
    fig.add_trace(go.Scatter(
        x=performance_history['date'],
        y=performance_history['ma7'],
        mode='lines',
        name='7-day MA',
        line=dict(color='#ffa502', width=2, dash='dash')
    ))
    
    # Add confidence bands
    performance_history['upper'] = performance_history['ma7'] + performance_history['accuracy'].rolling(7).std()
    performance_history['lower'] = performance_history['ma7'] - performance_history['accuracy'].rolling(7).std()
    
    fig.add_trace(go.Scatter(
        x=performance_history['date'],
        y=performance_history['upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_history['date'],
        y=performance_history['lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Band',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ))
    
    fig.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis_tickformat='.1%',
        template="plotly_dark",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Feature Importance")
        
        # Get feature importance (would be from model in production)
        feature_importance = {
            'home_goals_avg': 0.125,
            'away_goals_avg': 0.118,
            'h2h_over25_rate': 0.095,
            'league_over25_avg': 0.089,
            'combined_attack_strength': 0.085,
            'home_form': 0.078,
            'away_form': 0.075,
            'home_over25_rate': 0.068,
            'away_over25_rate': 0.065,
            'defensive_weakness': 0.062
        }
        
        fig = create_feature_importance_chart(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### 🎲 Prediction Distribution")
        
        # Generate sample prediction distribution
        predictions = np.random.beta(7, 3, 1000)
        
        fig = create_distribution_chart(
            pd.DataFrame({'predictions': predictions}),
            'predictions',
            'Prediction Probability Distribution'
        )
        
        fig.update_layout(
            xaxis_tickformat='.0%',
            xaxis_title="Predicted Probability"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.markdown("### 🎯 Model Performance by League")
    
    # Generate sample confusion matrix data
    leagues_performance = []
    
    for league in list(LEAGUES.keys())[:6]:
        leagues_performance.append({
            'League': league.split(' ', 1)[1],
            'Accuracy': np.random.uniform(0.72, 0.85),
            'Precision': np.random.uniform(0.70, 0.83),
            'Recall': np.random.uniform(0.68, 0.80),
            'F1 Score': np.random.uniform(0.70, 0.82),
            'Predictions': np.random.randint(100, 500)
        })
    
    perf_df = pd.DataFrame(leagues_performance)
    
    # Format for display
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        perf_df[col] = (perf_df[col] * 100).round(1).astype(str) + '%'
    
    st.dataframe(perf_df, hide_index=True, use_container_width=True)


def generate_trend_data(time_period: str, granularity: str) -> pd.DataFrame:
    """Generate sample trend data"""
    
    # Determine date range
    if time_period == "Last 30 days":
        days = 30
    elif time_period == "Last 90 days":
        days = 90
    elif time_period == "This Season":
        days = 120
    else:
        days = 365
    
    # Generate dates
    if granularity == "Daily":
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    elif granularity == "Weekly":
        dates = pd.date_range(end=datetime.now(), periods=days//7, freq='W')
    else:
        dates = pd.date_range(end=datetime.now(), periods=days//30, freq='M')
    
    # Generate data with trend
    base_rate = 0.65
    trend = 0.0001 * np.arange(len(dates))
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    noise = np.random.normal(0, 0.02, len(dates))
    
    over25_rate = base_rate + trend + seasonal + noise
    over25_rate = np.clip(over25_rate, 0.4, 0.8) * 100
    
    avg_goals = 2.5 + trend * 10 + seasonal * 2 + np.random.normal(0, 0.1, len(dates))
    
    return pd.DataFrame({
        'date': dates,
        'over25_rate': over25_rate,
        'avg_goals': avg_goals
    })


def generate_sample_betting_history() -> pd.DataFrame:
    """Generate sample betting history"""
    
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    history = []
    
    for date in dates:
        n_bets = np.random.randint(1, 10)
        
        for _ in range(n_bets):
            bet_type = np.random.choice(['single', 'double', 'triple'])
            stake = np.random.choice([10, 20, 50, 100])
            odds = np.random.uniform(1.5, 8.0)
            confidence = np.random.choice(['High', 'Medium', 'Low'])
            
            # Win probability based on confidence
            win_prob = {'High': 0.7, 'Medium': 0.6, 'Low': 0.5}[confidence]
            is_won = np.random.random() < win_prob
            
            profit = stake * (odds - 1) if is_won else -stake
            
            history.append({
                'bet_id': f"BET{len(history)}",
                'date': date,
                'bet_type': bet_type,
                'stake': stake,
                'odds': odds,
                'confidence': confidence,
                'is_won': is_won,
                'profit': profit,
                'returns': stake * odds if is_won else 0
            })
    
    return pd.DataFrame(history)


def generate_sample_performance_data() -> pd.DataFrame:
    """Generate sample model performance data"""
    
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    data = []
    base_accuracy = 0.75
    
    for i, date in enumerate(dates):
        # Add trend and seasonality
        trend = 0.0001 * i
        weekly_pattern = 0.02 * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, 0.01)
        
        accuracy = base_accuracy + trend + weekly_pattern + noise
        accuracy = np.clip(accuracy, 0.65, 0.85)
        
        data.append({
            'date': date,
            'accuracy': accuracy,
            'predictions': np.random.randint(50, 200),
            'correct': int(accuracy * np.random.randint(50, 200))
        })
    
    return pd.DataFrame(data)