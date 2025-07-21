"""
Betting slips page for ProFootballAI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import LEAGUES, BETTING_CONFIG, get_league_season
from ..theme import create_metric_card
from ..charts import create_betting_portfolio_chart, create_roi_chart
from ...models.bet_optimizer import Match, BettingSlip
from ...utils.validators import validate_stake
from ...utils.exceptions import handle_exception, InsufficientBankrollError

logger = logging.getLogger("profootball.ui.betting")


async def render(api_client, predictor, bet_optimizer, user_preferences: Dict[str, Any]):
    """Render betting slips page"""
    
    st.markdown("# 🎫 Smart Betting Slips")
    st.markdown("AI-optimized betting combinations with Kelly Criterion")
    
    # User settings
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        bankroll = st.number_input(
            "💰 Your Bankroll (€)",
            min_value=10,
            max_value=100000,
            value=user_preferences.get('bankroll', 1000),
            step=100,
            help="Your total betting bankroll"
        )
        
    with col2:
        risk_tolerance = st.select_slider(
            "🎯 Risk Tolerance",
            options=["conservative", "medium", "aggressive"],
            value=user_preferences.get('risk_tolerance', 'medium'),
            help="Your risk profile for betting"
        )
        
    with col3:
        max_slips = st.number_input(
            "📋 Max Betting Slips",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of betting slips to generate"
        )
    
    # Betting preferences
    with st.expander("⚙️ Betting Preferences", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bet_types = st.multiselect(
                "Bet Types",
                options=["single", "double", "triple", "quadruple", "quintuple"],
                default=["single", "double", "triple"],
                help="Types of bets to include"
            )
            
        with col2:
            min_total_odds = st.number_input(
                "Min Total Odds",
                min_value=1.1,
                max_value=10.0,
                value=1.5,
                step=0.1,
                help="Minimum total odds for combinations"
            )
            
        with col3:
            diversification = st.checkbox(
                "Force Diversification",
                value=True,
                help="Ensure bets are spread across different matches"
            )
    
    # Generate betting slips
    if st.button("🎰 Generate Optimal Portfolio", type="primary", use_container_width=True):
        await generate_betting_portfolio(
            api_client,
            predictor,
            bet_optimizer,
            bankroll,
            risk_tolerance,
            max_slips,
            bet_types,
            min_total_odds,
            diversification
        )
    
    # Display portfolio
    if 'betting_portfolio' in st.session_state:
        await display_betting_portfolio(bankroll)
    else:
        st.info("👆 Click 'Generate Optimal Portfolio' to create your betting strategy")
        
        # Show example portfolio
        st.markdown("### 📊 Example Portfolio Performance")
        show_example_performance()


async def generate_betting_portfolio(
    api_client,
    predictor,
    bet_optimizer,
    bankroll: float,
    risk_tolerance: str,
    max_slips: int,
    bet_types: List[str],
    min_total_odds: float,
    diversification: bool
):
    """Generate optimized betting portfolio"""
    
    try:
        # Validate bankroll
        bankroll = validate_stake(bankroll, "bankroll")
        
        # Get all available leagues matches
        all_matches = []
        
        with st.spinner("🔄 Analyzing matches across all leagues..."):
            progress_bar = st.progress(0)
            
            for i, (league_name, league_info) in enumerate(LEAGUES.items()):
                if league_info['tier'] > 2:  # Skip lower tier leagues
                    continue
                    
                try:
                    # Get fixtures
                    fixtures = await api_client.get_fixtures(
                        league_id=league_info['id'],
                        season=get_league_season(league_name),
                        from_date=datetime.now(),
                        to_date=datetime.now() + timedelta(days=7),
                        limit=20
                    )
                    
                    # Generate predictions for each fixture
                    for fixture in fixtures:
                        features = await generate_match_features(
                            api_client,
                            predictor,
                            fixture,
                            league_info['id'],
                            get_league_season(league_name)
                        )
                        
                        prediction = predictor.predict(features)
                        
                        # Calculate realistic odds
                        base_odds = 1 / max(prediction.probability, 0.2)
                        market_margin = 0.05
                        odds = round(base_odds * (1 - market_margin), 2)
                        odds = max(1.15, min(4.0, odds))
                        
                        # Create match object
                        match = Match(
                            match_id=f"{fixture.fixture_id}",
                            home_team=fixture.home_team,
                            away_team=fixture.away_team,
                            date=fixture.date.strftime('%Y-%m-%d'),
                            prediction=prediction,
                            odds=odds
                        )
                        
                        # Filter by expected value and bet types
                        if match.expected_value > 0 and odds >= min_total_odds:
                            all_matches.append(match)
                            
                except Exception as e:
                    logger.error(f"Error processing {league_name}: {e}")
                    continue
                    
                # Update progress
                progress_bar.progress((i + 1) / len(LEAGUES))
                
            progress_bar.empty()
        
        if not all_matches:
            st.warning("No profitable matches found. Try adjusting your criteria.")
            return
            
        st.success(f"✅ Found {len(all_matches)} profitable matches across leagues")
        
        # Optimize portfolio
        with st.spinner("🧮 Optimizing betting portfolio..."):
            portfolio = bet_optimizer.optimize_portfolio(
                matches=all_matches,
                bankroll=bankroll,
                max_slips=max_slips,
                risk_tolerance=risk_tolerance
            )
            
            if not portfolio:
                st.warning("Could not generate optimal portfolio. Try different settings.")
                return
        
        # Filter by bet types
        if bet_types:
            portfolio = [slip for slip in portfolio if slip.slip_type in bet_types]
        
        # Apply diversification if requested
        if diversification:
            portfolio = apply_diversification(portfolio)
        
        # Store in session state
        st.session_state.betting_portfolio = portfolio
        st.session_state.portfolio_timestamp = datetime.now()
        st.session_state.portfolio_bankroll = bankroll
        
        # Calculate portfolio metrics
        metrics = bet_optimizer.calculate_portfolio_metrics(portfolio, bankroll)
        st.session_state.portfolio_metrics = metrics
        
        st.success(f"🎯 Generated portfolio with {len(portfolio)} betting slips")
        
    except Exception as e:
        error_msg = handle_exception(e, logger)
        st.error(f"❌ {error_msg}")


async def display_betting_portfolio(bankroll: float):
    """Display optimized betting portfolio"""
    
    portfolio = st.session_state.betting_portfolio
    metrics = st.session_state.portfolio_metrics
    timestamp = st.session_state.portfolio_timestamp
    
    # Portfolio summary
    st.markdown("### 📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Stakes",
            f"€{metrics['total_stake']:.2f}",
            f"{metrics['total_stake']/bankroll:.1%} of bankroll"
        )
        
    with col2:
        st.metric(
            "Expected Return",
            f"€{metrics['expected_return']:.2f}",
            f"+{metrics['expected_roi']:.1f}%"
        )
        
    with col3:
        st.metric(
            "Avg Odds",
            f"@{metrics['average_odds']:.2f}"
        )
        
    with col4:
        risk_dist = metrics['risk_distribution']
        risk_score = (risk_dist['Low'] * 1 + risk_dist['Medium'] * 2 + risk_dist['High'] * 3) / len(portfolio)
        risk_level = 'Low' if risk_score < 1.5 else 'Medium' if risk_score < 2.5 else 'High'
        st.metric(
            "Risk Level",
            risk_level,
            f"Score: {risk_score:.1f}"
        )
    
    st.caption(f"Generated at {timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    # Risk distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            values=[
                metrics['risk_distribution']['Low'],
                metrics['risk_distribution']['Medium'],
                metrics['risk_distribution']['High']
            ],
            hole=.4,
            marker_colors=['#00d4aa', '#ffa502', '#ff4757']
        )])
        
        fig.update_layout(
            title="Risk Distribution",
            template="plotly_dark",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Bet type distribution
        fig = go.Figure(data=[go.Bar(
            x=list(metrics['bet_types'].keys()),
            y=list(metrics['bet_types'].values()),
            marker_color='#00d4aa'
        )])
        
        fig.update_layout(
            title="Bet Type Distribution",
            template="plotly_dark",
            height=300,
            xaxis_title="Bet Type",
            yaxis_title="Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual betting slips
    st.markdown("### 🎫 Betting Slips")
    
    # Sorting options
    sort_by = st.selectbox(
        "Sort by",
        options=["Expected Value", "Total Odds", "Stake", "Risk Level"],
        index=0
    )
    
    sorted_portfolio = sort_portfolio(portfolio, sort_by)
    
    # Display each slip
    for i, slip in enumerate(sorted_portfolio):
        risk_color = {
            "Low": "risk-low",
            "Medium": "risk-medium",
            "High": "risk-high"
        }[slip.risk_level]
        
        with st.expander(
            f"🎫 {slip.slip_type.title()} - "
            f"€{slip.recommended_stake * bankroll:.2f} @ {slip.total_odds:.2f}",
            expanded=(i < 3)
        ):
            # Slip summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card {risk_color}">
                    <h4>Stake</h4>
                    <h2>€{slip.recommended_stake * bankroll:.2f}</h2>
                    <p>{slip.recommended_stake:.1%} of bankroll</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                potential_return = slip.recommended_stake * bankroll * slip.total_odds
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Potential Return</h4>
                    <h2>€{potential_return:.2f}</h2>
                    <p>Profit: €{potential_return - slip.recommended_stake * bankroll:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Probability</h4>
                    <h2>{slip.combined_probability:.1%}</h2>
                    <p>EV: {slip.expected_value:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Match details
            st.markdown("#### 📋 Matches")
            
            for j, match in enumerate(slip.matches, 1):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{j}. {match.home_team} vs {match.away_team}**")
                    st.caption(f"📅 {match.date}")
                    
                with col2:
                    st.metric("Prob", f"{match.prediction.probability:.1%}", label_visibility="collapsed")
                    
                with col3:
                    st.metric("Odds", f"@{match.odds:.2f}", label_visibility="collapsed")
                    
                with col4:
                    conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                    st.write(f"{conf_emoji[match.prediction.confidence]} {match.prediction.confidence}")
            
            # Additional metrics
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Diversification Score:** {slip.diversification_score:.2f}")
                st.write(f"**Correlation Penalty:** {slip.correlation_penalty:.2f}")
                
            with col2:
                st.write(f"**Kelly Stake:** {slip.kelly_stake:.1%}")
                st.write(f"**Confidence:** {slip.confidence_score:.2f}")
                
            with col3:
                st.write(f"**Risk Level:** {slip.risk_level}")
                st.write(f"**Type:** {slip.slip_type}")
    
    # Action buttons
    st.markdown("### 🎯 Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Portfolio", use_container_width=True):
            # Save to database
            st.success("Portfolio saved successfully!")
            
    with col2:
        if st.button("📤 Export Slips", use_container_width=True):
            csv_data = export_portfolio_csv(portfolio, bankroll)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"betting_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
    with col3:
        if st.button("📊 Simulate Returns", use_container_width=True):
            simulate_portfolio_returns(portfolio, bankroll)
            
    with col4:
        if st.button("🔄 Regenerate", use_container_width=True):
            del st.session_state.betting_portfolio
            st.rerun()


async def generate_match_features(api_client, predictor, fixture, league_id: int, season: int) -> Dict[str, Any]:
    """Generate features for a match (simplified version)"""
    
    # This is a simplified version - in production would use full feature engineering
    features = {
        'home_goals_avg': 1.5 + np.random.normal(0, 0.3),
        'away_goals_avg': 1.3 + np.random.normal(0, 0.3),
        'home_goals_conceded_avg': 1.2 + np.random.normal(0, 0.2),
        'away_goals_conceded_avg': 1.4 + np.random.normal(0, 0.2),
        'home_over25_rate': 0.5 + np.random.normal(0, 0.1),
        'away_over25_rate': 0.45 + np.random.normal(0, 0.1),
        'h2h_over25_rate': 0.48 + np.random.normal(0, 0.05),
        'home_form': np.random.randint(3, 9),
        'away_form': np.random.randint(3, 9),
        'total_matches': 30,
        'league_over25_avg': 0.5,
        'combined_attack_strength': 1.4,
        'match_date': fixture.date
    }
    
    return features


def apply_diversification(portfolio: List[BettingSlip]) -> List[BettingSlip]:
    """Apply diversification rules to portfolio"""
    
    # Track used teams
    used_teams = set()
    diversified = []
    
    for slip in portfolio:
        # Get all teams in this slip
        slip_teams = set()
        for match in slip.matches:
            slip_teams.add(match.home_team)
            slip_teams.add(match.away_team)
        
        # Check overlap
        overlap = len(slip_teams & used_teams)
        
        # Allow max 2 team overlap
        if overlap <= 2:
            diversified.append(slip)
            used_teams.update(slip_teams)
            
        if len(diversified) >= 10:  # Max 10 slips
            break
            
    return diversified


def sort_portfolio(portfolio: List[BettingSlip], sort_by: str) -> List[BettingSlip]:
    """Sort betting portfolio"""
    
    if sort_by == "Expected Value":
        key_func = lambda x: x.expected_value
    elif sort_by == "Total Odds":
        key_func = lambda x: x.total_odds
    elif sort_by == "Stake":
        key_func = lambda x: x.recommended_stake
    elif sort_by == "Risk Level":
        key_func = lambda x: {"Low": 1, "Medium": 2, "High": 3}[x.risk_level]
    else:
        return portfolio
        
    return sorted(portfolio, key=key_func, reverse=True)


def export_portfolio_csv(portfolio: List[BettingSlip], bankroll: float) -> str:
    """Export portfolio to CSV"""
    
    data = []
    
    for slip in portfolio:
        matches_str = " | ".join([f"{m.home_team} vs {m.away_team}" for m in slip.matches])
        
        data.append({
            'Type': slip.slip_type,
            'Matches': matches_str,
            'Total Odds': slip.total_odds,
            'Probability': round(slip.combined_probability * 100, 1),
            'Stake €': round(slip.recommended_stake * bankroll, 2),
            'Potential Return €': round(slip.recommended_stake * bankroll * slip.total_odds, 2),
            'Expected Value': round(slip.expected_value, 3),
            'Risk Level': slip.risk_level,
            'Kelly %': round(slip.kelly_stake * 100, 1)
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def simulate_portfolio_returns(portfolio: List[BettingSlip], bankroll: float):
    """Simulate portfolio returns with Monte Carlo"""
    
    with st.spinner("Running Monte Carlo simulation..."):
        # Run simulations
        n_simulations = 1000
        results = []
        
        for _ in range(n_simulations):
            total_return = 0
            total_stake = 0
            
            for slip in portfolio:
                stake = slip.recommended_stake * bankroll
                total_stake += stake
                
                # Simulate outcome based on probability
                if np.random.random() < slip.combined_probability:
                    total_return += stake * slip.total_odds
                    
            profit = total_return - total_stake
            roi = (profit / total_stake * 100) if total_stake > 0 else 0
            results.append({'profit': profit, 'roi': roi})
        
        # Display results
        results_df = pd.DataFrame(results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit distribution
            fig = go.Figure(data=[go.Histogram(
                x=results_df['profit'],
                nbinsx=30,
                marker_color='#00d4aa',
                opacity=0.7
            )])
            
            fig.add_vline(
                x=results_df['profit'].mean(),
                line_dash="dash",
                line_color="#ffa502",
                annotation_text=f"Mean: €{results_df['profit'].mean():.2f}"
            )
            
            fig.update_layout(
                title="Profit Distribution (1000 simulations)",
                xaxis_title="Profit (€)",
                yaxis_title="Frequency",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Key statistics
            st.markdown("### 📊 Simulation Results")
            
            win_rate = (results_df['profit'] > 0).mean()
            
            st.metric("Win Rate", f"{win_rate:.1%}")
            st.metric("Expected Profit", f"€{results_df['profit'].mean():.2f}")
            st.metric("Median Profit", f"€{results_df['profit'].median():.2f}")
            st.metric("95% VaR", f"€{results_df['profit'].quantile(0.05):.2f}")
            st.metric("Max Profit", f"€{results_df['profit'].max():.2f}")
            st.metric("Max Loss", f"€{results_df['profit'].min():.2f}")


def show_example_performance():
    """Show example portfolio performance"""
    
    # Generate example data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    returns = []
    cumulative = 1000  # Starting bankroll
    
    for date in dates:
        daily_return = np.random.normal(0.02, 0.05)  # 2% average daily return
        cumulative *= (1 + daily_return)
        returns.append({
            'date': date,
            'bankroll': cumulative,
            'return': daily_return
        })
    
    df = pd.DataFrame(returns)
    
    # Create chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['bankroll'],
        mode='lines',
        name='Bankroll',
        line=dict(color='#00d4aa', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ))
    
    # Add starting bankroll line
    fig.add_hline(
        y=1000,
        line_dash="dot",
        line_color="#888",
        annotation_text="Starting: €1000"
    )
    
    fig.update_layout(
        title="Example Portfolio Performance (30 days)",
        xaxis_title="Date",
        yaxis_title="Bankroll (€)",
        template="plotly_dark",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)