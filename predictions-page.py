"""
Predictions page for ProFootballAI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional
import logging

from config import LEAGUES, get_league_season
from ..theme import create_prediction_card
from ..charts import create_prediction_confidence_chart, create_feature_importance_chart
from ...models.predictor import PredictionResult
from ...utils.validators import validate_league_id, validate_season
from ...utils.exceptions import handle_exception, ExceptionHandler

logger = logging.getLogger("profootball.ui.predictions")


async def render(api_client, predictor, user_preferences: Dict[str, Any]):
    """Render predictions page"""
    
    st.markdown("# 🔮 Over 2.5 Goals Predictions")
    st.markdown("AI-powered predictions for upcoming matches")
    
    # Filters section
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # League selection
        league_options = list(LEAGUES.keys())
        selected_league = st.selectbox(
            "Select League",
            options=league_options,
            index=0,
            help="Choose the league for predictions"
        )
        
    with col2:
        # Date range
        days_ahead = st.selectbox(
            "Time Period",
            options=[1, 3, 7, 14, 30],
            index=2,
            format_func=lambda x: f"Next {x} days",
            help="Select prediction time range"
        )
        
    with col3:
        # Minimum probability filter
        min_prob = user_preferences.get('min_probability', 60)
        min_probability = st.slider(
            "Min Probability %",
            min_value=50,
            max_value=90,
            value=min_prob,
            step=5,
            help="Filter predictions by minimum probability"
        )
        
    with col4:
        # Confidence filter
        confidence_filter = st.multiselect(
            "Confidence Level",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
            help="Filter by prediction confidence"
        )
    
    # Advanced filters (expandable)
    with st.expander("⚙️ Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_odds = st.number_input(
                "Minimum Odds",
                min_value=1.1,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Filter by minimum betting odds"
            )
            
        with col2:
            max_odds = st.number_input(
                "Maximum Odds",
                min_value=1.5,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Filter by maximum betting odds"
            )
            
        with col3:
            positive_ev_only = st.checkbox(
                "Positive EV Only",
                value=True,
                help="Show only bets with positive expected value"
            )
    
    # Generate predictions button
    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
        await generate_predictions(
            api_client,
            predictor,
            selected_league,
            days_ahead,
            min_probability,
            confidence_filter,
            min_odds,
            max_odds,
            positive_ev_only
        )
    
    # Display predictions
    if 'current_predictions' in st.session_state:
        await display_predictions(predictor)
    else:
        st.info("👆 Click 'Generate Predictions' to start analyzing upcoming matches")
        
        # Show example predictions
        st.markdown("### 📊 Example Predictions")
        example_predictions = create_example_predictions()
        
        for i, pred in enumerate(example_predictions[:3]):
            st.markdown(
                create_prediction_card(
                    f"{pred['home_team']} vs {pred['away_team']}",
                    pred['probability'],
                    pred['odds'],
                    pred['confidence']
                ),
                unsafe_allow_html=True
            )


async def generate_predictions(
    api_client,
    predictor,
    league: str,
    days_ahead: int,
    min_probability: int,
    confidence_filter: List[str],
    min_odds: float,
    max_odds: float,
    positive_ev_only: bool
):
    """Generate predictions for upcoming matches"""
    
    try:
        # Get league info
        league_info = LEAGUES.get(league)
        if not league_info:
            st.error(f"League not found: {league}")
            return
            
        league_id = league_info['id']
        season = get_league_season(league)
        
        # Validate inputs
        league_id = validate_league_id(league_id)
        season = validate_season(season)
        
        with st.spinner(f"🔄 Fetching upcoming matches for {league}..."):
            # Get fixtures
            from_date = datetime.now()
            to_date = from_date + timedelta(days=days_ahead)
            
            fixtures = await api_client.get_fixtures(
                league_id=league_id,
                season=season,
                from_date=from_date,
                to_date=to_date,
                limit=50
            )
            
            if not fixtures:
                st.warning(f"No upcoming matches found for {league} in the next {days_ahead} days")
                return
                
            st.success(f"✅ Found {len(fixtures)} upcoming matches")
        
        # Generate predictions
        predictions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, fixture in enumerate(fixtures):
            status_text.text(f"Analyzing: {fixture.home_team} vs {fixture.away_team}")
            
            try:
                # Get team statistics
                with ExceptionHandler(logger):
                    # Generate features
                    features = await generate_match_features(
                        api_client,
                        predictor,
                        fixture,
                        league_id,
                        season
                    )
                    
                    # Make prediction
                    prediction_result = predictor.predict(features)
                    
                    # Calculate odds (simplified - in real app would get from bookmakers)
                    base_odds = 1 / max(prediction_result.probability, 0.2)
                    market_margin = 0.05
                    odds = round(base_odds * (1 - market_margin), 2)
                    odds = max(min_odds, min(max_odds, odds))
                    
                    # Apply filters
                    if (prediction_result.probability * 100 >= min_probability and
                        prediction_result.confidence in confidence_filter and
                        min_odds <= odds <= max_odds and
                        (not positive_ev_only or prediction_result.expected_value > 0)):
                        
                        predictions.append({
                            'fixture': fixture,
                            'prediction_result': prediction_result,
                            'odds': odds,
                            'features': features
                        })
                        
            except Exception as e:
                logger.error(f"Error predicting {fixture.match_string}: {e}")
                continue
                
            # Update progress
            progress_bar.progress((i + 1) / len(fixtures))
        
        progress_bar.empty()
        status_text.empty()
        
        if predictions:
            st.success(f"🎯 Generated {len(predictions)} predictions matching your criteria")
            
            # Store in session state
            st.session_state.current_predictions = predictions
            st.session_state.prediction_timestamp = datetime.now()
            
            # Update total predictions counter
            if 'total_predictions' not in st.session_state:
                st.session_state.total_predictions = 0
            st.session_state.total_predictions += len(predictions)
            
        else:
            st.warning("No matches found matching your criteria. Try adjusting the filters.")
            
    except Exception as e:
        error_msg = handle_exception(e, logger)
        st.error(f"❌ {error_msg}")


async def display_predictions(predictor):
    """Display generated predictions"""
    
    predictions = st.session_state.current_predictions
    timestamp = st.session_state.prediction_timestamp
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(predictions))
        
    with col2:
        avg_prob = np.mean([p['prediction_result'].probability for p in predictions])
        st.metric("Avg Probability", f"{avg_prob:.1%}")
        
    with col3:
        high_conf = sum(1 for p in predictions if p['prediction_result'].confidence == "High")
        st.metric("High Confidence", f"{high_conf}/{len(predictions)}")
        
    with col4:
        positive_ev = sum(1 for p in predictions if p['prediction_result'].expected_value > 0)
        st.metric("Positive EV", f"{positive_ev}/{len(predictions)}")
    
    st.caption(f"Generated at {timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    # Sorting options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Date", "Probability", "Expected Value", "Odds"],
            index=1
        )
        
    with col2:
        sort_order = st.radio(
            "Order",
            options=["Desc", "Asc"],
            horizontal=True
        )
    
    # Sort predictions
    sorted_predictions = sort_predictions(predictions, sort_by, sort_order == "Asc")
    
    # Display predictions
    st.markdown("### 🎯 Match Predictions")
    
    for i, pred_data in enumerate(sorted_predictions):
        fixture = pred_data['fixture']
        prediction = pred_data['prediction_result']
        odds = pred_data['odds']
        
        with st.expander(
            f"⚽ {fixture.home_team} vs {fixture.away_team} - "
            f"{prediction.probability:.1%} @ {odds:.2f}",
            expanded=(i < 5)  # Expand first 5
        ):
            # Match info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**📅 Date:** {fixture.date.strftime('%A, %d %B %Y at %H:%M')}")
                st.markdown(f"**🏟️ Venue:** {fixture.venue}")
                st.markdown(f"**🏆 League:** {fixture.league}")
                
            with col2:
                # Prediction summary
                if prediction.prediction == 1:
                    st.success(f"✅ **Prediction: Over 2.5 Goals**")
                else:
                    st.error(f"❌ **Prediction: Under 2.5 Goals**")
                    
                st.metric("Probability", f"{prediction.probability:.1%}")
                st.metric("Confidence", prediction.confidence)
                
            # Detailed analysis
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Betting Analysis**")
                st.write(f"- Odds: **@{odds:.2f}**")
                st.write(f"- Expected Value: **{prediction.expected_value:.3f}**")
                st.write(f"- Kelly Stake: **{prediction.expected_value / (odds - 1):.1%}**")
                
            with col2:
                st.markdown("**⚠️ Risk Factors**")
                if prediction.risk_factors:
                    for risk in prediction.risk_factors[:3]:
                        st.write(f"- {risk}")
                else:
                    st.write("- No significant risks identified")
                    
            with col3:
                st.markdown("**✨ Key Factors**")
                top_features = list(prediction.feature_importance.items())[:3]
                for feat, imp in top_features:
                    st.write(f"- {feat}: {imp:.3f}")
            
            # Feature importance chart
            if st.checkbox(f"Show detailed analysis", key=f"analysis_{i}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature importance
                    fig = create_feature_importance_chart(prediction.feature_importance)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    # Prediction explanation
                    st.markdown("**🤖 AI Explanation**")
                    explanation = predictor.explain_prediction(
                        pred_data['features'],
                        prediction
                    )
                    st.text(explanation)
    
    # Export options
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export to CSV", use_container_width=True):
            csv_data = export_predictions_csv(sorted_predictions)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Report generation coming soon!")
            
    with col3:
        if st.button("🔄 Refresh Predictions", use_container_width=True):
            del st.session_state.current_predictions
            st.rerun()


async def generate_match_features(api_client, predictor, fixture, league_id: int, season: int) -> Dict[str, Any]:
    """Generate features for a match"""
    
    # Get team statistics
    team_stats = await api_client.get_team_stats(league_id, season)
    
    # Find stats for both teams
    home_stats = next((s for s in team_stats if s.team_id == fixture.home_team_id), None)
    away_stats = next((s for s in team_stats if s.team_id == fixture.away_team_id), None)
    
    # Get H2H statistics
    h2h_stats = await api_client.get_h2h_stats(fixture.home_team_id, fixture.away_team_id)
    
    # Get recent form
    home_form = await api_client.get_team_form(fixture.home_team_id)
    away_form = await api_client.get_team_form(fixture.away_team_id)
    
    # Build features dictionary
    features = {
        'home_goals_avg': home_stats.goals_per_match if home_stats else 1.5,
        'away_goals_avg': away_stats.goals_per_match if away_stats else 1.3,
        'home_goals_conceded_avg': (home_stats.goals_against / max(home_stats.matches_played, 1)) if home_stats else 1.2,
        'away_goals_conceded_avg': (away_stats.goals_against / max(away_stats.matches_played, 1)) if away_stats else 1.4,
        'home_over25_rate': home_stats.over25_rate if home_stats else 0.5,
        'away_over25_rate': away_stats.over25_rate if away_stats else 0.45,
        'h2h_over25_rate': h2h_stats['over25_rate'],
        'home_form': calculate_form_points(home_form),
        'away_form': calculate_form_points(away_form),
        'total_matches': (home_stats.matches_played if home_stats else 0) + (away_stats.matches_played if away_stats else 0),
        'league_over25_avg': np.mean([s.over25_rate for s in team_stats]) if team_stats else 0.5,
        'combined_attack_strength': ((home_stats.goals_per_match if home_stats else 1.5) + 
                                    (away_stats.goals_per_match if away_stats else 1.3)) / 2,
        'match_date': fixture.date
    }
    
    return features


def calculate_form_points(form: List[str]) -> int:
    """Calculate form points from W/D/L list"""
    points_map = {'W': 3, 'D': 1, 'L': 0}
    return sum(points_map.get(result, 0) for result in form)


def sort_predictions(predictions: List[Dict], sort_by: str, ascending: bool) -> List[Dict]:
    """Sort predictions list"""
    
    if sort_by == "Date":
        key_func = lambda x: x['fixture'].date
    elif sort_by == "Probability":
        key_func = lambda x: x['prediction_result'].probability
    elif sort_by == "Expected Value":
        key_func = lambda x: x['prediction_result'].expected_value
    elif sort_by == "Odds":
        key_func = lambda x: x['odds']
    else:
        return predictions
        
    return sorted(predictions, key=key_func, reverse=not ascending)


def export_predictions_csv(predictions: List[Dict]) -> str:
    """Export predictions to CSV format"""
    
    data = []
    for pred in predictions:
        fixture = pred['fixture']
        prediction = pred['prediction_result']
        
        data.append({
            'Date': fixture.date.strftime('%Y-%m-%d'),
            'Time': fixture.date.strftime('%H:%M'),
            'Home Team': fixture.home_team,
            'Away Team': fixture.away_team,
            'League': fixture.league,
            'Prediction': 'Over 2.5' if prediction.prediction == 1 else 'Under 2.5',
            'Probability %': round(prediction.probability * 100, 1),
            'Confidence': prediction.confidence,
            'Odds': pred['odds'],
            'Expected Value': round(prediction.expected_value, 3),
            'Risk Factors': '; '.join(prediction.risk_factors)
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def create_example_predictions() -> List[Dict[str, Any]]:
    """Create example predictions for demo"""
    
    teams = [
        ("Manchester City", "Arsenal"),
        ("Real Madrid", "Barcelona"),
        ("Bayern Munich", "Borussia Dortmund"),
        ("Juventus", "AC Milan"),
        ("PSG", "Monaco")
    ]
    
    predictions = []
    
    for home, away in teams:
        prob = np.random.uniform(0.55, 0.85)
        odds = round(1.85 + np.random.uniform(-0.3, 0.5), 2)
        
        predictions.append({
            'home_team': home,
            'away_team': away,
            'probability': prob,
            'odds': odds,
            'confidence': 'High' if prob > 0.75 else 'Medium' if prob > 0.65 else 'Low',
            'expected_value': (prob * odds) - 1
        })
    
    return predictions