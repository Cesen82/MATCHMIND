"""
ProFootballAI - Professional Over 2.5 Goals Prediction Suite
Main entry point for the Streamlit application
"""

import streamlit as st
import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import UI_CONFIG, LOGGING_CONFIG
from src.ui.theme import apply_theme
from src.ui.pages import dashboard, predictions, betting_slips, analytics
from src.utils.logger import setup_logging
from src.api.football_api import FootballAPIClient
from src.models.predictor import Over25Predictor
from src.models.bet_optimizer import BetOptimizer
from src.data.database import DatabaseManager

# Setup logging
setup_logging(LOGGING_CONFIG)
logger = logging.getLogger("profootball.main")


class ProFootballApp:
    """Main application class"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="ProFootballAI - Over 2.5 Predictor",
            page_icon=UI_CONFIG["page_icon"],
            layout=UI_CONFIG["layout"],
            initial_sidebar_state=UI_CONFIG["sidebar_state"],
            menu_items={
                'Get Help': 'https://github.com/yourusername/profootball-ai',
                'Report a bug': 'https://github.com/yourusername/profootball-ai/issues',
                'About': 'ProFootballAI - Professional AI-powered football predictions'
            }
        )
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.api_client = None
            st.session_state.predictor = None
            st.session_state.bet_optimizer = None
            st.session_state.db_manager = None
            st.session_state.current_league = None
            st.session_state.current_season = None
            st.session_state.predictions_cache = {}
            st.session_state.user_preferences = {
                'risk_tolerance': 'medium',
                'min_probability': 60,
                'bankroll': 1000,
                'theme': 'dark'
            }
            
    async def initialize_components(self):
        """Initialize core components asynchronously"""
        with st.spinner("🚀 Initializing AI components..."):
            try:
                # Initialize API client
                if not st.session_state.api_client:
                    st.session_state.api_client = FootballAPIClient()
                    
                # Initialize predictor
                if not st.session_state.predictor:
                    st.session_state.predictor = Over25Predictor()
                    
                # Initialize bet optimizer
                if not st.session_state.bet_optimizer:
                    st.session_state.bet_optimizer = BetOptimizer()
                    
                # Initialize database
                if not st.session_state.db_manager:
                    st.session_state.db_manager = DatabaseManager()
                    await st.session_state.db_manager.initialize()
                    
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize components: {e}")
                st.error(f"❌ Initialization failed: {str(e)}")
                return False
                
    def render_sidebar(self):
        """Render sidebar with navigation and settings"""
        with st.sidebar:
            # Logo and title
            st.markdown(
                """
                <div style="text-align: center; padding: 1rem;">
                    <h1 style="color: #00d4aa;">⚽ ProFootballAI</h1>
                    <p style="color: #888;">Professional Over 2.5 Predictor</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.divider()
            
            # Navigation
            st.markdown("### 📊 Navigation")
            page = st.radio(
                "Select Page",
                ["Dashboard", "Predictions", "Betting Slips", "Analytics"],
                label_visibility="collapsed"
            )
            
            st.divider()
            
            # User preferences
            st.markdown("### ⚙️ Settings")
            
            # Risk tolerance
            st.session_state.user_preferences['risk_tolerance'] = st.select_slider(
                "Risk Tolerance",
                options=["conservative", "medium", "aggressive"],
                value=st.session_state.user_preferences['risk_tolerance'],
                help="Adjust your betting risk profile"
            )
            
            # Minimum probability
            st.session_state.user_preferences['min_probability'] = st.slider(
                "Min Probability (%)",
                min_value=50,
                max_value=90,
                value=st.session_state.user_preferences['min_probability'],
                step=5,
                help="Minimum probability for predictions"
            )
            
            # Bankroll
            st.session_state.user_preferences['bankroll'] = st.number_input(
                "Bankroll (€)",
                min_value=10,
                max_value=100000,
                value=st.session_state.user_preferences['bankroll'],
                step=100,
                help="Your total betting bankroll"
            )
            
            st.divider()
            
            # API Status
            if st.session_state.api_client:
                status = st.session_state.api_client.get_status()
                st.markdown("### 📡 API Status")
                
                # Rate limit status
                rate_limit = status['rate_limit']
                if rate_limit['remaining_hourly'] > 50:
                    status_color = "🟢"
                elif rate_limit['remaining_hourly'] > 20:
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                    
                st.metric(
                    "API Calls Remaining",
                    f"{rate_limit['remaining_hourly']}/h",
                    delta=f"{rate_limit['remaining_daily']}/day"
                )
                
                # Cache status
                cache_status = status['cache']
                st.metric(
                    "Cache Hit Rate",
                    f"{cache_status['hit_rate']:.1%}",
                    delta=f"{cache_status['size_mb']:.1f} MB"
                )
                
            st.divider()
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
                    
            with col2:
                if st.button("💾 Export", use_container_width=True):
                    st.info("Export feature coming soon!")
                    
            # Footer
            st.markdown(
                """
                <div style="text-align: center; padding: 2rem 0; color: #666;">
                    <small>
                        ProFootballAI v2.0<br>
                        ⚠️ Bet Responsibly
                    </small>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            return page
            
    def render_header_metrics(self):
        """Render header with key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.predictor and st.session_state.predictor.is_trained:
            metrics = st.session_state.predictor.performance_metrics
            
            with col1:
                st.metric(
                    "🎯 Model Accuracy",
                    f"{metrics['accuracy']:.1%}",
                    delta="+2.3%",
                    help="Current model accuracy on test set"
                )
                
            with col2:
                st.metric(
                    "📊 AUC-ROC Score",
                    f"{metrics['auc_roc']:.3f}",
                    delta="+0.021",
                    help="Area under ROC curve"
                )
                
        with col3:
            total_predictions = st.session_state.get('total_predictions', 0)
            st.metric(
                "⚽ Predictions Made",
                f"{total_predictions:,}",
                delta=f"+{total_predictions // 10}",
                help="Total predictions in this session"
            )
            
        with col4:
            win_rate = st.session_state.get('win_rate', 0.0)
            st.metric(
                "🏆 Win Rate",
                f"{win_rate:.1%}",
                delta=f"+{win_rate * 0.05:.1%}",
                help="Overall prediction success rate"
            )
            
    async def run(self):
        """Main application loop"""
        # Apply theme
        apply_theme()
        
        # Initialize components
        if not st.session_state.get('components_initialized', False):
            if await self.initialize_components():
                st.session_state.components_initialized = True
            else:
                st.error("Failed to initialize application. Please check your configuration.")
                return
                
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Render header metrics
        self.render_header_metrics()
        
        # Divider
        st.divider()
        
        # Render selected page
        try:
            if page == "Dashboard":
                await dashboard.render(
                    st.session_state.api_client,
                    st.session_state.predictor,
                    st.session_state.db_manager
                )
            elif page == "Predictions":
                await predictions.render(
                    st.session_state.api_client,
                    st.session_state.predictor,
                    st.session_state.user_preferences
                )
            elif page == "Betting Slips":
                await betting_slips.render(
                    st.session_state.api_client,
                    st.session_state.predictor,
                    st.session_state.bet_optimizer,
                    st.session_state.user_preferences
                )
            elif page == "Analytics":
                await analytics.render(
                    st.session_state.api_client,
                    st.session_state.db_manager
                )
                
        except Exception as e:
            logger.error(f"Error rendering page {page}: {e}")
            st.error(f"❌ Error loading page: {str(e)}")
            
            # Show debug info in development
            if st.secrets.get("debug", False):
                with st.expander("🐛 Debug Information"):
                    st.exception(e)
                    

def main():
    """Entry point"""
    app = ProFootballApp()
    
    # Run async app
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application error: {str(e)}")
    finally:
        # Cleanup
        if st.session_state.api_client:
            loop.run_until_complete(st.session_state.api_client.close())
        loop.close()


if __name__ == "__main__":
    main()