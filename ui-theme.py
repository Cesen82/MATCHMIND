"""
Advanced theming system for ProFootballAI
"""

import streamlit as st
from config import UI_CONFIG


def apply_theme():
    """Apply custom dark theme with modern styling"""
    
    theme_css = f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main {{
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            font-family: 'Inter', sans-serif;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif;
            color: #f0f6fc;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            font-weight: 600;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            border-left: 4px solid {UI_CONFIG['primary_color']};
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, {UI_CONFIG['primary_color']}, {UI_CONFIG['secondary_color']});
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .metric-card:hover::before {{
            opacity: 1;
        }}
        
        /* Prediction Cards */
        .prediction-card {{
            background: linear-gradient(135deg, #0d4f3c 0%, #047857 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 212, 170, 0.3);
            backdrop-filter: blur(10px);
        }}
        
        /* Risk Level Cards */
        .risk-high {{
            border-left-color: {UI_CONFIG['danger_color']} !important;
            background: linear-gradient(135deg, #2d1b2e 0%, #4a1b2e 100%) !important;
        }}
        
        .risk-medium {{
            border-left-color: {UI_CONFIG['warning_color']} !important;
            background: linear-gradient(135deg, #2d2a1b 0%, #4a421b 100%) !important;
        }}
        
        .risk-low {{
            border-left-color: {UI_CONFIG['success_color']} !important;
            background: linear-gradient(135deg, #1b2d2a 0%, #1b4a42 100%) !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {UI_CONFIG['primary_color']}, {UI_CONFIG['secondary_color']});
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            background: linear-gradient(135deg, {UI_CONFIG['secondary_color']}, {UI_CONFIG['primary_color']});
        }}
        
        /* Select boxes */
        .stSelectbox > div > div {{
            background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
            border: 1px solid #30363d;
            border-radius: 8px;
            transition: border-color 0.3s ease;
        }}
        
        .stSelectbox > div > div:hover {{
            border-color: {UI_CONFIG['primary_color']};
        }}
        
        /* Sliders */
        .stSlider > div > div > div > div {{
            background: linear-gradient(90deg, {UI_CONFIG['primary_color']} 0%, {UI_CONFIG['secondary_color']} 100%);
        }}
        
        /* Text inputs */
        .stTextInput > div > div > input {{
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 8px;
            color: #f0f6fc;
            transition: border-color 0.3s ease;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {UI_CONFIG['primary_color']};
            box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2);
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #30363d;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: transparent;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: #21262d;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #30363d;
            transition: all 0.3s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #30363d;
            border-color: {UI_CONFIG['primary_color']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {UI_CONFIG['primary_color']}, {UI_CONFIG['secondary_color']}) !important;
            color: white !important;
            border-color: transparent !important;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
            border-radius: 8px;
            border: 1px solid #30363d;
            transition: all 0.3s ease;
        }}
        
        .streamlit-expanderHeader:hover {{
            border-color: {UI_CONFIG['primary_color']};
            background: linear-gradient(135deg, #30363d 0%, #3c424a 100%);
        }}
        
        /* Progress bars */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {UI_CONFIG['primary_color']}, {UI_CONFIG['secondary_color']});
            border-radius: 4px;
        }}
        
        /* Alerts */
        .stAlert {{
            background: linear-gradient(135deg, #21262d 0%, #30363d 100%);
            border: 1px solid #30363d;
            border-radius: 8px;
            border-left: 4px solid {UI_CONFIG['primary_color']};
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        }}
        
        /* Custom animations */
        @keyframes pulse {{
            0% {{
                box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.4);
            }}
            70% {{
                box-shadow: 0 0 0 10px rgba(0, 212, 170, 0);
            }}
            100% {{
                box-shadow: 0 0 0 0 rgba(0, 212, 170, 0);
            }}
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .slide-in {{
            animation: slideIn 0.5s ease-out;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #161b22;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #30363d;
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {UI_CONFIG['primary_color']};
        }}
        
        /* Loading spinner */
        .stSpinner > div {{
            border-color: {UI_CONFIG['primary_color']} !important;
        }}
        
        /* Custom utility classes */
        .text-primary {{
            color: {UI_CONFIG['primary_color']} !important;
        }}
        
        .text-secondary {{
            color: {UI_CONFIG['secondary_color']} !important;
        }}
        
        .bg-gradient {{
            background: linear-gradient(135deg, {UI_CONFIG['primary_color']}, {UI_CONFIG['secondary_color']});
        }}
        
        .shadow-glow {{
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.3);
        }}
        
        .hover-scale {{
            transition: transform 0.3s ease;
        }}
        
        .hover-scale:hover {{
            transform: scale(1.05);
        }}
    </style>
    """
    
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Add custom JavaScript for enhanced interactions
    custom_js = """
    <script>
        // Add smooth scroll behavior
        document.documentElement.style.scrollBehavior = 'smooth';
        
        // Add intersection observer for animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('slide-in');
                }
            });
        });
        
        // Observe all metric cards
        document.querySelectorAll('.metric-card').forEach(el => {
            observer.observe(el);
        });
    </script>
    """
    
    st.markdown(custom_js, unsafe_allow_html=True)


def create_metric_card(title: str, value: str, delta: str = None, help_text: str = None, card_type: str = "default"):
    """Create a custom metric card with enhanced styling"""
    
    delta_html = ""
    if delta:
        delta_color = UI_CONFIG['success_color'] if delta.startswith('+') else UI_CONFIG['danger_color']
        delta_html = f'<p style="color: {delta_color}; margin: 0; font-size: 0.9rem;">{delta}</p>'
    
    help_html = ""
    if help_text:
        help_html = f'<span style="color: #888; font-size: 0.8rem;" title="{help_text}">ⓘ</span>'
    
    card_class = f"metric-card {card_type}"
    
    return f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; font-size: 1rem; color: #888;">{title}</h4>
            {help_html}
        </div>
        <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: 700;">{value}</h2>
        {delta_html}
    </div>
    """


def create_prediction_card(match: str, probability: float, odds: float, confidence: str):
    """Create a custom prediction card"""
    
    confidence_colors = {
        "High": UI_CONFIG['success_color'],
        "Medium": UI_CONFIG['warning_color'],
        "Low": UI_CONFIG['danger_color']
    }
    
    confidence_color = confidence_colors.get(confidence, UI_CONFIG['primary_color'])
    
    return f"""
    <div class="prediction-card hover-scale">
        <h3 style="margin: 0 0 1rem 0; font-size: 1.2rem;">{match}</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div>
                <p style="margin: 0; color: #888; font-size: 0.9rem;">Probability</p>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">{probability:.1%}</p>
            </div>
            <div>
                <p style="margin: 0; color: #888; font-size: 0.9rem;">Odds</p>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 600;">@{odds:.2f}</p>
            </div>
            <div>
                <p style="margin: 0; color: #888; font-size: 0.9rem;">Confidence</p>
                <p style="margin: 0; font-size: 1rem; font-weight: 600; color: {confidence_color};">{confidence}</p>
            </div>
        </div>
    </div>
    """