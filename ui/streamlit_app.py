# ui/streamlit_app.py
"""
Modern Streamlit interface with real-time updates and comprehensive monitoring
Features: Background task processing, live updates, Norwegian theme, analytics dashboard
"""

import pandas as pd
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import core extraction engine
from core.extractor_engine import get_extraction_engine, ExtractionStatus, shutdown_extraction_engine

import streamlit as st
def main():
    """Main Streamlit application with modern UI and comprehensive features"""
    
    # Configure page with Norwegian theme
    st.set_page_config(
        page_title="Norwegian Service Aggregator",
        page_icon="🇳🇴",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/AdsTable/AI-Ultimate.Service.Aggregator',
            'Report a bug': 'https://github.com/AdsTable/AI-Ultimate.Service.Aggregator/issues',
            'About': "Norwegian Service Aggregator - Real-time service plan extraction and comparison platform"
        }
    )
    
    # Apply modern CSS styling with Norwegian theme
    apply_modern_css()
    
    # Initialize application state and services
    initialize_app()
    
    # Render main interface components
    render_main_interface()


def apply_modern_css():
    """Apply comprehensive CSS styling with Norwegian flag theme and modern design"""
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Norwegian flag colors: Red (#EF2B2D), Blue (#002868), White (#FFFFFF) */
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header with gradient and Norwegian flag colors */
    .main-header {
        background: linear-gradient(135deg, #002868 0%, #EF2B2D 50%, #FFFFFF 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 40, 104, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.5rem;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .main-header .live-indicator {
        position: relative;
        z-index: 1;
    }
    
    /* Metric cards with modern glass effect */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(248,249,250,0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-left: 6px solid #002868;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(0,40,104,0.02) 0%, rgba(239,43,45,0.02) 100%);
        pointer-events: none;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.15);
        border-left-color: #EF2B2D;
    }
    
    /* Status badges with animations */
    .status-badge {
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.25rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .status-pending {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 1px solid rgba(255,234,167,0.5);
    }
    
    .status-running {
        background: linear-gradient(135deg, #cce5ff 0%, #99ccff 100%);
        color: #004085;
        border: 1px solid rgba(153,204,255,0.5);
        animation: pulse 2s infinite;
    }
    
    .status-completed {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid rgba(195,230,203,0.5);
    }
    
    .status-failed {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 1px solid rgba(245,198,203,0.5);
    }
    
    /* Enhanced animations */
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Extraction cards with enhanced styling */
    .extraction-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 4px solid #EF2B2D;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease;
    }
    
    .extraction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(239,43,45,0.02) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .extraction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    
    /* Progress container with modern styling */
    .progress-container {
        background: linear-gradient(145deg, #f1f3f4 0%, #e8eaed 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    /* Enhanced buttons with Norwegian theme */
    .stButton > button {
        background: linear-gradient(135deg, #002868 0%, #EF2B2D 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,40,104,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,40,104,0.4);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Sidebar enhancements */
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border: 1px solid rgba(0,40,104,0.1);
        transition: all 0.3s ease;
    }
    
    .sidebar .stSelectbox > div > div:hover {
        border-color: #002868;
        box-shadow: 0 4px 12px rgba(0,40,104,0.1);
    }
    
    /* Data table with modern styling */
    .data-table {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.8);
        margin: 1rem 0;
    }
    
    /* Real-time indicator with enhanced animation */
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 50%;
        animation: liveBlink 2s infinite;
        margin-right: 0.5rem;
        box-shadow: 0 0 12px rgba(40,167,69,0.5);
    }
    
    @keyframes liveBlink {
        0%, 50% { 
            opacity: 1; 
            transform: scale(1); 
            box-shadow: 0 0 12px rgba(40,167,69,0.5);
        }
        51%, 100% { 
            opacity: 0.4; 
            transform: scale(0.9); 
            box-shadow: 0 0 8px rgba(40,167,69,0.3);
        }
    }
    
    /* Toast notifications */
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        animation: slideInRight 0.4s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
    }
    
    @keyframes slideInRight {
        from { 
            transform: translateX(100%); 
            opacity: 0;
        }
        to { 
            transform: translateX(0); 
            opacity: 1;
        }
    }
    
    .toast.success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    .toast.info {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
    }
    
    .toast.warning {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
    }
    
    .toast.error {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    }
    
    /* Chart containers */
    .chart-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.8);
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,40,104,0.3);
        border-radius: 50%;
        border-top-color: #002868;
        animation: spin 1s ease-in-out infinite;
        margin-right: 0.5rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .main-header p {
            font-size: 1.2rem;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
        
        .extraction-card {
            padding: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #002868 0%, #EF2B2D 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #001a4d 0%, #d12529 100%);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_app():
    """Initialize application state, services, and session management"""
    
    # Initialize session state with comprehensive configuration
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.active_tasks = {}
        st.session_state.last_refresh = time.time()
        st.session_state.auto_refresh = True
        st.session_state.refresh_interval = 5  # seconds
        st.session_state.notifications = []
        st.session_state.dark_mode = False
        st.session_state.language = 'en'  # 'en' or 'no'
        st.session_state.last_stats_update = 0
        st.session_state.cached_stats = {}
    
    # Initialize extraction engine with error handling
    if 'engine' not in st.session_state:
        with st.spinner("🔧 Initializing Norwegian Service Extraction Engine..."):
            try:
                st.session_state.engine = get_extraction_engine()
                show_toast("✅ Extraction engine initialized successfully", "success")
                
                # Log initialization
                logger.info("Streamlit app initialized successfully")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize extraction engine: {str(e)}")
                st.session_state.engine = None
                show_toast(f"❌ Engine initialization failed: {str(e)}", "error")
                return
    
    # Setup automatic refresh for real-time updates
    if st.session_state.auto_refresh:
        setup_auto_refresh()
    
    # Cleanup old notifications
    cleanup_old_notifications()


def setup_auto_refresh():
    """Setup automatic page refresh for real-time updates with intelligent timing"""
    current_time = time.time()
    
    # Check if it's time to refresh
    time_since_refresh = current_time - st.session_state.last_refresh
    
    # Dynamic refresh interval based on activity
    active_tasks = len(st.session_state.active_tasks)
    if active_tasks > 0:
        # More frequent updates when tasks are active
        refresh_interval = max(2, st.session_state.refresh_interval // 2)
    else:
        refresh_interval = st.session_state.refresh_interval
    
    if time_since_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()


def cleanup_old_notifications():
    """Remove old notifications to prevent memory buildup"""
    current_time = time.time()
    st.session_state.notifications = [
        notif for notif in st.session_state.notifications 
        if current_time - notif.get('timestamp', 0) < 300  # Keep for 5 minutes
    ]


def show_toast(message: str, toast_type: str = "info"):
    """Add toast notification to session state for display"""
    notification = {
        'message': message,
        'type': toast_type,
        'timestamp': time.time(),
        'id': len(st.session_state.notifications)
    }
    st.session_state.notifications.append(notification)


def render_main_interface():
    """Render the main application interface with all components"""
    
    # Main header with Norwegian branding
    render_main_header()
    
    # Check engine availability
    if st.session_state.engine is None:
        render_error_state()
        return
    
    # Render main content areas
    render_control_panel()
    render_real_time_dashboard()
    render_data_explorer()
    render_analytics_dashboard()
    
    # Render toast notifications
    render_toast_notifications()


def render_main_header():
    """Render main header with Norwegian theme and real-time indicators"""
    st.markdown("""
    <div class="main-header">
        <h1>🇳🇴 Norwegian Service Aggregator</h1>
        <p>Real-time Service Plan Extraction & Comparison Platform</p>
        <div style="margin-top: 1rem;">
            <span class="live-indicator"></span>
            <span style="font-size: 1rem; opacity: 0.9; font-weight: 500;">Live System Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_error_state():
    """Render error state when engine is not available"""
    st.error("🚫 Extraction engine not available")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Retry Initialization", type="primary", use_container_width=True):
            # Clear engine from session state to trigger re-initialization
            if 'engine' in st.session_state:
                del st.session_state.engine
            st.rerun()
        
        st.info("""
        **Troubleshooting Steps:**
        1. Check if all dependencies are installed
        2. Ensure database permissions are correct
        3. Verify network connectivity
        4. Check system resources (CPU, memory)
        """)


def render_control_panel():
    """Render comprehensive extraction control panel with Norwegian providers"""
    st.sidebar.header("🎛️ Extraction Control Panel")
    
    # Real-time settings section
    render_realtime_settings()
    
    st.sidebar.divider()
    
    # Norwegian providers section
    render_norwegian_providers_section()
    
    st.sidebar.divider()
    
    # Custom URL section
    render_custom_url_section()
    
    st.sidebar.divider()
    
    # Bulk operations section
    render_bulk_operations_section()
    
    st.sidebar.divider()
    
    # Active tasks monitoring
    render_active_tasks_sidebar()
    
    st.sidebar.divider()
    
    # System status section
    render_system_status_sidebar()


def render_realtime_settings():
    """Render real-time settings and configuration options"""
    st.sidebar.subheader("⚡ Real-time Settings")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox(
        "Auto-refresh Dashboard", 
        value=st.session_state.auto_refresh,
        help="Automatically refresh the interface to show live updates"
    )
    
    # Refresh interval slider
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.sidebar.slider(
            "Refresh interval (seconds)", 
            min_value=2, 
            max_value=30, 
            value=st.session_state.refresh_interval,
            help="How often to refresh the dashboard for live updates"
        )
    
    # Language selection
    language_options = {"English": "en", "Norsk": "no"}
    selected_lang = st.sidebar.selectbox(
        "Language / Språk",
        options=list(language_options.keys()),
        index=0 if st.session_state.language == "en" else 1
    )
    st.session_state.language = language_options[selected_lang]


def render_norwegian_providers_section():
    """Render Norwegian providers selection with enhanced options"""
    st.sidebar.subheader("🇳🇴 Norwegian Service Providers")
    
    # Comprehensive list of Norwegian providers
    norwegian_providers = {
        "🏢 Telia Mobile Plans": "https://www.telia.no/privat/mobil/abonnement",
        "📱 Telenor Mobile Services": "https://www.telenor.no/privat/mobil/abonnement/",
        "❄️ Ice Mobile Packages": "https://www.ice.no/mobil/abonnement/",
        "💬 Talkmore Subscriptions": "https://talkmore.no/abonnement/",
        "⚡ Fortum Electricity Plans": "https://www.fortum.no/privat/strom/stromavtaler",
        "🔌 Hafslund Power Contracts": "https://www.hafslund.no/strom/stromavtale/",
        "🌊 Lyse Energy Solutions": "https://www.lyse.no/strom/stromavtale/",
        "🏠 Fjordkraft Home Energy": "https://www.fjordkraft.no/strom/"
    }
    
    # Provider category filter
    categories = {
        "All Categories": norwegian_providers,
        "Mobile Services": {k: v for k, v in norwegian_providers.items() if "Mobile" in k or "Talkmore" in k},
        "Electricity": {k: v for k, v in norwegian_providers.items() if "Electric" in k or "Power" in k or "Energy" in k}
    }
    
    selected_category = st.sidebar.selectbox(
        "Provider Category",
        options=list(categories.keys()),
        help="Filter providers by service category"
    )
    
    filtered_providers = categories[selected_category]
    
    # Provider selection
    selected_provider = st.sidebar.selectbox(
        "Select Provider",
        list(filtered_providers.keys()),
        help="Choose a Norwegian service provider for background extraction"
    )
    
    # Extraction button with enhanced feedback
    if st.sidebar.button("🚀 Start Background Extraction", type="primary"):
        url = filtered_providers[selected_provider]
        provider_name = selected_provider.split(" ", 1)[1]  # Remove emoji
        start_background_extraction(url, provider_name)


def render_custom_url_section():
    """Render custom URL extraction with validation"""
    st.sidebar.subheader("🌐 Custom Provider Extraction")
    
    # URL input with validation
    custom_url = st.sidebar.text_input(
        "Norwegian Provider URL",
        placeholder="https://example.no/pricing",
        help="Enter any Norwegian service provider URL for extraction"
    )
    
    # URL validation feedback
    if custom_url:
        if validate_url(custom_url):
            st.sidebar.success("✅ Valid URL format")
        else:
            st.sidebar.error("❌ Invalid URL format")
    
    # Provider name input for custom URLs
    custom_provider_name = st.sidebar.text_input(
        "Provider Name (optional)",
        placeholder="Enter provider name",
        help="Optional: Provide a custom name for this provider"
    )
    
    # Extract button
    if st.sidebar.button("🔍 Extract from Custom URL"):
        if custom_url and validate_url(custom_url):
            provider_name = custom_provider_name if custom_provider_name else "Custom Provider"
            start_background_extraction(custom_url, provider_name)
        else:
            st.sidebar.error("❌ Please enter a valid URL")
            show_toast("Please enter a valid URL starting with http:// or https://", "error")


def validate_url(url: str) -> bool:
    """Validate URL format and Norwegian domain preference"""
    if not url:
        return False
    
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return bool(parsed.netloc)
    except Exception:
        return False


def render_bulk_operations_section():
    """Render bulk operations with progress tracking"""
    st.sidebar.subheader("🔄 Bulk Operations")
    
    # Bulk extraction options
    bulk_options = {
        "All Mobile Providers": [
            ("Telia Mobile", "https://www.telia.no/mobilabonnement"),
            ("Telenor Mobile", "https://www.telenor.no/mobilabonnement/"),
            ("Ice Mobile", "https://www.ice.no/mobilabonnement/"),
            ("Talkmore", "https://talkmore.no/privat/abonnement/enkelt/bestill")
        ],
        "All Electricity Providers": [
            ("Fortum", "https://www.fortum.com/no/strom/stromavtale"),
            ("Forbrukerradet", "https://www.forbrukerradet.no/strompris/"),
            ("Lyse", "https://www.lyse.no/strom")
        ],
        "All Providers": [
            ("Telia Mobile", "https://www.telia.no/mobilabonnement"),
            ("Telenor Mobile", "https://www.telenor.no/mobilabonnement/"),
            ("Ice Mobile", "https://www.ice.no/mobilabonnement/"),
            ("Talkmore", "https://talkmore.no/privat/abonnement/enkelt/bestill"),
            ("Fortum", "https://www.fortum.com/no/strom/stromavtale"),
            ("Forbrukerradet", "https://www.forbrukerradet.no/strompris/")
        ]
    }
    
    selected_bulk_option = st.sidebar.selectbox(
        "Bulk Extraction Set",
        options=list(bulk_options.keys()),
        help="Select a set of providers for bulk extraction"
    )
    
    providers_to_extract = bulk_options[selected_bulk_option]
    
    if st.sidebar.button(f"⚡ Extract {selected_bulk_option}", 
                        help=f"Start background extraction from {len(providers_to_extract)} providers"):
        start_bulk_extraction(providers_to_extract)


def render_active_tasks_sidebar():
    """Render active tasks monitoring with enhanced status display and real-time updates"""
    if not st.session_state.active_tasks:
        return
    
    st.sidebar.subheader("📊 Active Extraction Tasks")
    
    # Get current task statuses and organize by status
    task_statuses = {"running": [], "pending": [], "completed": [], "failed": []}
    completed_tasks = []
    
    for task_id, task_info in list(st.session_state.active_tasks.items()):
        current_task = st.session_state.engine.get_task_status(task_id)
        
        if current_task:
            status = current_task.status.value
            
            # Create status display with enhanced information
            if current_task.status == ExtractionStatus.RUNNING:
                progress_text = f" ({current_task.progress:.0%})"
                status_emoji = "🔄"
            elif current_task.status == ExtractionStatus.COMPLETED:
                progress_text = f" ({len(current_task.result)} plans)"
                status_emoji = "✅"
                completed_tasks.append(task_id)
            elif current_task.status == ExtractionStatus.FAILED:
                progress_text = " (failed)"
                status_emoji = "❌"
                completed_tasks.append(task_id)
            elif current_task.status == ExtractionStatus.PENDING:
                progress_text = " (queued)"
                status_emoji = "⏳"
            else:
                progress_text = ""
                status_emoji = "❓"
            
            # Display task status with styling
            status_class = f"status-{current_task.status.value}"
            
            st.sidebar.markdown(f"""
            <div class="status-badge {status_class}">
                {status_emoji} {task_info['provider_name']}: {current_task.status.value.title()}{progress_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Show additional details for running tasks
            if current_task.status == ExtractionStatus.RUNNING:
                st.sidebar.progress(current_task.progress, text=current_task.message)
            
        else:
            # Task not found in engine, probably completed
            completed_tasks.append(task_id)
    
    # Clean up completed tasks after showing them for a while
    current_time = datetime.now()
    for task_id in completed_tasks:
        task_info = st.session_state.active_tasks.get(task_id)
        if task_info and (current_time - task_info['started_at']).seconds > 45:
            del st.session_state.active_tasks[task_id]


def render_system_status_sidebar():
    """Render system health status and performance metrics in sidebar"""
    st.sidebar.subheader("🔧 System Status")
    
    try:
        # Get health status from engine
        health_status = st.session_state.engine.get_health_status()
        
        # Overall system status
        status = health_status.get('status', 'unknown')
        status_colors = {
            'healthy': '🟢',
            'degraded': '🟡', 
            'unhealthy': '🔴',
            'unknown': '⚪'
        }
        
        st.sidebar.markdown(f"""
        **Overall Status:** {status_colors.get(status, '⚪')} {status.title()}
        """)
        
        # Component status
        components = health_status.get('components', {})
        for component, comp_status in components.items():
            emoji = status_colors.get(comp_status, '⚪')
            st.sidebar.text(f"{emoji} {component.title()}: {comp_status}")
        
        # Key metrics
        metrics = health_status.get('metrics', {})
        if metrics:
            st.sidebar.markdown("**Metrics:**")
            st.sidebar.text(f"Active Tasks: {metrics.get('active_tasks', 0)}")
            st.sidebar.text(f"Queue Size: {metrics.get('queue_size', 0)}")
            st.sidebar.text(f"Total Plans: {metrics.get('total_plans', 0)}")
        
    except Exception as e:
        st.sidebar.error(f"Status check failed: {str(e)}")


def start_background_extraction(url: str, provider_name: str):
    """Start background extraction task with comprehensive error handling and user feedback"""
    try:
        # Validate inputs
        if not url or not provider_name:
            raise ValueError("URL and provider name are required")
        
        # Submit task to extraction engine
        task_id = st.session_state.engine.submit_extraction_task(url, provider_name)
        
        # Store task in session state for tracking
        st.session_state.active_tasks[task_id] = {
            'provider_name': provider_name,
            'url': url,
            'started_at': datetime.now(),
            'task_id': task_id
        }
        
        # Success feedback
        st.sidebar.success(f"✅ Extraction started for {provider_name}")
        st.sidebar.info(f"📋 Task ID: {task_id[:8]}...")
        
        # Show toast notification
        show_toast(f"🚀 Background extraction started for {provider_name}", "success")
        
        # Log the action
        logger.info(f"User started extraction for {provider_name} (URL: {url})")
        
    except Exception as e:
        error_msg = str(e)
        st.sidebar.error(f"❌ Failed to start extraction: {error_msg}")
        show_toast(f"❌ Extraction failed to start: {error_msg}", "error")
        logger.error(f"Failed to start extraction for {provider_name}: {e}")


def start_bulk_extraction(providers: List[tuple]):
    """Start bulk background extraction from multiple providers with progress tracking"""
    if not providers:
        st.sidebar.error("❌ No providers selected for bulk extraction")
        return
    
    task_ids = []
    successful_starts = 0
    
    # Progress indicator for bulk operation
    progress_placeholder = st.sidebar.empty()
    
    for i, (provider_name, url) in enumerate(providers):
        try:
            # Update progress
            progress = (i + 1) / len(providers)
            progress_placeholder.progress(progress, text=f"Starting {provider_name}...")
            
            # Submit extraction task
            task_id = st.session_state.engine.submit_extraction_task(url, provider_name)
            task_ids.append(task_id)
            
            # Store task in session state
            st.session_state.active_tasks[task_id] = {
                'provider_name': provider_name,
                'url': url,
                'started_at': datetime.now(),
                'task_id': task_id,
                'bulk_operation': True
            }
            
            successful_starts += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to start {provider_name}: {str(e)}")
            logger.error(f"Bulk extraction failed for {provider_name}: {e}")
    
    # Clear progress indicator
    progress_placeholder.empty()
    
    # Summary feedback
    if successful_starts > 0:
        st.sidebar.success(f"✅ Started {successful_starts}/{len(providers)} extraction tasks")
        show_toast(f"🔄 Bulk extraction started: {successful_starts} tasks", "info")
        logger.info(f"Bulk extraction started for {successful_starts} providers")
    else:
        st.sidebar.error("❌ No extraction tasks could be started")
        show_toast("❌ Bulk extraction failed to start any tasks", "error")


def render_real_time_dashboard():
    """Render comprehensive real-time dashboard with live metrics and monitoring"""
    
    # Get current statistics with caching for performance
    try:
        current_time = time.time()
        if (current_time - st.session_state.last_stats_update > 10 or 
            not st.session_state.cached_stats):
            st.session_state.cached_stats = st.session_state.engine.get_statistics()
            st.session_state.last_stats_update = current_time
        
        stats = st.session_state.cached_stats
        
    except Exception as e:
        st.error(f"❌ Failed to load statistics: {str(e)}")
        return
    
    # Main metrics overview with enhanced styling
    st.subheader("📊 Live System Dashboard")
    
    # Primary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_value = f"+{stats.get('recent_extractions', 0)}" if stats.get('recent_extractions', 0) > 0 else None
        st.metric(
            label="📋 Total Service Plans",
            value=f"{stats.get('total_plans', 0):,}",
            delta=delta_value,
            help="Total number of service plans extracted and stored in database"
        )
    
    with col2:
        provider_count = len(stats.get('providers', {}))
        st.metric(
            label="🏢 Active Providers",
            value=provider_count,
            help="Number of different service providers with extracted data"
        )
    
    with col3:
        category_count = len(stats.get('categories', {}))
        st.metric(
            label="📂 Service Categories",
            value=category_count,
            help="Types of services (mobile, electricity, internet, etc.)"
        )
    
    with col4:
        active_tasks = stats.get('active_tasks_count', 0)
        queue_size = stats.get('queue_size', 0)
        total_active = active_tasks + queue_size
        
        st.metric(
            label="⚡ Active Operations",
            value=total_active,
            delta=f"Queue: {queue_size}" if queue_size > 0 else None,
            help="Currently running extraction tasks and queued operations"
        )
    
    # Secondary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quality_metrics = stats.get('quality_metrics', {})
        avg_confidence = quality_metrics.get('avg_confidence', 0)
        st.metric(
            label="🎯 Average Confidence",
            value=f"{avg_confidence:.2%}",
            help="Average confidence score of extracted service plans"
        )
    
    with col2:
        plans_with_price = quality_metrics.get('plans_with_price', 0)
        total_plans = stats.get('total_plans', 1)  # Avoid division by zero
        price_coverage = plans_with_price / max(total_plans, 1)
        st.metric(
            label="💰 Price Coverage",
            value=f"{price_coverage:.1%}",
            help="Percentage of plans with successfully extracted pricing information"
        )
    
    with col3:
        recent_extractions = stats.get('recent_extractions', 0)
        st.metric(
            label="📈 Today's Extractions",
            value=recent_extractions,
            help="Number of plans extracted in the last 24 hours"
        )
    
    with col4:
        system_status = stats.get('system_status', 'unknown')
        status_display = {
            'healthy': '🟢 Healthy',
            'busy': '🟡 Busy', 
            'degraded': '🟠 Degraded',
            'unhealthy': '🔴 Error'
        }.get(system_status, '⚪ Unknown')
        
        st.metric(
            label="🔧 System Status",
            value=status_display,
            help="Overall system health and performance status"
        )
    
    # Real-time task monitoring section
    if st.session_state.active_tasks:
        render_real_time_task_monitor()
    
    # Performance and health charts
    if stats.get('total_plans', 0) > 0:
        render_performance_charts(stats)


def render_real_time_task_monitor():
    """Render real-time task monitoring with progress visualization"""
    st.subheader("⚡ Real-time Task Monitor")
    
    # Create responsive columns based on number of active tasks
    num_tasks = len(st.session_state.active_tasks)
    if num_tasks == 0:
        return
    
    # Limit columns to 3 for better layout
    num_columns = min(num_tasks, 3)
    task_cols = st.columns(num_columns)
    
    # Sort tasks by status (running first, then pending, then completed)
    status_priority = {
        ExtractionStatus.RUNNING: 1,
        ExtractionStatus.PENDING: 2,
        ExtractionStatus.COMPLETED: 3,
        ExtractionStatus.FAILED: 4
    }
    
    sorted_tasks = []
    for task_id, task_info in st.session_state.active_tasks.items():
        current_task = st.session_state.engine.get_task_status(task_id)
        if current_task:
            priority = status_priority.get(current_task.status, 5)
            sorted_tasks.append((priority, task_id, task_info, current_task))
    
    sorted_tasks.sort(key=lambda x: x[0])
    
    # Display tasks in columns
    for i, (_, task_id, task_info, current_task) in enumerate(sorted_tasks):
        col = task_cols[i % num_columns]
        
        with col:
            # Task status card with enhanced information
            status_emoji = {
                ExtractionStatus.PENDING: "⏳",
                ExtractionStatus.RUNNING: "🔄", 
                ExtractionStatus.COMPLETED: "✅",
                ExtractionStatus.FAILED: "❌"
            }.get(current_task.status, "❓")
            
            # Calculate elapsed time
            elapsed_time = datetime.now() - task_info['started_at']
            elapsed_str = f"{elapsed_time.seconds}s"
            
            st.markdown(f"""
            <div class="extraction-card">
                <h4>{status_emoji} {task_info['provider_name']}</h4>
                <p><strong>Status:</strong> {current_task.status.value.title()}</p>
                <p><strong>Message:</strong> {current_task.message}</p>
                <p><strong>Elapsed:</strong> {elapsed_str}</p>
                <p><strong>Task ID:</strong> {task_id[:8]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress visualization
            if current_task.status == ExtractionStatus.RUNNING:
                progress = current_task.progress
                st.progress(progress, text=f"Progress: {progress:.0%}")
                
                # Estimated time remaining
                if progress > 0.1:
                    elapsed_seconds = elapsed_time.total_seconds()
                    estimated_total = elapsed_seconds / progress
                    remaining = max(0, estimated_total - elapsed_seconds)
                    if remaining > 0:
                        st.caption(f"Est. remaining: {remaining:.0f}s")
                        
            elif current_task.status == ExtractionStatus.COMPLETED:
                result_count = len(current_task.result)
                st.success(f"✅ Completed: {result_count} plans extracted")
                
                # Show extracted plan names if available
                if current_task.result:
                    with st.expander("📋 Extracted Plans", expanded=False):
                        for plan in current_task.result[:5]:  # Show first 5
                            st.write(f"• {plan.name} - {plan.monthly_price} NOK")
                        if len(current_task.result) > 5:
                            st.write(f"... and {len(current_task.result) - 5} more")
                            
            elif current_task.status == ExtractionStatus.FAILED:
                st.error(f"❌ Failed: {current_task.error}")
                
                # Retry button for failed tasks
                if st.button(f"🔄 Retry", key=f"retry_{task_id}"):
                    retry_failed_task(task_id, task_info)


def retry_failed_task(task_id: str, task_info: dict):
    """Retry a failed extraction task"""
    try:
        # Remove old task from active tasks
        if task_id in st.session_state.active_tasks:
            del st.session_state.active_tasks[task_id]
        
        # Start new extraction
        start_background_extraction(task_info['url'], task_info['provider_name'])
        show_toast(f"🔄 Retrying extraction for {task_info['provider_name']}", "info")
        
    except Exception as e:
        st.error(f"❌ Failed to retry task: {str(e)}")
        show_toast(f"❌ Retry failed: {str(e)}", "error")


def render_performance_charts(stats: Dict[str, Any]):
    """Render comprehensive performance and analytics charts"""
    st.subheader("📈 Performance Analytics")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        render_provider_performance_chart(stats)
    
    with col2:
        render_category_distribution_chart(stats)
    
    # Full-width charts
    render_extraction_timeline_chart(stats)
    render_quality_metrics_chart(stats)


def render_provider_performance_chart(stats: Dict[str, Any]):
    """Render provider performance chart with enhanced metrics"""
    st.write("**📊 Provider Performance**")
    
    providers = stats.get('providers', {})
    if not providers:
        st.info("No provider data available yet")
        return
    
    # Prepare data for visualization
    provider_data = []
    for provider, metrics in providers.items():
        provider_data.append({
            'Provider': provider.title(),
            'Plans': metrics.get('count', 0),
            'Avg Confidence': metrics.get('avg_confidence', 0),
            'Quality Score': metrics.get('avg_validation', 0)
        })
    
    # Create enhanced bar chart
    df = pd.DataFrame(provider_data)
    
    if not df.empty:
        fig = px.bar(
            df, 
            x='Provider', 
            y='Plans',
            color='Avg Confidence',
            hover_data=['Quality Score'],
            title="Plans Extracted by Provider",
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Service Provider",
            yaxis_title="Number of Plans"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No provider performance data available")


def render_category_distribution_chart(stats: Dict[str, Any]):
    """Render service category distribution with pricing information"""
    st.write("**📱 Service Categories**")
    
    categories = stats.get('categories', {})
    if not categories:
        st.info("No category data available yet")
        return
    
    # Prepare data for pie chart
    category_data = []
    total_plans = 0
    
    for category, metrics in categories.items():
        count = metrics.get('count', 0)
        avg_price = metrics.get('avg_price', 0)
        
        category_data.append({
            'Category': category.title(),
            'Plans': count,
            'Avg Price (NOK)': avg_price
        })
        total_plans += count
    
    if category_data:
        df = pd.DataFrame(category_data)
        
        # Create pie chart with pricing information
        fig = px.pie(
            df, 
            values='Plans', 
            names='Category',
            title="Distribution by Service Category",
            hover_data=['Avg Price (NOK)']
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Plans: %{value}<br>Avg Price: %{customdata[0]:.0f} NOK<extra></extra>'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category distribution data available")


def render_extraction_timeline_chart(stats: Dict[str, Any]):
    """Render extraction timeline showing activity over time"""
    st.write("**⏱️ Extraction Activity Timeline**")
    
    try:
        # Get plans data for timeline analysis
        all_plans = st.session_state.engine.get_all_plans()
        
        if not all_plans:
            st.info("No extraction timeline data available yet")
            return
        
        # Process extraction times
        df = pd.DataFrame(all_plans)
        df['extracted_at'] = pd.to_datetime(df['extracted_at'])
        df['date'] = df['extracted_at'].dt.date
        
        # Group by date and count extractions
        daily_extractions = df.groupby('date').agg({
            'id': 'count',
            'confidence': 'mean',
            'monthly_price': lambda x: (x > 0).sum()
        }).reset_index()
        
        daily_extractions.columns = ['Date', 'Plans Extracted', 'Avg Confidence', 'Plans with Price']
        
        # Create timeline chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Extraction Volume', 'Quality Metrics'),
            vertical_spacing=0.1
        )
        
        # Volume chart
        fig.add_trace(
            go.Scatter(
                x=daily_extractions['Date'],
                y=daily_extractions['Plans Extracted'],
                mode='lines+markers',
                name='Plans Extracted',
                line=dict(color='#002868', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Quality metrics chart
        fig.add_trace(
            go.Scatter(
                x=daily_extractions['Date'],
                y=daily_extractions['Avg Confidence'],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='#EF2B2D', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text="Extraction Activity and Quality Over Time"
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Plans Extracted", row=1, col=1)
        fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to render timeline chart: {str(e)}")


def render_quality_metrics_chart(stats: Dict[str, Any]):
    """Render comprehensive quality metrics dashboard"""
    st.write("**🎯 Data Quality Metrics**")
    
    quality_metrics = stats.get('quality_metrics', {})
    if not quality_metrics:
        st.info("No quality metrics available yet")
        return
    
    # Quality metrics visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_confidence = quality_metrics.get('avg_confidence', 0)
        st.metric(
            "Average Confidence",
            f"{avg_confidence:.1%}",
            help="Overall confidence in extracted data accuracy"
        )
    
    with col2:
        avg_validation = quality_metrics.get('avg_validation', 0)
        st.metric(
            "Validation Score", 
            f"{avg_validation:.1%}",
            help="Data completeness and structure validation score"
        )
    
    with col3:
        high_confidence_count = quality_metrics.get('high_confidence_count', 0)
        total_plans = stats.get('total_plans', 1)
        high_confidence_rate = high_confidence_count / max(total_plans, 1)
        st.metric(
            "High Quality Rate",
            f"{high_confidence_rate:.1%}",
            help="Percentage of plans with confidence score ≥ 80%"
        )


def render_data_explorer():
    """Render comprehensive data exploration interface with advanced filtering"""
    st.subheader("🔍 Service Plans Data Explorer")
    
    # Get all plans data
    try:
        all_plans = st.session_state.engine.get_all_plans()
    except Exception as e:
        st.error(f"❌ Failed to load plans data: {str(e)}")
        return
    
    if not all_plans:
        render_empty_state()
        return
    
    # Advanced filtering interface
    filtered_plans = render_advanced_filtering_interface(all_plans)
    
    # Display results
    if filtered_plans:
        render_plans_data_table(filtered_plans)
        render_export_options(filtered_plans)
    else:
        st.warning("⚠️ No plans match your current filter criteria. Try adjusting the filters above.")


def render_advanced_filtering_interface(all_plans: List[Dict]) -> List[Dict]:
    """Render advanced filtering interface with multiple criteria"""
    st.write("**🎛️ Advanced Filters**")
    
    # Create filter columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Provider filter
        providers = ['All Providers'] + sorted(set(plan['provider'] for plan in all_plans))
        selected_provider = st.selectbox(
            "Provider",
            providers,
            help="Filter by service provider"
        )
    
    with col2:
        # Category filter  
        categories = ['All Categories'] + sorted(set(plan['category'] for plan in all_plans))
        selected_category = st.selectbox(
            "Category", 
            categories,
            help="Filter by service category"
        )
    
    with col3:
        # Price range filter
        prices = [plan['monthly_price'] for plan in all_plans if plan['monthly_price'] > 0]
        if prices:
            min_price, max_price = min(prices), max(prices)
            price_range = st.slider(
                "Price Range (NOK)",
                min_value=float(min_price),
                max_value=float(max_price), 
                value=(float(min_price), float(max_price)),
                help="Filter by monthly price range"
            )
        else:
            price_range = (0.0, 1000.0)
            st.slider("Price Range (NOK)", 0.0, 1000.0, (0.0, 1000.0), disabled=True)
    
    with col4:
        # Confidence filter
        confidence_threshold = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filter by minimum confidence score"
        )
    
    # Additional filters row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Search in plan names
        search_term = st.text_input(
            "Search Plan Names",
            placeholder="Type to search...",
            help="Search for specific terms in plan names"
        )
    
    with col2:
        # Extraction method filter
        methods = ['All Methods'] + sorted(set(plan.get('extraction_method', 'unknown') for plan in all_plans))
        selected_method = st.selectbox(
            "Extraction Method",
            methods,
            help="Filter by data extraction method"
        )
    
    with col3:
        # Date range filter
        if all_plans:
            dates = [datetime.fromisoformat(plan['extracted_at']).date() for plan in all_plans]
            min_date, max_date = min(dates), max(dates)
            
            date_range = st.date_input(
                "Extraction Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter by extraction date range"
            )
        else:
            date_range = None
    
    with col4:
        # Has pricing filter
        has_pricing_options = ['All Plans', 'With Pricing', 'Without Pricing']
        pricing_filter = st.selectbox(
            "Pricing Information",
            has_pricing_options,
            help="Filter by availability of pricing information"
        )
    
    # Apply all filters
    filtered_plans = apply_comprehensive_filters(
        all_plans, selected_provider, selected_category, price_range,
        confidence_threshold, search_term, selected_method, date_range, pricing_filter
    )
    
    # Show filter results summary
    st.info(f"🎯 Found {len(filtered_plans)} plans matching your criteria (from {len(all_plans)} total)")
    
    return filtered_plans


def apply_comprehensive_filters(plans: List[Dict], provider: str, category: str, 
                               price_range: tuple, confidence_threshold: float,
                               search_term: str, method: str, date_range, pricing_filter: str) -> List[Dict]:
    """Apply comprehensive filtering with multiple criteria"""
    filtered = plans.copy()
    
    # Provider filter
    if provider != 'All Providers':
        filtered = [p for p in filtered if p['provider'] == provider]
    
    # Category filter
    if category != 'All Categories':
        filtered = [p for p in filtered if p['category'] == category]
    
    # Price range filter
    if price_range:
        filtered = [p for p in filtered 
                   if price_range[0] <= p['monthly_price'] <= price_range[1]]
    
    # Confidence threshold filter
    filtered = [p for p in filtered if p.get('confidence', 0) >= confidence_threshold]
    
    # Search term filter
    if search_term:
        search_lower = search_term.lower()
        filtered = [p for p in filtered 
                   if search_lower in p['name'].lower() or 
                      any(search_lower in feature.lower() for feature in p.get('features', []))]
    
    # Extraction method filter
    if method != 'All Methods':
        filtered = [p for p in filtered if p.get('extraction_method', 'unknown') == method]
    
    # Date range filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = [p for p in filtered 
                   if start_date <= datetime.fromisoformat(p['extracted_at']).date() <= end_date]
    
    # Pricing filter
    if pricing_filter == 'With Pricing':
        filtered = [p for p in filtered if p['monthly_price'] > 0]
    elif pricing_filter == 'Without Pricing':
        filtered = [p for p in filtered if p['monthly_price'] <= 0]
    
    return filtered


def render_plans_data_table(filtered_plans: List[Dict]):
    """Render comprehensive data table with enhanced formatting and interactions"""
    st.write("**📋 Service Plans Results**")
    
    # Prepare data for display
    display_data = []
    for plan in filtered_plans:
        # Format features for display
        features = plan.get('features', [])
        features_display = ', '.join(features[:3])
        if len(features) > 3:
            features_display += f' (+{len(features) - 3} more)'
        
        # Format extraction date
        try:
            extracted_date = datetime.fromisoformat(plan['extracted_at'])
            formatted_date = extracted_date.strftime('%Y-%m-%d %H:%M')
        except:
            formatted_date = plan.get('extracted_at', 'Unknown')
        
        display_data.append({
            'Provider': plan['provider'],
            'Plan Name': plan['name'],
            'Category': plan['category'].title(),
            'Monthly Price (NOK)': plan['monthly_price'],
            'Features': features_display,
            'Confidence': plan.get('confidence', 0),
            'Validation Score': plan.get('validation_score', 0),
            'Extraction Method': plan.get('extraction_method', 'unknown').title(),
            'Extracted At': formatted_date,
            'Plan ID': plan['id']
        })
    
    # Create DataFrame
    df = pd.DataFrame(display_data)
    
    # Configure column display
    column_config = {
        "Monthly Price (NOK)": st.column_config.NumberColumn(
            "Monthly Price (NOK)",
            format="%.0f kr",
            help="Monthly subscription price in Norwegian Kroner"
        ),
        "Confidence": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0,
            max_value=1,
            format="%.1%",
            help="Confidence score of the extracted data"
        ),
        "Validation Score": st.column_config.ProgressColumn(
            "Validation Score", 
            min_value=0,
            max_value=1,
            format="%.1%",
            help="Data completeness and validation score"
        ),
        "Plan ID": st.column_config.TextColumn(
            "Plan ID",
            help="Unique identifier for the service plan",
            width="small"
        )
    }
    
    # Display interactive dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=400
    )
    
    # Plan details expander
    if len(filtered_plans) > 0:
        with st.expander("📝 Detailed Plan Information", expanded=False):
            # Plan selection for detailed view
            plan_names = [f"{plan['provider']} - {plan['name']}" for plan in filtered_plans]
            selected_plan_index = st.selectbox(
                "Select plan for detailed view:",
                range(len(plan_names)),
                format_func=lambda i: plan_names[i]
            )
            
            if selected_plan_index is not None:
                render_detailed_plan_view(filtered_plans[selected_plan_index])


def render_detailed_plan_view(plan: Dict[str, Any]):
    """Render detailed view of a single service plan"""
    st.write(f"### 📋 {plan['name']} - {plan['provider']}")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Basic Information**")
        st.write(f"Provider: {plan['provider']}")
        st.write(f"Category: {plan['category'].title()}")
        st.write(f"Monthly Price: {plan['monthly_price']} NOK")
    
    with col2:
        st.write("**Quality Metrics**")
        st.write(f"Confidence: {plan.get('confidence', 0):.1%}")
        st.write(f"Validation Score: {plan.get('validation_score', 0):.1%}")
        st.write(f"Extraction Method: {plan.get('extraction_method', 'unknown').title()}")
    
    with col3:
        st.write("**Metadata**")
        extracted_date = datetime.fromisoformat(plan['extracted_at']).strftime('%Y-%m-%d %H:%M:%S')
        st.write(f"Extracted: {extracted_date}")
        st.write(f"Data Source: {plan.get('data_source', 'web').title()}")
        st.write(f"Plan ID: {plan['id']}")
    
    # Features section
    features = plan.get('features', [])
    if features:
        st.write("**📋 Plan Features**")
        for i, feature in enumerate(features, 1):
            st.write(f"{i}. {feature}")
    else:
        st.write("**📋 Plan Features**: No features extracted")
    
    # Raw data section
    with st.expander("🔧 Raw Data (JSON)", expanded=False):
        st.json(plan)


def render_export_options(filtered_plans: List[Dict]):
    """Render comprehensive data export options with multiple formats"""
    st.write("**📤 Export Options**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CSV export
        csv_df = pd.DataFrame(filtered_plans)
        csv_data = csv_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"norwegian_service_plans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download filtered results as CSV spreadsheet"
        )
    
    with col2:
        # JSON export with formatting
        json_data = json.dumps(filtered_plans, indent=2, ensure_ascii=False)
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"norwegian_service_plans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download filtered results as structured JSON data"
        )
    
    with col3:
        # Excel export (if openpyxl is available)
        try:
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df = pd.DataFrame(filtered_plans)
                df.to_excel(writer, sheet_name='Service Plans', index=False)
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"norwegian_service_plans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download filtered results as Excel workbook"
            )
        except ImportError:
            st.button("📊 Excel Export", disabled=True, help="Requires openpyxl package")
    
    with col4:
        # Summary report export
        summary_report = generate_summary_report(filtered_plans)
        st.download_button(
            label="📑 Summary Report",
            data=summary_report,
            file_name=f"extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download summary report with key statistics"
        )


def generate_summary_report(plans: List[Dict]) -> str:
    """Generate comprehensive summary report of extracted plans"""
    if not plans:
        return "No plans available for summary."
    
    # Calculate statistics
    total_plans = len(plans)
    providers = set(plan['provider'] for plan in plans)
    categories = set(plan['category'] for plan in plans)
    
    # Price statistics
    prices = [plan['monthly_price'] for plan in plans if plan['monthly_price'] > 0]
    avg_price = sum(prices) / len(prices) if prices else 0
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    
    # Confidence statistics
    confidences = [plan.get('confidence', 0) for plan in plans]
    avg_confidence = sum(confidences) / len(confidences)
    high_confidence_count = sum(1 for c in confidences if c >= 0.8)
    
    # Generate report
    report = f"""
NORWEGIAN SERVICE PLANS - EXTRACTION SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=================================================

OVERVIEW STATISTICS
------------------
Total Plans Extracted: {total_plans}
Unique Providers: {len(providers)}
Service Categories: {len(categories)}
Plans with Pricing: {len(prices)} ({len(prices)/total_plans*100:.1f}%)

PROVIDER BREAKDOWN
-----------------
"""
    
    # Provider statistics
    provider_counts = {}
    for plan in plans:
        provider = plan['provider']
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    
    for provider, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_plans * 100
        report += f"{provider}: {count} plans ({percentage:.1f}%)\n"
    
    report += f"""
CATEGORY BREAKDOWN
-----------------
"""
    
    # Category statistics
    category_counts = {}
    for plan in plans:
        category = plan['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_plans * 100
        report += f"{category.title()}: {count} plans ({percentage:.1f}%)\n"
    
    report += f"""
PRICING ANALYSIS
---------------
Average Price: {avg_price:.0f} NOK/month
Minimum Price: {min_price:.0f} NOK/month
Maximum Price: {max_price:.0f} NOK/month
Plans with Pricing: {len(prices)}/{total_plans} ({len(prices)/total_plans*100:.1f}%)

QUALITY METRICS
--------------
Average Confidence Score: {avg_confidence:.1%}
High Confidence Plans: {high_confidence_count}/{total_plans} ({high_confidence_count/total_plans*100:.1f}%)
Data Completeness: {avg_confidence:.1%}

EXTRACTION METHODS
-----------------
"""
    
    # Extraction method statistics
    method_counts = {}
    for plan in plans:
        method = plan.get('extraction_method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_plans * 100
        report += f"{method.title()}: {count} plans ({percentage:.1f}%)\n"
    
    report += f"""
=================================================
End of Report
"""
    
    return report


def render_analytics_dashboard():
    """Render comprehensive analytics dashboard with advanced visualizations"""
    st.subheader("📊 Advanced Analytics Dashboard")
    
    try:
        all_plans = st.session_state.engine.get_all_plans()
        if not all_plans:
            st.info("📋 No data available for analytics. Extract some service plans first!")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_plans)
        df['extracted_at'] = pd.to_datetime(df['extracted_at'])
        
        # Render different analytics sections
        render_price_analysis_dashboard(df)
        render_competitive_analysis_dashboard(df)
        render_quality_trends_dashboard(df)
        render_market_insights_dashboard(df)
        
    except Exception as e:
        st.error(f"❌ Failed to load analytics data: {str(e)}")


def render_price_analysis_dashboard(df: pd.DataFrame):
    """Render comprehensive price analysis with market positioning"""
    st.write("**💰 Price Analysis & Market Positioning**")
    
    # Filter plans with pricing data
    priced_plans = df[df['monthly_price'] > 0].copy()
    
    if priced_plans.empty:
        st.info("No pricing data available for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by category
        fig = px.box(
            priced_plans,
            x='category',
            y='monthly_price',
            title="Price Distribution by Service Category",
            labels={'monthly_price': 'Monthly Price (NOK)', 'category': 'Service Category'}
        )
        fig.update_traces(marker_color='#002868')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price vs confidence scatter plot
        fig = px.scatter(
            priced_plans,
            x='monthly_price',
            y='confidence',
            color='provider',
            size='validation_score',
            title="Price vs Data Confidence",
            labels={
                'monthly_price': 'Monthly Price (NOK)',
                'confidence': 'Confidence Score'
            },
            hover_data=['name']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = priced_plans['monthly_price'].mean()
        st.metric("Average Price", f"{avg_price:.0f} NOK")
    
    with col2:
        median_price = priced_plans['monthly_price'].median()
        st.metric("Median Price", f"{median_price:.0f} NOK")
    
    with col3:
        price_std = priced_plans['monthly_price'].std()
        st.metric("Price Volatility", f"{price_std:.0f} NOK")
    
    with col4:
        price_range = priced_plans['monthly_price'].max() - priced_plans['monthly_price'].min()
        st.metric("Price Range", f"{price_range:.0f} NOK")


def render_competitive_analysis_dashboard(df: pd.DataFrame):
    """Render competitive analysis with provider comparisons"""
    st.write("**🏆 Competitive Analysis**")
    
    # Provider performance comparison
    provider_stats = df.groupby('provider').agg({
        'id': 'count',
        'monthly_price': ['mean', 'min', 'max'],
        'confidence': 'mean',
        'validation_score': 'mean'
    }).round(2)
    
    provider_stats.columns = ['Plans Count', 'Avg Price', 'Min Price', 'Max Price', 'Avg Confidence', 'Avg Validation']
    provider_stats = provider_stats.reset_index()
    
    # Competitive positioning chart
    fig = px.scatter(
        provider_stats,
        x='Avg Price',
        y='Avg Confidence',
        size='Plans Count',
        color='provider',
        title="Provider Competitive Positioning",
        labels={
            'Avg Price': 'Average Price (NOK)',
            'Avg Confidence': 'Data Quality Score'
        },
        hover_data=['Plans Count']
    )
    
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="High Quality Threshold")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Provider comparison table
    st.write("**📊 Provider Performance Comparison**")
    st.dataframe(
        provider_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg Price": st.column_config.NumberColumn("Avg Price", format="%.0f NOK"),
            "Min Price": st.column_config.NumberColumn("Min Price", format="%.0f NOK"),
            "Max Price": st.column_config.NumberColumn("Max Price", format="%.0f NOK"),
            "Avg Confidence": st.column_config.ProgressColumn("Avg Confidence", min_value=0, max_value=1),
            "Avg Validation": st.column_config.ProgressColumn("Avg Validation", min_value=0, max_value=1)
        }
    )


def render_quality_trends_dashboard(df: pd.DataFrame):
    """Render data quality trends and improvement analytics"""
    st.write("**📈 Data Quality Trends**")
    
    # Quality over time analysis
    df['date'] = df['extracted_at'].dt.date
    daily_quality = df.groupby('date').agg({
        'confidence': 'mean',
        'validation_score': 'mean',
        'id': 'count'
    }).reset_index()
    
    daily_quality.columns = ['Date', 'Avg Confidence', 'Avg Validation', 'Plans Count']
    
    # Quality trends chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Quality Scores Over Time', 'Extraction Volume'),
        vertical_spacing=0.1
    )
    
    # Quality scores
    fig.add_trace(
        go.Scatter(
            x=daily_quality['Date'],
            y=daily_quality['Avg Confidence'],
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color='#002868', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_quality['Date'],
            y=daily_quality['Avg Validation'],
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='#EF2B2D', width=3)
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=daily_quality['Date'],
            y=daily_quality['Plans Count'],
            name='Plans Extracted',
            marker_color='#002868',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="Quality Score", row=1, col=1)
    fig.update_yaxes(title_text="Plans Count", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def render_market_insights_dashboard(df: pd.DataFrame):
    """Render market insights with AI-powered recommendations"""
    st.write("**🧠 Market Insights & Recommendations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Market Coverage Analysis**")
        
        # Market coverage metrics
        total_providers = df['provider'].nunique()
        total_categories = df['category'].nunique()
        plans_with_pricing = len(df[df['monthly_price'] > 0])
        total_plans = len(df)
        
        coverage_metrics = {
            'Metric': ['Provider Coverage', 'Category Coverage', 'Pricing Coverage', 'Data Quality'],
            'Value': [
                f"{total_providers} providers",
                f"{total_categories} categories", 
                f"{plans_with_pricing/total_plans*100:.1f}%",
                f"{df['confidence'].mean()*100:.1f}%"
            ],
            'Status': [
                'Good' if total_providers >= 5 else 'Needs Improvement',
                'Good' if total_categories >= 3 else 'Needs Improvement',
                'Good' if plans_with_pricing/total_plans >= 0.7 else 'Needs Improvement',
                'Good' if df['confidence'].mean() >= 0.8 else 'Needs Improvement'
            ]
        }
        
        coverage_df = pd.DataFrame(coverage_metrics)
        st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**🎯 Optimization Recommendations**")
        
        recommendations = generate_optimization_recommendations(df)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")


def generate_optimization_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate AI-powered optimization recommendations based on data analysis"""
    recommendations = []
    
    # Data quality recommendations
    avg_confidence = df['confidence'].mean()
    if avg_confidence < 0.8:
        recommendations.append("Improve extraction patterns to increase data confidence scores")
    
    # Coverage recommendations  
    missing_prices = len(df[df['monthly_price'] <= 0]) / len(df)
    if missing_prices > 0.3:
        recommendations.append("Enhance price extraction algorithms for better pricing coverage")
    
    # Provider diversity recommendations
    provider_distribution = df['provider'].value_counts()
    if provider_distribution.std() > provider_distribution.mean():
        recommendations.append("Balance extraction across providers for more representative data")
    
    # Category expansion recommendations
    category_counts = df['category'].value_counts()
    if 'mobile' in category_counts.index and category_counts['mobile'] > len(df) * 0.8:
        recommendations.append("Expand extraction to more service categories beyond mobile")
    
    # Freshness recommendations
    latest_extraction = df['extracted_at'].max()
    hours_since_latest = (datetime.now() - latest_extraction).total_seconds() / 3600
    if hours_since_latest > 24:
        recommendations.append("Run more frequent extractions to keep data current")
    
    # Quality improvement recommendations
    low_quality_plans = len(df[df['confidence'] < 0.5])
    if low_quality_plans > 0:
        recommendations.append(f"Review and improve {low_quality_plans} low-quality plan extractions")
    
    return recommendations if recommendations else ["System is performing optimally - no immediate recommendations"]


def render_empty_state():
    """Render empty state with helpful getting started information"""
    st.info("""
    📋 **No service plans available yet**
    
    **Get started by:**
    1. 🚀 Use the sidebar to extract data from Norwegian providers
    2. 🔍 Try the predefined providers (Telia, Telenor, Ice, Talkmore, etc.)
    3. 🌐 Enter a custom Norwegian service provider URL
    4. 🔄 Run bulk extraction to get comprehensive data from all known providers
    
    **Norwegian Providers Available:**
    - **Mobile Services:** Telia, Telenor, Ice, Talkmore
    - **Electricity:** Fortum, Hafslund, Lyse, Fjordkraft
    """)
    
    # Quick start buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Quick Start - Telia Mobile", type="secondary"):
            start_background_extraction("https://www.telia.no/privat/mobil/abonnement", "Telia Mobile")
    
    with col2:
        if st.button("📱 Quick Start - Telenor Mobile", type="secondary"):
            start_background_extraction("https://www.telenor.no/privat/mobil/abonnement/", "Telenor Mobile")
    
    with col3:
        if st.button("⚡ Quick Start - Fortum Energy", type="secondary"):
            start_background_extraction("https://www.fortum.no/privat/strom/stromavtaler", "Fortum Energy")


def render_toast_notifications():
    """Render toast notifications for user feedback"""
    if not st.session_state.notifications:
        return
    
    # Show recent notifications (last 5)
    recent_notifications = st.session_state.notifications[-5:]
    
    for notification in recent_notifications:
        # Check if notification is still fresh (last 10 seconds)
        if time.time() - notification['timestamp'] < 10:
            toast_type = notification['type']
            message = notification['message']
            
            # Use Streamlit's native notification system
            if toast_type == 'success':
                st.success(message)
            elif toast_type == 'error':
                st.error(message)
            elif toast_type == 'warning':
                st.warning(message)
            else:
                st.info(message)


# Application cleanup and shutdown handling
def cleanup_on_shutdown():
    """Cleanup function called when application shuts down"""
    try:
        if 'engine' in st.session_state and st.session_state.engine:
            logger.info("Cleaning up extraction engine on shutdown")
            # Note: Don't shutdown the singleton engine here as it may be used by other sessions
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Register cleanup function
import atexit
atexit.register(cleanup_on_shutdown)


# Import logging for better error tracking
import logging
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    main()