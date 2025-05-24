# app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime
import base64
from io import BytesIO

# --- Configuration and Utility Imports ---
try:
    from config import (
        APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS,
        CONCEPTUAL_COLUMN_CATEGORIES,
        RISK_FREE_RATE, LOG_FILE, LOG_LEVEL, LOG_FORMAT,
        DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS, EXPECTED_COLUMNS
    )
    from kpi_definitions import KPI_CONFIG
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message, log_execution_time
    from components.sidebar_manager import SidebarManager
    from components.column_mapper_ui import ColumnMapperUI
    from components.scroll_buttons import ScrollButtons
    from services.data_service import DataService, get_benchmark_data_static, create_db_tables # Import create_db_tables
    from services.analysis_service import AnalysisService
    from services.auth_service import AuthService # <<< NEW: Import AuthService
except ImportError as e:
    # Fallback for critical import errors to allow app to at least show an error
    st.error(f"Fatal Error: A critical module could not be imported. The application cannot start. Details: {e}")
    # Basic logging setup if main logger fails
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Fatal Error during initial imports: {e}", exc_info=True)
    # Define critical variables to prevent further NameErrors if st.stop() doesn't halt immediately
    APP_TITLE = "TradingAppError"
    LOG_FILE = "logs/critical_error_app.log"
    LOG_LEVEL = "ERROR"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # Attempt to set up a minimal logger
    try:
        logger = logging.getLogger(APP_TITLE)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(LOG_LEVEL.upper())
    except Exception as log_setup_e:
        print(f"Fallback logger setup failed: {log_setup_e}")
    st.stop()


# --- Page Config (Must be the first Streamlit command) ---
# Using PAGE_CONFIG_APP_TITLE from config.py for consistency
PAGE_CONFIG_APP_TITLE = APP_TITLE # Use the main APP_TITLE from config
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png" # Ensure this path is correct

st.set_page_config(
    page_title=PAGE_CONFIG_APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB,
    layout="wide",
    initial_sidebar_state="expanded", # Keep expanded for main app, will be managed for auth
    menu_items={
        'Get Help': 'https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4', # Replace with your repo
        'Report a bug': "https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4/issues", # Replace
        'About': f"## {PAGE_CONFIG_APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)

# --- Initialize Logger, Services, and Database Tables ---
logger = setup_logger(logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")

# Initialize services
data_service = DataService()
analysis_service_instance = AnalysisService()
auth_service = AuthService() # <<< NEW: Initialize AuthService

# Create database tables (including User table) if they don't exist
# This should be called once at startup.
try:
    create_db_tables()
    logger.info("Database tables checked/created successfully.")
except Exception as db_init_e:
    logger.critical(f"Failed to initialize database tables: {db_init_e}", exc_info=True)
    st.error(f"Database Initialization Error: {db_init_e}. The application might not function correctly.")
    # Depending on severity, you might st.stop() here if DB is critical for all operations.

# --- Theme Management & CSS ---
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "dark" # Default to dark theme

theme_js = f"""
<script>
    const currentTheme = '{st.session_state.current_theme}';
    document.documentElement.setAttribute('data-theme', currentTheme);
    if (currentTheme === "dark") {{
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    }} else {{
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
    }}
</script>
"""
st.components.v1.html(theme_js, height=0)

try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path):
        load_css(css_file_path)
    else:
        logger.error(f"style.css not found at '{css_file_path}'. Custom styles may not apply.")
except Exception as e_css:
    logger.error(f"Failed to load style.css: {e_css}", exc_info=True)


# --- Authentication State Initialization ---
if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = None # Will store {'user_id': id, 'username': username}
if 'auth_flow_page' not in st.session_state:
    st.session_state.auth_flow_page = 'login' # 'login' or 'register'


# --- Authentication UI Functions ---
def display_login_form():
    """Displays the login form and handles login logic."""
    with st.container(): # Use a container for better layout control
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        with st.form("login_form", border=True): # Add border for visual grouping
            st.markdown(f"<h2 style='text-align: center;'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Username and password are required.")
                else:
                    user = auth_service.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated_user = {'user_id': user.id, 'username': user.username}
                        st.session_state.auth_flow_page = None # Clear auth flow page
                        logger.info(f"User '{username}' logged in successfully.")
                        st.success(f"Welcome back, {username}!")
                        st.rerun() # Rerun to show main app
                    else:
                        st.error("Invalid username or password.")
            
            if st.button("Don't have an account? Register", use_container_width=True, key="goto_register_btn"):
                st.session_state.auth_flow_page = 'register'
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
    """Displays the registration form and handles registration logic."""
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        with st.form("registration_form", border=True):
            st.markdown(f"<h2 style='text-align: center;'>Register for {APP_TITLE}</h2>", unsafe_allow_html=True)
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email (Optional)", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
            
            reg_submitted = st.form_submit_button("Register", use_container_width=True, type="primary")

            if reg_submitted:
                if not reg_username or not reg_password or not reg_password_confirm:
                    st.error("Username, password, and password confirmation are required.")
                elif reg_password != reg_password_confirm:
                    st.error("Passwords do not match.")
                else:
                    # Basic password complexity check (example)
                    if len(reg_password) < 8:
                         st.error("Password must be at least 8 characters long.")
                    else:
                        user = auth_service.register_user(reg_username, reg_password, reg_email if reg_email else None)
                        if user:
                            st.success(f"User '{reg_username}' registered successfully! Please login.")
                            st.session_state.auth_flow_page = 'login' # Switch to login
                            st.rerun()
                        else:
                            # AuthService get_user_by_username or email check might return None if user exists
                            # Need more specific error from service or check here
                            if auth_service.get_user_by_username(reg_username):
                                st.error(f"Username '{reg_username}' already exists. Please choose another.")
                            elif reg_email and auth_service.get_user_by_username(reg_email): # Assuming get_user_by_email exists or is similar
                                st.error(f"Email '{reg_email}' is already registered.")
                            else:
                                st.error("Registration failed. Please try again or contact support.")
            
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn"):
                st.session_state.auth_flow_page = 'login'
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# --- Main Application Logic ---
if st.session_state.authenticated_user is None:
    # No user logged in, show authentication forms
    st.sidebar.empty() # Clear sidebar for auth flow
    if st.session_state.auth_flow_page == 'login':
        display_login_form()
    elif st.session_state.auth_flow_page == 'register':
        display_registration_form()
    else: # Default to login if auth_flow_page is None or invalid
        st.session_state.auth_flow_page = 'login'
        display_login_form()
    st.stop() # Stop further execution if not authenticated

# --- USER IS AUTHENTICATED - Proceed with main app logic ---

# Initialize session state for the main application (if not already done by auth flow)
default_session_state = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None,
    'kpi_results': None, 'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE, 'uploaded_file_name': None,
    'uploaded_file_bytes_for_mapper': None, 'last_processed_file_id': None,
    'user_column_mapping': None, 'column_mapping_confirmed': False,
    'csv_headers_for_mapping': None, 'last_uploaded_file_for_mapping_id': None,
    'last_applied_filters': None, 'sidebar_filters': None, 'active_tab': "📈 Overview",
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER,
    'selected_benchmark_display_name': next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == DEFAULT_BENCHMARK_TICKER), "None"),
    'benchmark_daily_returns': None, 'initial_capital': 100000.0,
    'last_fetched_benchmark_ticker': None, 'last_benchmark_data_filter_shape': None,
    'last_kpi_calc_state_id': None,
    'max_drawdown_period_details': None
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Sidebar Setup for Authenticated User ---
# Logo
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"
logo_base64 = None
if os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        with open(LOGO_PATH_SIDEBAR, "rb") as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()
    except Exception as e_logo:
        logger.error(f"Error encoding logo: {e_logo}", exc_info=True)

if logo_base64:
    st.logo(f"data:image/png;base64,{logo_base64}", icon_image=f"data:image/png;base64,{logo_base64}")
elif os.path.exists(LOGO_PATH_SIDEBAR): # Fallback if st.logo fails or not preferred for some reason
     st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')
else:
    logger.warning(f"Logo file NOT FOUND at {LOGO_PATH_SIDEBAR}")


st.sidebar.header(APP_TITLE)
st.sidebar.markdown(f"Logged in as: **{st.session_state.authenticated_user['username']}**")
if st.sidebar.button("🔒 Logout", key="logout_button_main_app", use_container_width=True):
    logger.info(f"User '{st.session_state.authenticated_user['username']}' logging out.")
    # Clear all session state keys related to user data and auth
    user_specific_keys_to_clear = [
        'authenticated_user', 'processed_data', 'filtered_data', 'kpi_results',
        'kpi_confidence_intervals', 'uploaded_file_name', 'uploaded_file_bytes_for_mapper',
        'last_processed_file_id', 'user_column_mapping', 'column_mapping_confirmed',
        'csv_headers_for_mapping', 'last_uploaded_file_for_mapping_id', 'last_applied_filters',
        'sidebar_filters', 'benchmark_daily_returns', 'max_drawdown_period_details',
        'last_fetched_benchmark_ticker', 'last_benchmark_data_filter_shape', 'last_kpi_calc_state_id'
        # Add any other user-specific keys here
    ]
    for key in user_specific_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.auth_flow_page = 'login' # Prepare for next login
    st.success("You have been logged out.")
    st.rerun()

st.sidebar.markdown("---")

# Theme Toggle
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_auth", use_container_width=True):
    st.session_state.current_theme = "dark" if st.session_state.current_theme == "light" else "light"
    st.rerun()
st.sidebar.markdown("---")

# --- EXISTING MAIN APP LOGIC (File Upload, Processing, etc.) ---
# This part is now guarded by the authentication check above.
# TODO: In Phase 2, this uploader will interact with user-specific file storage.
uploaded_file = st.sidebar.file_uploader(
    "Upload Trading Journal (CSV)",
    type=["csv"],
    key="app_wide_file_uploader_auth", # Ensure key is unique if old one might conflict
    help="Upload your trading data. This will be processed for the current session."
)

sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
current_sidebar_filters = sidebar_manager.render_sidebar_controls() # This also handles RFR, Benchmark, Initial Capital
st.session_state.sidebar_filters = current_sidebar_filters

# Update session state based on sidebar controls (RFR, Benchmark, Initial Capital)
if current_sidebar_filters:
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar:
        st.session_state.risk_free_rate = rfr_from_sidebar
        st.session_state.kpi_results = None # Force KPI recalc

    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == benchmark_ticker_from_sidebar), "None")
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None

    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar:
        st.session_state.initial_capital = initial_capital_from_sidebar
        st.session_state.kpi_results = None

# --- Data Loading and Processing Pipeline (from original app.py) ---
# This section remains largely the same for now, operating on the uploaded file.
# In Phase 2, 'uploaded_file' might come from user's saved files.
@log_execution_time
def get_and_process_data_with_profiling(file_obj, mapping, name):
    # Ensure file_obj is BytesIO if that's what the service expects
    if hasattr(file_obj, 'getvalue') and not isinstance(file_obj, BytesIO):
        file_bytes_io = BytesIO(file_obj.getvalue())
        file_bytes_io.seek(0)
        return data_service.get_processed_trading_data(file_bytes_io, user_column_mapping=mapping, original_file_name=name)
    elif isinstance(file_obj, BytesIO):
        file_obj.seek(0)
        return data_service.get_processed_trading_data(file_obj, user_column_mapping=mapping, original_file_name=name)
    logger.error("get_and_process_data_with_profiling: file_obj is not compatible.")
    return None


if uploaded_file is not None:
    current_file_id_for_mapping = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}-mapping_stage"
    if st.session_state.last_uploaded_file_for_mapping_id != current_file_id_for_mapping:
        logger.info(f"New file '{uploaded_file.name}' for mapping. Resetting relevant session state.")
        for key_to_reset in ['column_mapping_confirmed', 'user_column_mapping', 'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None # Reset to None
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_uploaded_file_for_mapping_id = current_file_id_for_mapping
        try:
            # Store bytes in session state for ColumnMapperUI and subsequent processing
            st.session_state.uploaded_file_bytes_for_mapper = BytesIO(uploaded_file.getvalue())
            st.session_state.uploaded_file_bytes_for_mapper.seek(0) # Reset pointer
            # For header peeking, create a new BytesIO object to avoid issues with the stored one
            df_peek = pd.read_csv(BytesIO(st.session_state.uploaded_file_bytes_for_mapper.getvalue()), nrows=5)
            st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
            st.session_state.uploaded_file_bytes_for_mapper.seek(0) # Ensure it's reset for ColumnMapperUI
        except Exception as e_header:
            logger.error(f"Could not read CSV headers/preview from '{uploaded_file.name}': {e_header}", exc_info=True)
            display_custom_message(f"Error reading from '{uploaded_file.name}': {e_header}. Ensure it's a valid CSV.", "error")
            st.session_state.csv_headers_for_mapping = None
            st.session_state.uploaded_file_bytes_for_mapper = None
            st.stop() # Stop if header reading fails

    if st.session_state.csv_headers_for_mapping and not st.session_state.column_mapping_confirmed:
        st.session_state.processed_data = None # Ensure no stale data is shown
        st.session_state.filtered_data = None
        column_mapper = ColumnMapperUI(
            uploaded_file_name=st.session_state.uploaded_file_name,
            uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper, # Pass the BytesIO from session
            csv_headers=st.session_state.csv_headers_for_mapping,
            conceptual_columns_map=CONCEPTUAL_COLUMNS,
            conceptual_column_types=CONCEPTUAL_COLUMN_TYPES,
            conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS,
            critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS,
            conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES
        )
        user_mapping_result = column_mapper.render()
        if user_mapping_result is not None: # Mapping confirmed by user
            st.session_state.user_column_mapping = user_mapping_result
            st.session_state.column_mapping_confirmed = True
            st.rerun() # Rerun to proceed to data processing
        else:
            st.stop() # Stop if mapping is not yet confirmed

    if st.session_state.column_mapping_confirmed and st.session_state.user_column_mapping:
        current_file_id_proc = f"{st.session_state.uploaded_file_name}-{uploaded_file.size}-{uploaded_file.type}-processing"
        # Process data if it's a new file, mapping is confirmed, or processed_data is None
        if st.session_state.last_processed_file_id != current_file_id_proc or st.session_state.processed_data is None:
            with st.spinner(f"Processing '{st.session_state.uploaded_file_name}'..."):
                file_obj_for_service = st.session_state.uploaded_file_bytes_for_mapper
                if file_obj_for_service: # Check if bytes are still available
                    file_obj_for_service.seek(0) # IMPORTANT: Reset pointer before re-reading
                    st.session_state.processed_data = get_and_process_data_with_profiling(
                        file_obj_for_service, # Pass the BytesIO object
                        st.session_state.user_column_mapping,
                        st.session_state.uploaded_file_name
                    )
                else: # Fallback if bytes were somehow lost from session_state (should not happen ideally)
                    logger.warning("uploaded_file_bytes_for_mapper was None at processing stage. Attempting to re-read from original uploaded_file.")
                    if uploaded_file: # Check if original uploader object is still valid
                        temp_bytes = BytesIO(uploaded_file.getvalue())
                        st.session_state.processed_data = get_and_process_data_with_profiling(
                            temp_bytes,
                            st.session_state.user_column_mapping,
                            st.session_state.uploaded_file_name
                        )
                    else:
                        st.session_state.processed_data = None
                        logger.error("Original uploaded_file object also not available for reprocessing.")

            st.session_state.last_processed_file_id = current_file_id_proc
            # Reset downstream data when new data is processed
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']:
                st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data # Initially, filtered is same as processed

            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
                display_custom_message(f"Successfully processed '{st.session_state.uploaded_file_name}'. View data and analyses in the respective pages.", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty:
                display_custom_message(f"Processing of '{st.session_state.uploaded_file_name}' resulted in empty data. Please check your CSV file content and column mapping.", "warning")
                st.session_state.column_mapping_confirmed = False # Allow re-mapping
                st.session_state.user_column_mapping = None
            else: # processed_data is None
                display_custom_message(f"Failed to process '{st.session_state.uploaded_file_name}'. Check logs and column mapping. You may need to re-upload or adjust mappings.", "error")
                st.session_state.column_mapping_confirmed = False # Allow re-mapping
                st.session_state.user_column_mapping = None


elif st.session_state.get('uploaded_file_name') and uploaded_file is None: # File was removed from uploader
    if st.session_state.processed_data is not None: # Only reset if there was data
        logger.info("File uploader is now empty. Resetting all data-dependent session states.")
        keys_to_reset_on_file_removal = [
            'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals',
            'uploaded_file_name', 'uploaded_file_bytes_for_mapper', 'last_processed_file_id',
            'user_column_mapping', 'column_mapping_confirmed', 'csv_headers_for_mapping',
            'last_uploaded_file_for_mapping_id', 'last_applied_filters', #'sidebar_filters', # Keep sidebar filter values
            'benchmark_daily_returns', 'last_fetched_benchmark_ticker',
            'last_benchmark_data_filter_shape', 'last_kpi_calc_state_id',
            'max_drawdown_period_details'
        ]
        for key_to_reset in keys_to_reset_on_file_removal:
            if key_to_reset in default_session_state and key_to_reset != 'sidebar_filters': # Check if key has a default
                 st.session_state[key_to_reset] = default_session_state[key_to_reset]
            elif key_to_reset != 'sidebar_filters': # Otherwise, set to None
                 st.session_state[key_to_reset] = None
        st.rerun()


# --- Data Filtering Logic (from original app.py) ---
@log_execution_time
def filter_data_with_profiling(df, filters, col_map):
    return data_service.filter_data(df, filters, col_map)

if st.session_state.processed_data is not None and not st.session_state.processed_data.empty and st.session_state.sidebar_filters:
    # Trigger filtering if filtered_data is None (e.g., after new processing) or if filters changed
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."):
            st.session_state.filtered_data = filter_data_with_profiling(
                st.session_state.processed_data, st.session_state.sidebar_filters, EXPECTED_COLUMNS
            )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        # Reset KPIs and benchmark data as filters changed
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None


# --- Benchmark Data Fetching Logic (from original app.py) ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None: refetch_benchmark = True
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker: refetch_benchmark = True
        # Also refetch if the date range of filtered_data has significantly changed
        # This check uses shape, but a date range check would be more precise
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape: refetch_benchmark = True

        if refetch_benchmark:
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date')
            min_d_str_to_fetch, max_d_str_to_fetch = None, None

            if date_col_conceptual in st.session_state.filtered_data.columns:
                dates_for_bm_filtered = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                if not dates_for_bm_filtered.empty:
                    min_d_filtered, max_d_filtered = dates_for_bm_filtered.min(), dates_for_bm_filtered.max()
                    # Ensure dates are valid and range is sensible
                    if pd.notna(min_d_filtered) and pd.notna(max_d_filtered) and (max_d_filtered.date() - min_d_filtered.date()).days >= 0:
                        min_d_str_to_fetch = min_d_filtered.strftime('%Y-%m-%d')
                        max_d_str_to_fetch = max_d_filtered.strftime('%Y-%m-%d')
            
            if min_d_str_to_fetch and max_d_str_to_fetch:
                with st.spinner(f"Fetching benchmark data for {selected_ticker}..."):
                    st.session_state.benchmark_daily_returns = get_benchmark_data_static(selected_ticker, min_d_str_to_fetch, max_d_str_to_fetch)
                st.session_state.last_fetched_benchmark_ticker = selected_ticker
                st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape # Store shape for comparison
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty:
                    display_custom_message(f"Could not fetch benchmark data for {selected_ticker} or no data returned for the period. Ensure ticker is valid and data exists for the range.", "warning")
            else:
                logger.warning(f"Cannot fetch benchmark for {selected_ticker} due to invalid/missing date range in filtered data.")
                st.session_state.benchmark_daily_returns = None # Ensure it's None if not fetched
            st.session_state.kpi_results = None # Force KPI recalc after benchmark change
    elif st.session_state.benchmark_daily_returns is not None: # If benchmark was deselected (set to None/"")
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None


# --- KPI Calculation Logic (from original app.py) ---
@log_execution_time
def get_core_kpis_with_profiling(df, rfr, benchmark_returns, capital):
    return analysis_service_instance.get_core_kpis(df, rfr, benchmark_returns, capital)

@log_execution_time
def get_advanced_drawdown_analysis_with_profiling(equity_series):
    return analysis_service_instance.get_advanced_drawdown_analysis(equity_series)

if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    # Create a unique ID for the current state for KPI calculation
    current_kpi_state_id_parts = [
        st.session_state.filtered_data.shape, # Basic check
        # More robust: hash of key columns or a checksum of the dataframe if performance allows
        # For now, shape + parameters should catch most changes
        st.session_state.risk_free_rate,
        st.session_state.initial_capital,
        st.session_state.selected_benchmark_ticker
    ]
    if st.session_state.benchmark_daily_returns is not None and not st.session_state.benchmark_daily_returns.empty:
        try:
            # Hash the benchmark series (index and values)
            benchmark_hash = pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum()
            current_kpi_state_id_parts.append(benchmark_hash)
        except Exception as e_hash_bm:
            logger.warning(f"Hashing benchmark_daily_returns failed: {e_hash_bm}. Using shape as fallback.")
            current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else:
        current_kpi_state_id_parts.append(None) # Placeholder if no benchmark data
    current_kpi_state_id = tuple(current_kpi_state_id_parts)

    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info("Recalculating KPIs, Confidence Intervals, and Max Drawdown Details due to data/parameter change.")
        with st.spinner("Calculating performance metrics..."):
            kpi_res = get_core_kpis_with_profiling(
                st.session_state.filtered_data,
                st.session_state.risk_free_rate,
                st.session_state.benchmark_daily_returns,
                st.session_state.initial_capital
            )
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res
                st.session_state.last_kpi_calc_state_id = current_kpi_state_id # Update last calc state ID

                # Advanced Drawdown (if data available)
                date_col_for_dd = EXPECTED_COLUMNS.get('date')
                cum_pnl_col_for_dd = 'cumulative_pnl' # This is an engineered column
                equity_series_for_dd = pd.Series(dtype=float)
                if date_col_for_dd and cum_pnl_col_for_dd and \
                   date_col_for_dd in st.session_state.filtered_data.columns and \
                   cum_pnl_col_for_dd in st.session_state.filtered_data.columns:
                    
                    temp_df_for_equity = st.session_state.filtered_data[[date_col_for_dd, cum_pnl_col_for_dd]].copy()
                    temp_df_for_equity[date_col_for_dd] = pd.to_datetime(temp_df_for_equity[date_col_for_dd], errors='coerce')
                    temp_df_for_equity.dropna(subset=[date_col_for_dd], inplace=True) # Drop rows where date conversion failed
                    if not temp_df_for_equity.empty:
                        # Ensure index is DatetimeIndex and sorted for drawdown analysis
                        equity_series_for_dd = temp_df_for_equity.set_index(date_col_for_dd)[cum_pnl_col_for_dd].sort_index().dropna()
                
                if not equity_series_for_dd.empty and len(equity_series_for_dd) >= 5: # Min points for meaningful analysis
                    adv_dd_results = get_advanced_drawdown_analysis_with_profiling(equity_series_for_dd)
                    st.session_state.max_drawdown_period_details = adv_dd_results.get('max_drawdown_details') if adv_dd_results and 'error' not in adv_dd_results else None
                    if adv_dd_results and 'error' in adv_dd_results: logger.warning(f"Advanced drawdown analysis error: {adv_dd_results['error']}")
                else:
                    st.session_state.max_drawdown_period_details = None
                    if equity_series_for_dd.empty: logger.info("Equity series for drawdown analysis is empty.")
                    else: logger.info(f"Not enough data points ({len(equity_series_for_dd)}) for advanced drawdown analysis.")

                # Confidence Intervals (if data available)
                pnl_col_for_ci = EXPECTED_COLUMNS.get('pnl')
                if pnl_col_for_ci and pnl_col_for_ci in st.session_state.filtered_data.columns:
                    pnl_series_for_ci = st.session_state.filtered_data[pnl_col_for_ci].dropna()
                    if len(pnl_series_for_ci) >= 10: # Min points for bootstrap
                        ci_res = analysis_service_instance.get_bootstrapped_kpi_cis(st.session_state.filtered_data, ['avg_trade_pnl', 'win_rate', 'sharpe_ratio'])
                        st.session_state.kpi_confidence_intervals = ci_res if ci_res and 'error' not in ci_res else {}
                    else: st.session_state.kpi_confidence_intervals = {}; logger.info(f"Not enough PnL data points ({len(pnl_series_for_ci)}) for CI bootstrapping.")
                else: st.session_state.kpi_confidence_intervals = {}; logger.info("PnL column for CI not found or empty.")
            else: # KPI calculation failed
                error_msg = kpi_res.get('error', 'Unknown error') if kpi_res else 'KPI calculation failed at service level'
                display_custom_message(f"KPI calculation error: {error_msg}. Please check data and mappings.", "error")
                st.session_state.kpi_results = None
                st.session_state.kpi_confidence_intervals = {}
                st.session_state.max_drawdown_period_details = None

elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    # Data is filtered to empty
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty: # Check if there was data before filtering
        display_custom_message("No data matches the current filter criteria. Adjust filters or upload a new file.", "info")
    # Reset KPIs if filtered data is empty
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}
    st.session_state.max_drawdown_period_details = None


# --- Welcome Page / Main Content Display Logic (from original app.py) ---
def main_page_layout():
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-section'>", unsafe_allow_html=True)
    st.markdown("<h1 class='welcome-title'>Trading Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='welcome-subtitle'>Powered by {PAGE_CONFIG_APP_TITLE}</p>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Unlock insights from your trading data with powerful analytics and visualizations.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h2 class='features-title' style='text-align: center; color: var(--secondary-color);'>Get Started</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1], gap="large")
    with col1:
        st.markdown("<div class='feature-item'><h4>📄 Upload Data</h4><p>Begin by uploading your trade journal (CSV) via the sidebar. Our intelligent mapping assistant will guide you.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-item'><h4>📊 Analyze Performance</h4><p>Dive deep into comprehensive performance metrics once your data is loaded and processed.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-item'><h4>💡 Discover Insights</h4><p>Leverage advanced tools like categorical analysis and AI-driven suggestions from the navigation pages.</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py" # Relative path for st.switch_page
    if os.path.exists(user_guide_page_path): # Check if the page file exists
        if st.button("📘 Read the User Guide", key="welcome_user_guide_button_auth", help="Navigate to the User Guide page"):
            st.switch_page(user_guide_page_path)
    else:
        st.markdown("<p style='text-align: center; font-style: italic;'>User guide page not found.</p>", unsafe_allow_html=True)
        logger.warning(f"User Guide page file not found at expected path: {user_guide_page_path}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Page Navigation and Display Logic ---
# This logic determines what to show based on the application state (file uploaded, mapping confirmed, data processed)

if not uploaded_file and st.session_state.processed_data is None:
    # No file uploaded AND no data processed yet (e.g., fresh session, or after file removal and state reset)
    main_page_layout() # Show the welcome page
    # st.stop() is implicitly handled by Streamlit's flow if this is the end of the script for this condition
elif uploaded_file and st.session_state.processed_data is None and not st.session_state.column_mapping_confirmed:
    # File uploaded, but mapping not done or data not processed yet. ColumnMapperUI handles its display.
    # If csv_headers_for_mapping is None, it means header reading failed.
    if st.session_state.csv_headers_for_mapping is None and uploaded_file:
         # Error message already displayed by header reading logic.
         # No need to call main_page_layout() here as ColumnMapperUI or error message takes precedence.
         pass
    # Otherwise, ColumnMapperUI is active, so don't show welcome page.
elif st.session_state.processed_data is not None and (st.session_state.filtered_data is None or st.session_state.filtered_data.empty):
    # Data processed, but current filters result in empty data.
    # An info message is already displayed by the filtering logic if processed_data was not empty.
    # The main app structure (sidebar, etc.) should be visible, allowing users to adjust filters or navigate.
    # So, don't show the welcome page here. Let the app proceed to render pages which will show "no data" messages.
    if not (st.session_state.kpi_results and 'error' not in st.session_state.kpi_results): # If KPIs also failed or are empty
        pass # Message already handled by KPI calculation logic
elif st.session_state.processed_data is None and st.session_state.get('uploaded_file_name') and not st.session_state.get('column_mapping_confirmed'):
    # This case implies a file was uploaded, its name stored, but then something went wrong before mapping confirmed (e.g. error during header read)
    # The ColumnMapperUI or an error message would be shown.
    pass


# --- Scroll Buttons (Rendered if user is authenticated and main app is shown) ---
scroll_buttons_component = ScrollButtons()
scroll_buttons_component.render()

logger.info(f"App '{APP_TITLE}' run cycle finished for user '{st.session_state.authenticated_user['username'] if st.session_state.authenticated_user else 'Guest'}'.")

