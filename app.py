# app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime # Ensure datetime is imported directly
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

    from services import (
        DataService, AnalysisService, AuthService,
        create_db_tables, get_benchmark_data_static
    )
except ImportError as e:
    # ... (fallback imports as before) ...
    st.error(f"Fatal Error: A critical module could not be imported. App cannot start. Details: {e}")
    logging.basicConfig(level=logging.ERROR); logging.error(f"Fatal Error: {e}", exc_info=True)
    APP_TITLE="TradingAppError"; RISK_FREE_RATE=0.02; DEFAULT_BENCHMARK_TICKER="SPY"; AVAILABLE_BENCHMARKS={"None":""}
    class DataService: pass; class AnalysisService: pass; class AuthService: pass;
    def create_db_tables(): pass; def get_benchmark_data_static(*args, **kwargs): return None
    class SidebarManager: def __init__(self,*args): pass; def render_sidebar_controls(self): return {}
    class ColumnMapperUI: def __init__(self,*args,**kwargs): pass; def render(self): return None
    class ScrollButtons: def __init__(self,*args,**kwargs): pass; def render(self): pass
    def load_css(f): pass; def display_custom_message(m,t): st.error(m); def setup_logger(**kwargs): return logging.getLogger(APP_TITLE)
    st.stop()


# --- Page Config ---
PAGE_CONFIG_APP_TITLE = APP_TITLE
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png"
st.set_page_config(page_title=PAGE_CONFIG_APP_TITLE, page_icon=LOGO_PATH_FOR_BROWSER_TAB, layout="wide", initial_sidebar_state="expanded", menu_items={}) # Simplified menu

logger = setup_logger(logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT)
logger.info(f"Application '{APP_TITLE}' starting.")

data_service = DataService()
analysis_service_instance = AnalysisService()
auth_service = AuthService()

try: create_db_tables(); logger.info("DB tables checked/created.")
except Exception as e: logger.critical(f"Failed to init DB tables: {e}", exc_info=True); st.error(f"DB Init Error: {e}.")

# --- Theme Management & CSS (incorporating user preference) ---
# Initialize user_preferences if not present (e.g., before login)
if 'user_preferences' not in st.session_state: st.session_state.user_preferences = {}

# Determine theme: 1. User preference (if logged in), 2. Session state (if toggled before login), 3. Default 'dark'
# This logic will be refined after login when user_preferences are loaded.
effective_theme = st.session_state.user_preferences.get('default_theme', st.session_state.get('current_theme', 'dark'))
if 'current_theme' not in st.session_state or st.session_state.current_theme != effective_theme:
    st.session_state.current_theme = effective_theme
    # No rerun here yet, will happen after login or if user toggles manually

theme_js = f"""
<script>
    const currentTheme = '{st.session_state.current_theme}';
    document.documentElement.setAttribute('data-theme', currentTheme);
    if (currentTheme === "dark") {{ document.body.classList.add('dark-mode'); document.body.classList.remove('light-mode'); }}
    else {{ document.body.classList.add('light-mode'); document.body.classList.remove('dark-mode'); }}
</script>
"""
st.components.v1.html(theme_js, height=0)
try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path): load_css(css_file_path)
    else: logger.error(f"style.css not found at '{css_file_path}'.")
except Exception as e_css: logger.error(f"Failed to load style.css: {e_css}", exc_info=True)

# --- Authentication State & UI ---
if 'authenticated_user' not in st.session_state: st.session_state.authenticated_user = None
if 'auth_flow_page' not in st.session_state: st.session_state.auth_flow_page = 'login'

# --- Session states for file and mapping (as before) ---
# ... (no changes to these specific session state initializations)
if 'selected_user_file_id' not in st.session_state: st.session_state.selected_user_file_id = None
if 'current_file_content_for_processing' not in st.session_state: st.session_state.current_file_content_for_processing = None
if 'pending_file_to_save_content' not in st.session_state: st.session_state.pending_file_to_save_content = None
if 'pending_file_to_save_name' not in st.session_state: st.session_state.pending_file_to_save_name = None
if 'trigger_file_save_processing' not in st.session_state: st.session_state.trigger_file_save_processing = False
if 'last_uploaded_raw_file_id_tracker' not in st.session_state: st.session_state.last_uploaded_raw_file_id_tracker = None
if 'trigger_file_load_id' not in st.session_state: st.session_state.trigger_file_load_id = None
if 'last_processed_mapping_for_file_id' not in st.session_state: st.session_state.last_processed_mapping_for_file_id = None


def display_login_form():
    # ... (implementation as before)
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("login_form_main_v3"):
                st.markdown(f"<h2 style='text-align: center;'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
                username = st.text_input("Username", key="login_username_main_v3")
                password = st.text_input("Password", type="password", key="login_password_main_v3")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    if not username or not password: st.error("Username and password are required.")
                    else:
                        user = auth_service.authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated_user = {'user_id': user.id, 'username': user.username}
                            st.session_state.auth_flow_page = None
                            logger.info(f"User '{username}' logged in successfully.")
                            
                            # --- Load user settings on login ---
                            user_settings = auth_service.get_user_settings(user.id)
                            if user_settings:
                                st.session_state.user_preferences = {
                                    'default_theme': user_settings.default_theme,
                                    'default_risk_free_rate': user_settings.default_risk_free_rate,
                                    'default_benchmark_ticker': user_settings.default_benchmark_ticker
                                }
                                # Apply theme preference immediately
                                if st.session_state.current_theme != user_settings.default_theme:
                                    st.session_state.current_theme = user_settings.default_theme
                                    # No rerun here, will happen with the main rerun after login success
                                logger.info(f"Loaded preferences for user {user.id}: {st.session_state.user_preferences}")
                            else:
                                # Should not happen if settings are created on registration, but handle defensively
                                st.session_state.user_preferences = {
                                    'default_theme': 'dark', # App default
                                    'default_risk_free_rate': RISK_FREE_RATE, # Global default
                                    'default_benchmark_ticker': DEFAULT_BENCHMARK_TICKER # Global default
                                }
                                logger.warning(f"No saved preferences found for user {user.id}, using app defaults.")
                            
                            st.success(f"Welcome back, {username}!"); st.rerun() # Rerun to apply theme and load main app
                        else: st.error("Invalid username or password.")
            if st.button("Don't have an account? Register", use_container_width=True, key="goto_register_btn_main_v4"):
                st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
    # ... (implementation as before, AuthService now creates default UserSettings)
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("registration_form_main_v3"):
                st.markdown(f"<h2 style='text-align: center;'>Register for {APP_TITLE}</h2>", unsafe_allow_html=True)
                reg_username = st.text_input("Username", key="reg_username_main_v3")
                reg_email = st.text_input("Email (Optional)", key="reg_email_main_v3")
                reg_password = st.text_input("Password", type="password", key="reg_password_main_v3")
                reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm_main_v3")
                reg_submitted = st.form_submit_button("Register", use_container_width=True, type="primary")
                if reg_submitted:
                    if not reg_username or not reg_password or not reg_password_confirm: st.error("Username, password, and confirmation are required.")
                    elif reg_password != reg_password_confirm: st.error("Passwords do not match.")
                    elif len(reg_password) < 8: st.error("Password must be at least 8 characters long.")
                    else:
                        user = auth_service.register_user(reg_username, reg_password, reg_email if reg_email else None)
                        if user:
                            st.success(f"User '{reg_username}' registered successfully! Default settings applied. Please login."); st.session_state.auth_flow_page = 'login'; st.rerun()
                        else:
                            if auth_service.get_user_by_username(reg_username): st.error(f"Username '{reg_username}' already exists.")
                            else: st.error("Registration failed. Username/email might be taken or internal error.")
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn_main_v4"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.authenticated_user is None:
    # ... (auth flow logic as before) ...
    st.sidebar.empty();
    if st.session_state.auth_flow_page == 'login': display_login_form()
    elif st.session_state.auth_flow_page == 'register': display_registration_form()
    else: st.session_state.auth_flow_page = 'login'; display_login_form()
    st.stop()

current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- Process Pending File Save/Load (as before) ---
# ... (No changes needed in these blocks for user settings persistence) ...
if st.session_state.get('trigger_file_save_processing') and st.session_state.get('pending_file_to_save_content') is not None:
    file_content_to_save_bytes = st.session_state.pending_file_to_save_content; original_name_to_save = st.session_state.pending_file_to_save_name
    st.session_state.trigger_file_save_processing = False; st.session_state.pending_file_to_save_content = None; st.session_state.pending_file_to_save_name = None
    temp_uploaded_file_for_service = BytesIO(file_content_to_save_bytes); temp_uploaded_file_for_service.name = original_name_to_save
    with st.empty().status(f"Saving '{original_name_to_save}'...", expanded=True) as status_save:
        status_save.write(f"Saving '{original_name_to_save}'...")
        saved_user_file_record = data_service.save_user_file(current_user_id, temp_uploaded_file_for_service)
        if saved_user_file_record:
            status_save.update(label="File saved! Triggering load...", state="complete")
            st.session_state.selected_user_file_id = saved_user_file_record.id; st.session_state.trigger_file_load_id = saved_user_file_record.id
            st.session_state.current_file_content_for_processing = None; st.session_state.processed_data = None
            st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None; st.session_state.uploaded_file_name = None
            st.session_state.last_processed_mapping_for_file_id = None; logger.info(f"File '{original_name_to_save}' saved for user {current_user_id}. Triggering load."); st.rerun()
        else: status_save.update(label=f"Failed to save '{original_name_to_save}'.", state="error"); st.error(f"Could not save file '{original_name_to_save}'.")

if st.session_state.get('trigger_file_load_id') is not None:
    file_id_to_load = st.session_state.trigger_file_load_id; st.session_state.trigger_file_load_id = None
    user_files = data_service.list_user_files(current_user_id)
    selected_file_record = next((f for f in user_files if f.id == file_id_to_load), None)
    if selected_file_record:
        with st.empty().status(f"Loading '{selected_file_record.original_file_name}'...", expanded=True) as status_load:
            status_load.write(f"Fetching content for '{selected_file_record.original_file_name}'...")
            file_content = data_service.get_user_file_content(file_id_to_load, current_user_id)
            if file_content:
                status_load.update(label="File loaded! Preparing for processing...", state="complete")
                st.session_state.current_file_content_for_processing = file_content; st.session_state.selected_user_file_id = file_id_to_load
                st.session_state.uploaded_file_name = selected_file_record.original_file_name
                st.session_state.processed_data = None; st.session_state.column_mapping_confirmed = False
                st.session_state.user_column_mapping = None; st.session_state.last_processed_file_id = None
                st.session_state.last_processed_mapping_for_file_id = None; logger.info(f"File ID {file_id_to_load} loaded for user {current_user_id}. Rerunning."); st.rerun()
            else: status_load.update(label=f"Failed to load content for '{selected_file_record.original_file_name}'.", state="error"); st.error(f"Could not load content for {selected_file_record.original_file_name}.")
                st.session_state.current_file_content_for_processing = None
                if st.session_state.selected_user_file_id == file_id_to_load: st.session_state.selected_user_file_id = None; st.session_state.uploaded_file_name = None
    elif file_id_to_load is not None: st.error(f"File ID {file_id_to_load} not found."); st.session_state.selected_user_file_id = None; st.session_state.current_file_content_for_processing = None; st.session_state.uploaded_file_name = None


# --- Initialize Main App Session State (as before) ---
default_session_state_main_app = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None, 'kpi_results': None,
    'kpi_confidence_intervals': {}, 'risk_free_rate': st.session_state.user_preferences.get('default_risk_free_rate', RISK_FREE_RATE),
    'uploaded_file_name': None, 'last_processed_file_id': None,
    'user_column_mapping': None, 'column_mapping_confirmed': False, 'csv_headers_for_mapping': None,
    'last_uploaded_file_for_mapping_id': None, 'last_applied_filters': None, 'sidebar_filters': None,
    'selected_benchmark_ticker': st.session_state.user_preferences.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER),
    'benchmark_daily_returns': None, 'initial_capital': 100000.0, 'last_fetched_benchmark_ticker': None,
    'last_benchmark_data_filter_shape': None, 'last_kpi_calc_state_id': None, 'max_drawdown_period_details': None,
    'uploaded_file_bytes_for_mapper': None, 'last_processed_mapping_for_file_id': None
}
default_session_state_main_app['selected_benchmark_display_name'] = next(
    (name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == default_session_state_main_app['selected_benchmark_ticker']), "None"
)
for key, value in default_session_state_main_app.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Sidebar for Authenticated User ---
# ... (Logo as before) ...
if logo_base64: st.logo(f"data:image/png;base64,{logo_base64}", icon_image=f"data:image/png;base64,{logo_base64}")
elif os.path.exists(LOGO_PATH_SIDEBAR): st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')

st.sidebar.header(APP_TITLE)
st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_button_main_app_v5_settings", use_container_width=True): # Incremented key
    logger.info(f"User '{current_username}' logging out.")
    # Preserve only current_theme if desired, or clear all for full reset
    # For settings persistence, user_preferences should be cleared on logout.
    keys_to_clear_on_logout = list(st.session_state.keys())
    for key_logout in keys_to_clear_on_logout:
        if key_logout not in ['current_theme']: # Keep theme if toggled before login
            del st.session_state[key_logout]
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None
    st.session_state.user_preferences = {} # Clear user preferences
    st.success("You have been logged out."); st.rerun()

st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_auth_v5_settings", use_container_width=True): # Incremented key
    new_theme = "dark" if st.session_state.current_theme == "light" else "light"
    st.session_state.current_theme = new_theme
    # Save theme preference to DB
    auth_service.update_user_settings(current_user_id, {'default_theme': new_theme})
    st.session_state.user_preferences['default_theme'] = new_theme # Update session state too
    logger.info(f"User {current_user_id} theme preference updated to {new_theme} and saved.")
    st.rerun()
st.sidebar.markdown("---")

# ... (File selection and delete logic as before, no direct changes for settings persistence here) ...
st.sidebar.subheader("📁 Your Trading Journals")
user_files = data_service.list_user_files(current_user_id) # Scoped to user
file_options = {f"{f.original_file_name} (Uploaded: {f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files}
file_options["✨ Upload New File..."] = "upload_new"
default_file_selection_label = "✨ Upload New File..."
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    matching_label = next((label for label, id_val in file_options.items() if id_val == st.session_state.selected_user_file_id), None)
    if matching_label: default_file_selection_label = matching_label

selected_file_label_in_sidebar = st.sidebar.selectbox(
    "Select a journal or upload new:", options=list(file_options.keys()),
    index=list(file_options.keys()).index(default_file_selection_label),
    key="select_user_file_v7_settings" # Incremented key
)
selected_file_id_from_sidebar_dropdown = file_options.get(selected_file_label_in_sidebar)

if selected_file_id_from_sidebar_dropdown == "upload_new":
    newly_uploaded_file_object = st.sidebar.file_uploader("Upload New Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader_auth_v7_settings", help="Your uploaded CSV will be saved to your account.")
    if newly_uploaded_file_object is not None:
        current_raw_file_id = id(newly_uploaded_file_object)
        if st.session_state.get('last_uploaded_raw_file_id_tracker') != current_raw_file_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file_object.getvalue(); st.session_state.pending_file_to_save_name = newly_uploaded_file_object.name
            st.session_state.trigger_file_save_processing = True; st.session_state.last_uploaded_raw_file_id_tracker = current_raw_file_id
            logger.debug(f"New file '{newly_uploaded_file_object.name}' for upload. Triggering save."); st.rerun()
elif selected_file_id_from_sidebar_dropdown is not None and st.session_state.selected_user_file_id != selected_file_id_from_sidebar_dropdown:
    st.session_state.trigger_file_load_id = selected_file_id_from_sidebar_dropdown
    st.session_state.last_processed_mapping_for_file_id = None # Reset mapping trigger for new file
    logger.debug(f"User selected file ID {selected_file_id_from_sidebar_dropdown}. Triggering load."); st.rerun()

if selected_file_id_from_sidebar_dropdown != "upload_new" and selected_file_id_from_sidebar_dropdown is not None:
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label_in_sidebar.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_sidebar_dropdown}_v6_settings"):
        if data_service.delete_user_file(selected_file_id_from_sidebar_dropdown, current_user_id, permanent_delete_local_file=True):
            st.sidebar.success(f"File '{selected_file_label_in_sidebar}' deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_from_sidebar_dropdown:
                # Clear all relevant states
                keys_to_clear = ['selected_user_file_id', 'current_file_content_for_processing', 'processed_data', 'filtered_data', 'kpi_results', 'uploaded_file_name', 'column_mapping_confirmed', 'user_column_mapping', 'last_uploaded_file_for_mapping_id', 'csv_headers_for_mapping', 'last_processed_mapping_for_file_id', 'last_processed_file_id']
                for k in keys_to_clear: st.session_state[k] = None
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")


# --- SidebarManager and handling its output for settings persistence ---
sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
current_sidebar_filters = sidebar_manager.render_sidebar_controls() # This now uses user_preferences for defaults
st.session_state.sidebar_filters = current_sidebar_filters

if current_sidebar_filters:
    settings_changed = False
    updated_settings_payload: Dict[str, Any] = {}

    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar:
        st.session_state.risk_free_rate = rfr_from_sidebar
        st.session_state.kpi_results = None # Force KPI recalc
        updated_settings_payload['default_risk_free_rate'] = rfr_from_sidebar
        settings_changed = True
    
    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', DEFAULT_BENCHMARK_TICKER)
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == benchmark_ticker_from_sidebar), "None")
        st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None
        updated_settings_payload['default_benchmark_ticker'] = benchmark_ticker_from_sidebar
        settings_changed = True

    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar:
        st.session_state.initial_capital = initial_capital_from_sidebar
        st.session_state.kpi_results = None # Force KPI recalc
        # Initial capital is not typically a "user preference" in the same way as theme/RFR,
        # but more of a per-analysis input. So, not saving this to UserSettings for now.
        # If it were, add: updated_settings_payload['default_initial_capital'] = initial_capital_from_sidebar; settings_changed = True

    if settings_changed and updated_settings_payload:
        auth_service.update_user_settings(current_user_id, updated_settings_payload)
        # Update session_state.user_preferences to reflect saved changes immediately
        for key, value in updated_settings_payload.items():
            st.session_state.user_preferences[key] = value
        logger.info(f"User {current_user_id} preferences updated and saved: {updated_settings_payload}")
        # A rerun might be good here if other parts of UI depend on these session states directly for display defaults
        # st.rerun() # Consider if needed, or if sidebar_manager already handles re-rendering correctly

# --- Data Processing Pipeline (as before, with mapping persistence logic) ---
# ... (The existing data processing pipeline logic from the previous update for column mapping persistence remains here) ...
# This section is complex and was updated in the previous step for column mapping.
# No direct changes needed here for *user settings* persistence, but it's part of the flow.
active_file_content_to_process = st.session_state.get('current_file_content_for_processing')
active_file_name_for_processing = st.session_state.get('uploaded_file_name')

@log_execution_time
def get_and_process_data_with_profiling(file_obj_bytesio: BytesIO, mapping: Dict[str, str], name_for_log: str):
    file_obj_bytesio.seek(0)
    return data_service.get_processed_trading_data(file_obj_bytesio, user_column_mapping=mapping, original_file_name=name_for_log)

if active_file_content_to_process and active_file_name_for_processing and current_user_id and st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    current_file_id_for_mapping = st.session_state.selected_user_file_id
    if st.session_state.get('last_processed_mapping_for_file_id') != current_file_id_for_mapping:
        logger.info(f"Processing mapping for file ID: {current_file_id_for_mapping}")
        st.session_state.last_processed_mapping_for_file_id = current_file_id_for_mapping
        loaded_mapping = data_service.get_user_column_mapping(current_user_id, current_file_id_for_mapping)
        if loaded_mapping:
            st.session_state.user_column_mapping = loaded_mapping; st.session_state.column_mapping_confirmed = True
            st.session_state.csv_headers_for_mapping = None; st.session_state.uploaded_file_bytes_for_mapper = None
            logger.info(f"Loaded saved mapping for file ID: {current_file_id_for_mapping}"); st.session_state.last_processed_file_id = None
        else:
            logger.info(f"No saved mapping for file ID: {current_file_id_for_mapping}. Showing UI."); st.session_state.user_column_mapping = None; st.session_state.column_mapping_confirmed = False
            try:
                active_file_content_to_process.seek(0); header_peek_io = BytesIO(active_file_content_to_process.getvalue())
                df_peek = pd.read_csv(header_peek_io, nrows=5); st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
                active_file_content_to_process.seek(0); st.session_state.uploaded_file_bytes_for_mapper = BytesIO(active_file_content_to_process.getvalue())
            except Exception as e_header: logger.error(f"Could not read CSV headers for '{active_file_name_for_processing}': {e_header}", exc_info=True); display_custom_message(f"Error reading headers: {e_header}.", "error"); st.session_state.csv_headers_for_mapping = None; st.session_state.current_file_content_for_processing = None; st.stop()
    if not st.session_state.get('column_mapping_confirmed', False) and st.session_state.get('csv_headers_for_mapping'):
        st.session_state.processed_data = None; st.session_state.filtered_data = None; st.session_state.kpi_results = None
        column_mapper = ColumnMapperUI(uploaded_file_name=active_file_name_for_processing, uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper, csv_headers=st.session_state.csv_headers_for_mapping, conceptual_columns_map=CONCEPTUAL_COLUMNS, conceptual_column_types=CONCEPTUAL_COLUMN_TYPES, conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS, critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS, conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES)
        user_mapping_result = column_mapper.render()
        if user_mapping_result is not None:
            save_success = data_service.save_user_column_mapping(current_user_id, current_file_id_for_mapping, user_mapping_result)
            if save_success: logger.info(f"Saved column mapping for file ID: {current_file_id_for_mapping}")
            else: logger.error(f"Failed to save column mapping for file ID: {current_file_id_for_mapping}"); display_custom_message("Error: Could not save mapping.", "error")
            st.session_state.user_column_mapping = user_mapping_result; st.session_state.column_mapping_confirmed = True
            st.session_state.last_processed_file_id = None; st.session_state.last_processed_mapping_for_file_id = current_file_id_for_mapping
            logger.info(f"Mapping confirmed & saved for '{active_file_name_for_processing}'. Rerunning."); st.rerun()
        else: display_custom_message("Please complete column mapping.", "info", icon="⚙️"); st.stop()
    if st.session_state.get('column_mapping_confirmed') and st.session_state.get('user_column_mapping'):
        if st.session_state.last_processed_file_id != current_file_id_for_mapping or st.session_state.processed_data is None:
            with st.spinner(f"Processing '{active_file_name_for_processing}'..."):
                active_file_content_to_process.seek(0)
                st.session_state.processed_data = get_and_process_data_with_profiling(active_file_content_to_process, st.session_state.user_column_mapping, active_file_name_for_processing)
            st.session_state.last_processed_file_id = current_file_id_for_mapping
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']: st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data
            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty: display_custom_message(f"Processed '{active_file_name_for_processing}'.", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty: display_custom_message(f"Processing '{active_file_name_for_processing}' resulted in empty data.", "warning")
            else: display_custom_message(f"Failed to process '{active_file_name_for_processing}'.", "error"); st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None; st.session_state.last_processed_mapping_for_file_id = None; st.session_state.current_file_content_for_processing = None; st.session_state.selected_user_file_id = None; st.rerun()
elif not active_file_content_to_process and st.session_state.authenticated_user:
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        logger.info("No active file. Clearing processed data states.")
        keys_to_clear = ['processed_data', 'filtered_data', 'kpi_results', 'uploaded_file_name', 'last_processed_file_id', 'user_column_mapping', 'column_mapping_confirmed', 'csv_headers_for_mapping', 'last_uploaded_file_for_mapping_id', 'benchmark_daily_returns', 'max_drawdown_period_details', 'uploaded_file_bytes_for_mapper', 'current_file_content_for_processing', 'last_processed_mapping_for_file_id']
        for k in keys_to_clear: st.session_state[k] = None
        if st.session_state.get('selected_user_file_id') is not None: st.session_state.selected_user_file_id = None

# --- Data Filtering, KPI Calc, Main Page Layout (as before) ---
# ... (These sections remain unchanged by the user settings persistence logic itself) ...
# (KPI Calculation logic from previous step)
if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty and st.session_state.get('sidebar_filters'):
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."):
            st.session_state.filtered_data = filter_data_with_profiling(st.session_state.processed_data, st.session_state.sidebar_filters, EXPECTED_COLUMNS)
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']: st.session_state[key_to_reset] = None
if st.session_state.get('filtered_data') is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None or st.session_state.last_fetched_benchmark_ticker != selected_ticker or st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape: refetch_benchmark = True
        if refetch_benchmark:
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date')
            min_d_str, max_d_str = None, None
            if date_col_conceptual in st.session_state.filtered_data.columns:
                try:
                    dates_filt = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                    if not dates_filt.empty: min_d, max_d = dates_filt.min(), dates_filt.max()
                    if pd.notna(min_d) and pd.notna(max_d) and (max_d.date() >= min_d.date()): min_d_str, max_d_str = min_d.strftime('%Y-%m-%d'), max_d.strftime('%Y-%m-%d')
                except Exception as e: logger.error(f"Error processing dates for benchmark: {e}", exc_info=True)
            if min_d_str and max_d_str:
                with st.spinner(f"Fetching benchmark: {st.session_state.selected_benchmark_display_name}..."): st.session_state.benchmark_daily_returns = get_benchmark_data_static(selected_ticker, min_d_str, max_d_str)
                st.session_state.last_fetched_benchmark_ticker = selected_ticker; st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty: display_custom_message(f"Could not fetch benchmark {selected_ticker}.", "warning")
            else: logger.warning(f"Cannot fetch benchmark {selected_ticker}: invalid date range."); st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None
    elif st.session_state.benchmark_daily_returns is not None: st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None
# (KPI calculation logic continued as before)
    current_kpi_state_id_parts = [st.session_state.filtered_data.shape, st.session_state.risk_free_rate, st.session_state.initial_capital, st.session_state.selected_benchmark_ticker]
    if st.session_state.get('benchmark_daily_returns') is not None and not st.session_state.benchmark_daily_returns.empty:
        try: current_kpi_state_id_parts.append(pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum())
        except Exception: current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else: current_kpi_state_id_parts.append(None)
    current_kpi_state_id = tuple(current_kpi_state_id_parts)
    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info(f"Recalculating KPIs. State ID: {current_kpi_state_id}")
        with st.spinner("Calculating KPIs..."):
            kpi_res = get_core_kpis_with_profiling(st.session_state.filtered_data, st.session_state.risk_free_rate, st.session_state.benchmark_daily_returns, st.session_state.initial_capital)
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res; st.session_state.last_kpi_calc_state_id = current_kpi_state_id
                # Advanced Drawdown & CI logic as before
                date_col_dd, cum_pnl_col_dd = EXPECTED_COLUMNS.get('date'), 'cumulative_pnl'
                equity_series_dd = pd.Series(dtype=float)
                if date_col_dd and cum_pnl_col_dd and date_col_dd in st.session_state.filtered_data and cum_pnl_col_dd in st.session_state.filtered_data:
                    df_eq_dd = st.session_state.filtered_data[[date_col_dd, cum_pnl_col_dd]].copy(); df_eq_dd[date_col_dd] = pd.to_datetime(df_eq_dd[date_col_dd], errors='coerce'); df_eq_dd.dropna(subset=[date_col_dd, cum_pnl_col_dd], inplace=True)
                    if not df_eq_dd.empty: equity_series_dd = df_eq_dd.set_index(date_col_dd)[cum_pnl_col_dd].sort_index()
                if not equity_series_dd.empty and len(equity_series_dd) >= 5:
                    adv_dd_res = get_advanced_drawdown_analysis_with_profiling(equity_series_dd)
                    st.session_state.max_drawdown_period_details = adv_dd_res.get('max_drawdown_details') if adv_dd_res and 'error' not in adv_dd_res else None
                else: st.session_state.max_drawdown_period_details = None
                pnl_col_ci = EXPECTED_COLUMNS.get('pnl')
                if pnl_col_ci and pnl_col_ci in st.session_state.filtered_data:
                    pnl_s_ci = st.session_state.filtered_data[pnl_col_ci].dropna()
                    if len(pnl_s_ci) >= 10: st.session_state.kpi_confidence_intervals = analysis_service_instance.get_bootstrapped_kpi_cis(st.session_state.filtered_data, ['avg_trade_pnl', 'win_rate', 'sharpe_ratio']) or {}
                    else: st.session_state.kpi_confidence_intervals = {}
                else: st.session_state.kpi_confidence_intervals = {}
            else: display_custom_message(f"KPI calculation error: {kpi_res.get('error', 'Unknown') if kpi_res else 'Service fail'}.", "error"); st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None
elif st.session_state.get('filtered_data') is not None and st.session_state.filtered_data.empty:
    if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty: display_custom_message("No data matches filters.", "info")
    st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None

# (main_page_layout and final display logic as before)
processed_data_main_final = st.session_state.get('processed_data')
condition_for_main_layout_final = not st.session_state.get('current_file_content_for_processing') and \
    not (st.session_state.get('column_mapping_confirmed') and processed_data_main_final is not None and not processed_data_main_final.empty)

if condition_for_main_layout_final: main_page_layout()
elif (st.session_state.get('processed_data') is None or st.session_state.get('processed_data').empty) and \
     st.session_state.get('current_file_content_for_processing') and \
     not st.session_state.get('column_mapping_confirmed'):
    if not st.session_state.get('csv_headers_for_mapping'): display_custom_message("Preparing for column mapping...", "info")

scroll_buttons_component = ScrollButtons(); scroll_buttons_component.render()
logger.info(f"App run cycle finished for user '{current_username}'. Active file ID: {st.session_state.get('selected_user_file_id')}, Processed: {st.session_state.get('processed_data') is not None and not st.session_state.get('processed_data').empty}")

