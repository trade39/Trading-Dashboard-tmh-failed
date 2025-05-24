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
from typing import Dict, Any, Optional

# --- Configuration and Utility Imports ---
# Ensure 'config' is at the root or PYTHONPATH is set correctly
try:
    from config import (
        APP_TITLE, LOGO_PATH_FOR_BROWSER_TAB, LOGO_PATH_SIDEBAR,
        RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS,
        LOG_FILE, LOG_LEVEL, LOG_FORMAT,
        CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS,
        CONCEPTUAL_COLUMN_CATEGORIES, EXPECTED_COLUMNS, APP_BASE_URL
    )
    from kpi_definitions import KPI_CONFIG # Assuming this is also at the root
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message, log_execution_time
    from components import (
        SidebarManager, ColumnMapperUI, ScrollButtons,
        # AuthUI components would be imported if auth_ui.py exists
    )
    from services import (
        DataService, AnalysisService, AuthService,
        create_db_tables, get_benchmark_data_static # Ensure get_benchmark_data_static is exported from services
    )
except ImportError as e:
    # This is a critical failure point for the app.
    # Attempt to show an error in Streamlit if st is available, otherwise print and exit.
    critical_error_msg = f"Fatal Error: A critical module could not be imported. App cannot start. Details: {e}. Check PYTHONPATH and file locations."
    try:
        import streamlit as st_fallback
        st_fallback.set_page_config(page_title="App Load Error", layout="centered")
        st_fallback.error(critical_error_msg)
        # Attempt to log this even if full logger isn't up
        logging.basicConfig(level=logging.ERROR)
        logging.critical(critical_error_msg, exc_info=True)
        st_fallback.stop()
    except Exception:
        print(critical_error_msg, file=sys.stderr)
        sys.exit(1)


# --- 1. Initial Setup ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={} # Customize as needed
)

logger = setup_logger(
    logger_name=APP_TITLE,
    log_file=LOG_FILE,
    level=LOG_LEVEL,
    log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting up.")

# --- Corrected CSS Loading ---
try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path):
        load_css(css_file_path)
        logger.info("Successfully loaded style.css.")
    else:
        logger.error(f"style.css not found at '{os.path.abspath(css_file_path)}'. Custom styles may not apply.")
        # Optionally, display a muted warning in the app if styles are critical
        # st.sidebar.caption("⚠️ Custom styles (style.css) not found.")
except Exception as e_css:
    logger.error(f"Failed to load style.css: {e_css}", exc_info=True)
    # st.sidebar.caption("⚠️ Error loading custom styles.")


# Initialize Database Tables (runs once)
try:
    create_db_tables()
    logger.info("Database tables checked/created successfully.")
except Exception as e_db_init:
    logger.critical(f"Failed to initialize database tables: {e_db_init}", exc_info=True)
    st.error(f"Database Initialization Error: {e_db_init}. The application may not function correctly.")
    # Depending on severity, you might st.stop() here if DB is absolutely critical for any operation.

# Initialize Services
auth_service = AuthService()
data_service = DataService()
analysis_service = AnalysisService() # Assuming AnalysisService is now implemented
# Other services like AIModelService, PortfolioAnalysisService can be initialized here if needed globally
# or instantiated within AnalysisService or page modules if their scope is more limited.


# --- 2. Session State Initialization ---
# Helper to initialize session state keys
def init_session_state_key(key: str, default_value: Any):
    if key not in st.session_state:
        st.session_state[key] = default_value

# Core Authentication State
init_session_state_key('authenticated_user', None)
init_session_state_key('auth_flow_page', 'login') # Default to login page
init_session_state_key('password_reset_token', None)

# User Preferences and Settings (populated on login or from defaults)
init_session_state_key('user_preferences', {}) # Will hold {'default_theme', 'default_risk_free_rate', etc.}
init_session_state_key('current_theme', st.session_state.user_preferences.get('default_theme', 'dark')) # Default to dark

# Data Processing Lifecycle State
init_session_state_key('selected_user_file_id', None)
init_session_state_key('uploaded_file_name', None) # Name of the currently selected/uploaded file
init_session_state_key('current_file_content_for_processing', None) # BytesIO object of the file
init_session_state_key('user_column_mapping', None) # The confirmed mapping dict
init_session_state_key('column_mapping_confirmed', False)
init_session_state_key('last_processed_file_id_for_mapping_ui', None) # Tracks file for ColumnMapperUI re-init
init_session_state_key('initial_mapping_override_for_ui', None) # Holds loaded mapping for ColumnMapperUI

# Analysis Results State (populated by AnalysisService)
init_session_state_key('processed_data', None) # Fully processed DataFrame for analysis
init_session_state_key('filtered_data', None)  # DataFrame after sidebar filters
init_session_state_key('kpi_results', None)    # Dict of calculated KPIs
init_session_state_key('kpi_confidence_intervals', {})
init_session_state_key('benchmark_daily_returns', None)
init_session_state_key('max_drawdown_period_details', None)
init_session_state_key('last_analysis_run_signature', None) # To track if analysis needs re-run

# Sidebar Filter State (these will be updated by SidebarManager's return values)
init_session_state_key('risk_free_rate', st.session_state.user_preferences.get('default_risk_free_rate', RISK_FREE_RATE))
init_session_state_key('selected_benchmark_ticker', st.session_state.user_preferences.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER))
init_session_state_key('selected_benchmark_display_name', next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == st.session_state.selected_benchmark_ticker), "None"))
init_session_state_key('initial_capital', 100000.0) # Default, can be from user settings if implemented
init_session_state_key('global_date_filter_range', None) # Tuple (start_date, end_date)
init_session_state_key('global_symbol_filter', "All")
init_session_state_key('global_strategy_filter', "All")

# UI Interaction Triggers
init_session_state_key('trigger_file_save_processing', False)
init_session_state_key('pending_file_to_save_content', None)
init_session_state_key('pending_file_to_save_name', None)
init_session_state_key('trigger_file_load_id', None) # Set by sidebar when a file is selected
init_session_state_key('last_uploaded_raw_file_id_tracker', None) # To prevent re-processing same upload on rerun


# --- 3. Theme Management ---
# Ensure theme is applied on each run based on session state
# This JS should be robust to re-runs.
effective_theme = st.session_state.current_theme
theme_js = f"""
<script>
    // console.log("Applying theme from app.py: {effective_theme}");
    const rootHtml = document.documentElement;
    const currentBodyClass = rootHtml.classList.contains('dark-mode') ? 'dark' : (rootHtml.classList.contains('light-mode') ? 'light' : null);
    const targetTheme = '{effective_theme}';

    if (targetTheme !== currentBodyClass) {{
        // console.log(`Theme change detected: from ${currentBodyClass} to ${targetTheme}`);
        rootHtml.setAttribute('data-theme', targetTheme);
        if (targetTheme === "dark") {{
            rootHtml.classList.add('dark-mode');
            rootHtml.classList.remove('light-mode');
        }} else {{
            rootHtml.classList.add('light-mode');
            rootHtml.classList.remove('dark-mode');
        }}
    }}
</script>
"""
st.components.v1.html(theme_js, height=0)


# --- 4. Authentication Flow ---
# (Using placeholder auth_ui functions; these would be imported from components.auth_ui)
def display_login_form_placeholder():
    # ... (Implementation from Batch 1 app.py, adapted to use self.auth_service) ...
    # This function should now ideally be in components/auth_ui.py
    # For brevity, direct implementation here for now.
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("login_form_main_v5_refactored"): # Unique key
                st.markdown(f"<h2 style='text-align: center;'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
                username = st.text_input("Username", key="login_username_main_v5_refactored")
                password = st.text_input("Password", type="password", key="login_password_main_v5_refactored")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    if not username or not password:
                        st.error("Username and password are required.")
                    else:
                        user = auth_service.authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated_user = {'user_id': user.id, 'username': user.username}
                            st.session_state.auth_flow_page = None # Clear auth flow
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
                                # Apply other preferences to session state for app-wide use
                                st.session_state.risk_free_rate = user_settings.default_risk_free_rate
                                st.session_state.selected_benchmark_ticker = user_settings.default_benchmark_ticker
                                st.session_state.selected_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == user_settings.default_benchmark_ticker), "None")

                            else: # Default preferences if no settings found (should be rare after registration)
                                st.session_state.user_preferences = {
                                    'default_theme': 'dark',
                                    'default_risk_free_rate': RISK_FREE_RATE,
                                    'default_benchmark_ticker': DEFAULT_BENCHMARK_TICKER
                                }
                            st.success(f"Welcome back, {username}!"); st.rerun()
                        else:
                            st.error("Invalid username or password.")
            col_auth_links1, col_auth_links2 = st.columns(2)
            with col_auth_links1:
                if st.button("Forgot Password?", use_container_width=True, key="forgot_password_link_v5_refactored"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.rerun()
            with col_auth_links2:
                if st.button("Register New Account", use_container_width=True, key="goto_register_btn_main_v5_refactored"):
                    st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form_placeholder():
    # ... (Implementation from Batch 1 app.py, adapted) ...
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center;'>Register for {APP_TITLE}</h2>", unsafe_allow_html=True)
            st.markdown("""
            <small>Password must:
            <ul>
                <li>Be at least 8 characters long</li>
                <li>Contain at least one uppercase letter (A-Z)</li>
                <li>Contain at least one lowercase letter (a-z)</li>
                <li>Contain at least one digit (0-9)</li>
                <li>Contain at least one special character (e.g., !@#$%^&*)</li>
            </ul></small>
            """, unsafe_allow_html=True)
            with st.form("registration_form_main_v5_refactored"): # Unique key
                reg_username = st.text_input("Username", key="reg_username_main_v5_refactored")
                reg_email = st.text_input("Email (Required for password reset)", key="reg_email_main_v5_refactored")
                reg_password = st.text_input("Password", type="password", key="reg_password_main_v5_refactored")
                reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm_main_v5_refactored")
                reg_submitted = st.form_submit_button("Register", use_container_width=True, type="primary")
                if reg_submitted:
                    if not reg_username or not reg_password or not reg_password_confirm or not reg_email:
                        st.error("Username, Email, password, and confirmation are required.")
                    elif reg_password != reg_password_confirm: st.error("Passwords do not match.")
                    else:
                        registration_result = auth_service.register_user(reg_username, reg_password, reg_email)
                        if registration_result.get("user"):
                            st.success(f"User '{reg_username}' registered! Please login."); st.session_state.auth_flow_page = 'login'; st.rerun()
                        else: st.error(registration_result.get("error", "Registration failed."))
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn_main_v5_refactored"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_forgot_password_request_form_placeholder():
    # ... (Implementation from Batch 1 app.py, adapted) ...
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center;'>Forgot Password</h2>", unsafe_allow_html=True)
            st.write("Enter your email. If an account exists, a reset link will be sent.")
            with st.form("forgot_password_request_form_v5_refactored"): # Unique key
                email_input = st.text_input("Your Email Address", key="forgot_pw_email_v5_refactored")
                submit_request = st.form_submit_button("Send Reset Link", use_container_width=True, type="primary")
                if submit_request:
                    if not email_input: st.error("Please enter your email address.")
                    else:
                        with st.spinner("Processing..."): result = auth_service.create_password_reset_token(email_input)
                        if result.get("success"): st.success(result["success"]) # Message is now generic
                        else: st.error(result.get("error", "Could not process request.")) # Generic error
            if st.button("Back to Login", use_container_width=True, key="forgot_pw_back_to_login_v5_refactored"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_reset_password_form_placeholder():
    # ... (Implementation from Batch 1 app.py, adapted) ...
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center;'>Reset Your Password</h2>", unsafe_allow_html=True)
            token = st.session_state.get('password_reset_token')
            if not token:
                st.error("Invalid or missing reset token. Please request a new link.");
                if st.button("Request New Link",use_container_width=True,key="reset_form_req_new_link_v5_refactored"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.session_state.password_reset_token = None; st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True); return

            st.markdown("""<small>New password must meet complexity rules (8+ chars, upper, lower, digit, special).</small>""", unsafe_allow_html=True)
            with st.form("reset_password_form_v5_refactored"): # Unique key
                new_password = st.text_input("New Password", type="password", key="reset_pw_new_v5_refactored")
                confirm_new_password = st.text_input("Confirm New Password", type="password", key="reset_pw_confirm_v5_refactored")
                submit_reset = st.form_submit_button("Reset Password", use_container_width=True, type="primary")
                if submit_reset:
                    if not new_password or not confirm_new_password: st.error("Please enter and confirm your new password.")
                    elif new_password != confirm_new_password: st.error("New passwords do not match.")
                    else:
                        with st.spinner("Resetting..."): result = auth_service.reset_password_with_token(token, new_password)
                        if result.get("success"):
                            st.success(result["success"]); st.info("You can now log in.");
                            st.session_state.auth_flow_page = 'login'; st.session_state.password_reset_token = None
                            # Auto-rerun will happen, or provide a button
                            st.page_link("app.py", label="Go to Login", icon="🔑") # Use page_link for navigation
                        else: st.error(result.get("error", "Could not reset password."))
            if st.button("Cancel", use_container_width=True, type="secondary", key="reset_pw_cancel_v5_refactored"):
                st.session_state.auth_flow_page = 'login'; st.session_state.password_reset_token = None; st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)

# Handle password reset token from URL query parameters
query_params = st.query_params
if "page" in query_params and query_params.get("page") == "reset_password_form" and "token" in query_params:
    token_from_url = query_params.get("token")
    if isinstance(token_from_url, list): token_from_url = token_from_url[0] # Get first element if it's a list

    if token_from_url and st.session_state.auth_flow_page != 'reset_password_form': # Process only if not already on the page
        logger.info(f"Password reset token '{token_from_url[:8]}...' received from URL.")
        # Verify token immediately to see if it's valid before switching page
        user_id_from_token_verify = auth_service.verify_password_reset_token(token_from_url)
        if user_id_from_token_verify:
            st.session_state.password_reset_token = token_from_url
            st.session_state.auth_flow_page = 'reset_password_form'
            logger.info(f"Token valid. Switching to reset password form for user ID {user_id_from_token_verify}.")
        else:
            logger.warning(f"Invalid or expired token received from URL: {token_from_url[:8]}...")
            st.session_state.auth_flow_page = 'login' # Redirect to login
            display_custom_message("The password reset link is invalid or has expired. Please request a new one.", "error")
        
        # Clear query params after processing to prevent re-triggering on refresh
        # st.query_params.clear() # This causes issues with Streamlit's execution flow
        # Instead, use st.experimental_set_query_params to remove them
        st.experimental_set_query_params() # Clears all query params
        st.rerun()


if st.session_state.authenticated_user is None:
    # Clear sidebar for unauthenticated users
    st.sidebar.empty()
    # Render the appropriate authentication form
    if st.session_state.auth_flow_page == 'login': display_login_form_placeholder()
    elif st.session_state.auth_flow_page == 'register': display_registration_form_placeholder()
    elif st.session_state.auth_flow_page == 'forgot_password_request': display_forgot_password_request_form_placeholder()
    elif st.session_state.auth_flow_page == 'reset_password_form': display_reset_password_form_placeholder()
    else: # Default to login if auth_flow_page is in an unexpected state
        st.session_state.auth_flow_page = 'login'
        display_login_form_placeholder()
    st.stop() # Stop further execution for unauthenticated users

# --- Main Application Logic (if authenticated) ---
current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- 5. Main Application Layout ---
# --- Sidebar Rendering ---
st.sidebar.header(APP_TITLE)
if LOGO_PATH_SIDEBAR and os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        st.logo(LOGO_PATH_SIDEBAR, icon_image=LOGO_PATH_SIDEBAR) # Use st.logo
    except Exception as e_logo:
        logger.error(f"Error loading logo with st.logo: {e_logo}. Trying st.sidebar.image as fallback.")
        try: st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')
        except Exception as e_logo_fb: logger.error(f"Error loading logo with st.sidebar.image: {e_logo_fb}")
elif LOGO_PATH_SIDEBAR:
    logger.warning(f"Sidebar logo path specified but not found: {LOGO_PATH_SIDEBAR}")

st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_button_main_app_v5_refactored", use_container_width=True):
    logger.info(f"User '{current_username}' logging out.")
    # Clear sensitive session state keys, keep theme
    keys_to_clear_on_logout = list(st.session_state.keys())
    for key_logout in keys_to_clear_on_logout:
        if key_logout not in ['current_theme']: # Keep current_theme
            del st.session_state[key_logout]
    st.session_state.auth_flow_page = 'login' # Set to login page
    st.session_state.authenticated_user = None # Ensure user is None
    st.session_state.user_preferences = {} # Clear preferences
    st.success("You have been logged out successfully.")
    st.rerun()

st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_v5_refactored", use_container_width=True):
    new_theme = "dark" if st.session_state.current_theme == "light" else "light"
    st.session_state.current_theme = new_theme
    # Persist theme preference to DB
    auth_service.update_user_settings(current_user_id, {'default_theme': new_theme})
    st.session_state.user_preferences['default_theme'] = new_theme
    logger.info(f"User {current_user_id} theme preference updated to {new_theme}.")
    st.rerun()
st.sidebar.markdown("---")

# File Management in Sidebar
st.sidebar.subheader("📁 Your Trading Journals")
user_files_list = data_service.list_user_files(current_user_id)
file_options_dict = {f"{f.original_file_name} ({f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files_list}
file_options_dict["✨ Upload New File..."] = "upload_new"

default_file_selection_label = "✨ Upload New File..."
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    matching_label = next((label for label, id_val in file_options_dict.items() if id_val == st.session_state.selected_user_file_id), None)
    if matching_label: default_file_selection_label = matching_label

selected_file_label = st.sidebar.selectbox(
    "Select or Upload Journal:",
    options=list(file_options_dict.keys()),
    index=list(file_options_dict.keys()).index(default_file_selection_label),
    key="select_user_file_v8_refactored"
)
selected_file_id_from_dropdown = file_options_dict.get(selected_file_label)

if selected_file_id_from_dropdown == "upload_new":
    newly_uploaded_file = st.sidebar.file_uploader(
        "Upload New Journal (CSV)", type=["csv"], key="app_wide_file_uploader_v8_refactored"
    )
    if newly_uploaded_file is not None:
        # Check if this is a genuinely new upload to prevent re-processing on simple reruns
        current_raw_file_id = id(newly_uploaded_file) # Use object ID as a simple tracker for new uploads
        if st.session_state.last_uploaded_raw_file_id_tracker != current_raw_file_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file.getvalue()
            st.session_state.pending_file_to_save_name = newly_uploaded_file.name
            st.session_state.trigger_file_save_processing = True
            st.session_state.last_uploaded_raw_file_id_tracker = current_raw_file_id
            st.rerun() # Rerun to trigger the save processing logic
elif selected_file_id_from_dropdown is not None and st.session_state.selected_user_file_id != selected_file_id_from_dropdown:
    # User selected an existing file from the dropdown
    st.session_state.trigger_file_load_id = selected_file_id_from_dropdown
    st.session_state.column_mapping_confirmed = False # Reset mapping confirmation for new file
    st.session_state.user_column_mapping = None
    st.session_state.initial_mapping_override_for_ui = None # Clear any previous override
    st.session_state.last_processed_file_id_for_mapping_ui = None # Ensure mapper re-initializes
    st.session_state.processed_data = None # Clear old processed data
    logger.info(f"User selected existing file ID: {selected_file_id_from_dropdown}. Triggering load.")
    st.rerun()

if selected_file_id_from_dropdown != "upload_new" and selected_file_id_from_dropdown is not None:
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_dropdown}_v8_refactored"):
        if data_service.delete_user_file(selected_file_id_from_dropdown, current_user_id, permanent_delete_local_file=True): # Set to True for actual deletion
            st.sidebar.success(f"File '{selected_file_label}' deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_from_dropdown:
                # Clear all data related to the deleted file
                keys_to_reset = [
                    'selected_user_file_id', 'uploaded_file_name', 'current_file_content_for_processing',
                    'user_column_mapping', 'column_mapping_confirmed', 'initial_mapping_override_for_ui',
                    'last_processed_file_id_for_mapping_ui', 'processed_data', 'filtered_data',
                    'kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns',
                    'max_drawdown_period_details', 'last_analysis_run_signature'
                ]
                for k_del in keys_to_reset:
                    if k_del in st.session_state: del st.session_state[k_del]
            st.rerun()
        else:
            st.sidebar.error("Failed to delete file.")

# Instantiate SidebarManager and render other controls
# Pass None for processed_data initially if it's not ready, or pass the actual data if available
# The SidebarManager itself should handle None gracefully for symbol/strategy dropdowns
sidebar_manager_instance = SidebarManager(st.session_state.get('processed_data'))
sidebar_filter_values = sidebar_manager_instance.render_sidebar_controls()

# Update session state based on sidebar filter values
# This ensures that changes in sidebar controls trigger a state update and potential re-analysis
# Only update if there's a change to avoid unnecessary reruns if possible, though Streamlit handles this well.
if sidebar_filter_values:
    if st.session_state.risk_free_rate != sidebar_filter_values.get('risk_free_rate'):
        st.session_state.risk_free_rate = sidebar_filter_values.get('risk_free_rate')
        st.session_state.last_analysis_run_signature = None # Invalidate analysis
    if st.session_state.selected_benchmark_ticker != sidebar_filter_values.get('selected_benchmark_ticker'):
        st.session_state.selected_benchmark_ticker = sidebar_filter_values.get('selected_benchmark_ticker')
        st.session_state.selected_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == st.session_state.selected_benchmark_ticker), "None")
        st.session_state.last_analysis_run_signature = None # Invalidate analysis
    if st.session_state.initial_capital != sidebar_filter_values.get('initial_capital'):
        st.session_state.initial_capital = sidebar_filter_values.get('initial_capital')
        st.session_state.last_analysis_run_signature = None # Invalidate analysis
    
    # For complex filters like date range, symbol, strategy, store them directly
    st.session_state.global_date_filter_range = sidebar_filter_values.get('selected_date_range')
    st.session_state.global_symbol_filter = sidebar_filter_values.get('selected_symbol')
    st.session_state.global_strategy_filter = sidebar_filter_values.get('selected_strategy')
    # A change in these filters should also invalidate the main analysis signature
    # This is handled by how `current_analysis_signature` is constructed below.


# --- Data Processing Lifecycle ---
# 1. Save newly uploaded file (if triggered)
if st.session_state.trigger_file_save_processing and st.session_state.pending_file_to_save_content is not None:
    file_bytes = st.session_state.pending_file_to_save_content
    original_name = st.session_state.pending_file_to_save_name
    
    # Reset triggers
    st.session_state.trigger_file_save_processing = False
    st.session_state.pending_file_to_save_content = None
    st.session_state.pending_file_to_save_name = None
    
    temp_file_obj = BytesIO(file_bytes)
    temp_file_obj.name = original_name # Assign name for DataService

    with st.spinner(f"Saving '{original_name}'..."):
        saved_file_record = data_service.save_user_file(current_user_id, temp_file_obj)
    if saved_file_record:
        st.session_state.selected_user_file_id = saved_file_record.id
        st.session_state.trigger_file_load_id = saved_file_record.id # Trigger loading of this newly saved file
        # Reset states for the new file
        st.session_state.column_mapping_confirmed = False
        st.session_state.user_column_mapping = None
        st.session_state.initial_mapping_override_for_ui = None
        st.session_state.last_processed_file_id_for_mapping_ui = None
        st.session_state.processed_data = None
        logger.info(f"File '{original_name}' saved (ID: {saved_file_record.id}). Triggering load.")
        st.rerun()
    else:
        display_custom_message(f"Failed to save file '{original_name}'.", "error")

# 2. Load file content (if triggered)
if st.session_state.trigger_file_load_id is not None:
    file_id_to_load = st.session_state.trigger_file_load_id
    st.session_state.trigger_file_load_id = None # Reset trigger

    with st.spinner(f"Loading file ID {file_id_to_load}..."):
        file_content_bytesio = data_service.get_user_file_content(file_id_to_load, current_user_id)
        user_file_record = data_service.get_user_file_record_by_id(file_id_to_load, current_user_id) # Assumes this method exists

    if file_content_bytesio and user_file_record:
        st.session_state.current_file_content_for_processing = file_content_bytesio
        st.session_state.uploaded_file_name = user_file_record.original_file_name
        st.session_state.selected_user_file_id = file_id_to_load # Ensure this is set
        # Reset further processing states
        st.session_state.column_mapping_confirmed = False
        st.session_state.user_column_mapping = None
        st.session_state.initial_mapping_override_for_ui = None # Crucial for ColumnMapperUI
        st.session_state.processed_data = None
        logger.info(f"File '{user_file_record.original_file_name}' (ID: {file_id_to_load}) content loaded.")
        st.rerun()
    elif file_id_to_load is not None : # Only show error if a load was actually attempted
        display_custom_message(f"Failed to load content for file ID {file_id_to_load}.", "error")
        st.session_state.selected_user_file_id = None # Clear selection if load failed
        st.session_state.uploaded_file_name = None
        st.session_state.current_file_content_for_processing = None


# 3. Column Mapping UI (if file content is loaded but mapping not confirmed)
if st.session_state.current_file_content_for_processing and \
   not st.session_state.column_mapping_confirmed and \
   st.session_state.selected_user_file_id and \
   st.session_state.selected_user_file_id != "upload_new":

    current_file_id_for_mapper = st.session_state.selected_user_file_id
    
    # Load saved mapping only once per file selection for the UI
    if st.session_state.last_processed_file_id_for_mapping_ui != current_file_id_for_mapper:
        with st.spinner("Loading saved column mapping if available..."):
            st.session_state.initial_mapping_override_for_ui = data_service.get_user_column_mapping(
                current_user_id, current_file_id_for_mapper
            )
        st.session_state.last_processed_file_id_for_mapping_ui = current_file_id_for_mapper
        if st.session_state.initial_mapping_override_for_ui:
            logger.info(f"Loaded saved mapping for file ID {current_file_id_for_mapper} to be used as initial override.")
        else:
            logger.info(f"No saved mapping found for file ID {current_file_id_for_mapper}. ColumnMapperUI will use auto-detection.")


    # Prepare headers for ColumnMapperUI
    csv_headers_for_ui = []
    if st.session_state.current_file_content_for_processing:
        st.session_state.current_file_content_for_processing.seek(0)
        try:
            # Robust header reading
            temp_df_for_headers = pd.read_csv(st.session_state.current_file_content_for_processing, nrows=0) # Read only headers
            csv_headers_for_ui = temp_df_for_headers.columns.tolist()
            st.session_state.current_file_content_for_processing.seek(0) # Reset pointer
        except Exception as e_header_read:
            logger.error(f"Could not read headers for ColumnMapperUI: {e_header_read}", exc_info=True)
            display_custom_message(f"Error reading CSV headers: {e_header_read}. Please ensure the file is a valid CSV.", "error")
            st.stop() # Stop if headers can't be read

    if not csv_headers_for_ui:
        display_custom_message("Could not extract headers from the uploaded CSV file. Cannot proceed with column mapping.", "error")
    else:
        column_mapper = ColumnMapperUI(
            uploaded_file_name=st.session_state.uploaded_file_name,
            uploaded_file_bytes=st.session_state.current_file_content_for_processing, # Pass the BytesIO
            csv_headers=csv_headers_for_ui,
            conceptual_columns_map=CONCEPTUAL_COLUMNS,
            conceptual_column_types=CONCEPTUAL_COLUMN_TYPES,
            conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS,
            critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS,
            conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES,
            initial_mapping_override=st.session_state.initial_mapping_override_for_ui
        )
        user_mapping_result = column_mapper.render()

        if user_mapping_result:
            if data_service.save_user_column_mapping(current_user_id, current_file_id_for_mapper, user_mapping_result):
                logger.info(f"Column mapping saved successfully for file ID {current_file_id_for_mapper}.")
            else:
                display_custom_message("Error: Could not save your column mapping preferences to the database.", "error")
            
            st.session_state.user_column_mapping = user_mapping_result
            st.session_state.column_mapping_confirmed = True
            st.session_state.processed_data = None # Force re-processing with new mapping
            st.session_state.last_analysis_run_signature = None # Invalidate analysis
            logger.info("Column mapping confirmed by user. Triggering data processing.")
            st.rerun()
        else:
            # This message is good if the mapper is actively displayed and awaiting confirmation
            display_custom_message("Please complete and confirm column mapping to proceed with data analysis.", "info", icon="⚙️")
            st.stop() # Stop here if mapping is not yet confirmed

# 4. Main Data Processing and Analysis (if file and mapping are ready)
# This block runs if a file is selected, content is loaded, AND mapping is confirmed.
if st.session_state.current_file_content_for_processing and \
   st.session_state.user_column_mapping and \
   st.session_state.column_mapping_confirmed:

    # Construct a signature for the current analysis inputs
    # This helps decide if a full re-analysis is needed
    current_analysis_inputs_signature = (
        st.session_state.selected_user_file_id,
        # Hash of user_column_mapping (convert to sorted tuple of items for consistent hashing)
        tuple(sorted(st.session_state.user_column_mapping.items())) if st.session_state.user_column_mapping else None,
        st.session_state.risk_free_rate,
        st.session_state.selected_benchmark_ticker,
        st.session_state.initial_capital,
        # Hash of filter values (convert dicts/lists to sorted tuples)
        tuple(sorted(st.session_state.global_date_filter_range)) if st.session_state.global_date_filter_range else None,
        st.session_state.global_symbol_filter,
        st.session_state.global_strategy_filter
    )
    
    # Check if analysis needs to be re-run
    if st.session_state.last_analysis_run_signature != current_analysis_inputs_signature or \
       st.session_state.processed_data is None: # Or if processed_data is missing for any reason

        logger.info("Analysis inputs changed or processed_data is missing. Re-running full analysis.")
        with st.spinner("🚀 Initializing analysis with new data/settings..."):
            # (a) Get processed_data (feature engineered)
            st.session_state.current_file_content_for_processing.seek(0) # Ensure stream is at start
            processed_df_result = data_service.get_processed_trading_data(
                st.session_state.current_file_content_for_processing,
                st.session_state.user_column_mapping,
                st.session_state.uploaded_file_name
            )
            if processed_df_result is None or processed_df_result.empty:
                display_custom_message(f"Failed to process trading data from '{st.session_state.uploaded_file_name}'. The file might be empty after processing or an error occurred.", "error")
                st.session_state.processed_data = pd.DataFrame() # Set to empty to avoid errors
                st.session_state.filtered_data = pd.DataFrame()
                st.session_state.kpi_results = {"error": "Data processing failed."}
                st.stop()
            else:
                st.session_state.processed_data = processed_df_result
                logger.info(f"Successfully processed data for '{st.session_state.uploaded_file_name}'. Shape: {st.session_state.processed_data.shape}")

            # (b) Apply global filters to get filtered_data
            # These filters are now directly from session_state, updated by SidebarManager
            filters_for_analysis = {
                'selected_date_range': st.session_state.global_date_filter_range,
                'selected_symbol': st.session_state.global_symbol_filter,
                'selected_strategy': st.session_state.global_strategy_filter
            }
            st.session_state.filtered_data = data_service.filter_data(
                st.session_state.processed_data,
                filters_for_analysis,
                EXPECTED_COLUMNS # Pass the map of conceptual_key -> actual_col_name
            )
            if st.session_state.filtered_data.empty:
                logger.info("Filtered data is empty. Some analyses might not run.")
                # display_custom_message("No data matches the current filter criteria.", "info") # Pages will handle this

            # (c) Perform core analysis if filtered_data is available
            if not st.session_state.filtered_data.empty:
                date_col_for_analysis = EXPECTED_COLUMNS.get('date')
                min_date_str, max_date_str = None, None
                if date_col_for_analysis and date_col_for_analysis in st.session_state.filtered_data.columns:
                    try:
                        valid_dates = pd.to_datetime(st.session_state.filtered_data[date_col_for_analysis], errors='coerce').dropna()
                        if not valid_dates.empty:
                            min_d, max_d = valid_dates.min(), valid_dates.max()
                            if pd.notna(min_d) and pd.notna(max_d):
                                min_date_str, max_date_str = min_d.strftime('%Y-%m-%d'), max_d.strftime('%Y-%m-%d')
                    except Exception as e_date_range:
                        logger.error(f"Error determining date range for benchmark: {e_date_range}")

                if st.session_state.selected_benchmark_ticker and st.session_state.selected_benchmark_ticker.upper() != "NONE" and min_date_str and max_date_str:
                    st.session_state.benchmark_daily_returns = get_benchmark_data_static(
                        st.session_state.selected_benchmark_ticker, min_date_str, max_date_str
                    )
                    if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty:
                        logger.warning(f"Could not fetch benchmark data for {st.session_state.selected_benchmark_ticker}.")
                else:
                    st.session_state.benchmark_daily_returns = None
                
                # Calculate Core KPIs
                st.session_state.kpi_results = analysis_service.get_core_kpis(
                    st.session_state.filtered_data,
                    st.session_state.risk_free_rate,
                    st.session_state.benchmark_daily_returns,
                    st.session_state.initial_capital
                )
                
                # Calculate Bootstrap CIs (example for a few key KPIs)
                # This could be expanded or made configurable
                kpis_for_ci = ['avg_trade_pnl', 'win_rate'] # Add more as needed
                st.session_state.kpi_confidence_intervals = analysis_service.get_bootstrapped_kpi_cis(
                    st.session_state.filtered_data, kpis_to_bootstrap=kpis_for_ci
                )

                # Advanced Drawdown Analysis
                equity_curve_series = pd.Series(dtype=float)
                cum_pnl_col_name = 'cumulative_pnl' # This is an engineered column from data_processing
                if cum_pnl_col_name in st.session_state.filtered_data.columns and \
                   date_col_for_analysis and date_col_for_analysis in st.session_state.filtered_data.columns:
                    
                    temp_equity_df = st.session_state.filtered_data[[date_col_for_analysis, cum_pnl_col_name]].copy()
                    temp_equity_df[date_col_for_analysis] = pd.to_datetime(temp_equity_df[date_col_for_analysis], errors='coerce')
                    temp_equity_df.dropna(subset=[date_col_for_analysis, cum_pnl_col_name], inplace=True)
                    if not temp_equity_df.empty:
                        equity_curve_series = temp_equity_df.set_index(date_col_for_analysis)[cum_pnl_col_name].sort_index()
                
                if not equity_curve_series.empty:
                    adv_dd_results = analysis_service.get_advanced_drawdown_analysis(equity_curve_series)
                    if adv_dd_results and 'error' not in adv_dd_results:
                        st.session_state.max_drawdown_period_details = adv_dd_results.get('max_drawdown_details')
                        # Store full drawdown periods table if needed by pages
                        st.session_state.all_drawdown_periods = adv_dd_results.get('drawdown_periods')
                    else:
                        st.session_state.max_drawdown_period_details = None
                        st.session_state.all_drawdown_periods = None
                        logger.warning(f"Advanced drawdown analysis failed: {adv_dd_results.get('error') if adv_dd_results else 'No result'}")
                else:
                    st.session_state.max_drawdown_period_details = None
                    st.session_state.all_drawdown_periods = None
                    logger.info("Equity curve series for advanced drawdown analysis is empty.")
            else: # If filtered_data is empty
                st.session_state.kpi_results = {"info": "No data matches filters."}
                st.session_state.benchmark_daily_returns = None
                st.session_state.kpi_confidence_intervals = {}
                st.session_state.max_drawdown_period_details = None
                st.session_state.all_drawdown_periods = None

            # Update the signature after successful analysis run
            st.session_state.last_analysis_run_signature = current_analysis_inputs_signature
            logger.info("Global analysis re-run complete. Session state updated.")
            # No st.rerun() here, let Streamlit flow to page rendering naturally
    
    elif not st.session_state.current_file_content_for_processing:
        # This is the state where user is logged in, but no file is selected/loaded yet.
        # The default page (e.g., User Guide or Overview) will handle its display.
        # Or display a global welcome message here if no page is selected.
        if st.session_state.get('current_page_streamlit', 'pages/0_❓_User_Guide.py') == 'app.py' or not os.path.exists(st.session_state.get('current_page_streamlit','')): # Heuristic for main app context
            st.markdown("<div class='main-content-placeholder'>", unsafe_allow_html=True)
            st.image(LOGO_PATH_SIDEBAR if LOGO_PATH_SIDEBAR and os.path.exists(LOGO_PATH_SIDEBAR) else "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png", width=200) # Fallback image
            st.markdown(f"## Welcome to {APP_TITLE}!")
            st.markdown("Please select a trading journal from the sidebar or upload a new one to begin your analysis.")
            st.page_link("pages/0_❓_User_Guide.py", label="Read the User Guide", icon="❓")
            st.markdown("</div>", unsafe_allow_html=True)


# --- Render Scroll Buttons (globally available) ---
scroll_buttons = ScrollButtons()
scroll_buttons.render()

logger.info(f"App '{APP_TITLE}' run cycle finished. Current page: {st.session_state.get('current_page_streamlit', 'N/A')}")
# The actual page content from the `pages/` directory will be rendered by Streamlit's multipage app mechanism after this script.
