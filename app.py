# app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime
import base64 # For logo encoding if needed by st.logo
from io import BytesIO
from typing import Dict, Any, Optional

# --- Configuration and Utility Imports ---
try:
    from config import (
        APP_TITLE, LOGO_PATH_FOR_BROWSER_TAB, LOGO_PATH_SIDEBAR, # Added LOGO_PATH_SIDEBAR
        RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS,
        LOG_FILE, LOG_LEVEL, LOG_FORMAT,
        CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS,
        CONCEPTUAL_COLUMN_CATEGORIES, EXPECTED_COLUMNS, APP_BASE_URL
    )
    from kpi_definitions import KPI_CONFIG
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message # Removed log_execution_time as it's a decorator
    from components import ( # Import from components package
        SidebarManager, ColumnMapperUI, ScrollButtons
    )
    # AuthUI components would be in a separate components/auth_ui.py
    # from components.auth_ui import (
    # display_login_form, display_registration_form,
    # display_forgot_password_request_form, display_reset_password_form
    # )
    from services import ( # Import from services package
        DataService, AnalysisService, AuthService,
        create_db_tables, get_benchmark_data_static
    )
except ImportError as e:
    critical_error_msg = f"Fatal Error: A critical module could not be imported. App cannot start. Details: {e}. Check PYTHONPATH and file locations."
    try:
        import streamlit as st_fallback
        st_fallback.set_page_config(page_title="App Load Error", layout="centered")
        st_fallback.error(critical_error_msg)
        logging.basicConfig(level=logging.ERROR) # Basic logging for this critical error
        logging.critical(critical_error_msg, exc_info=True)
        st_fallback.stop()
    except Exception: # Fallback if even Streamlit can't be imported
        print(critical_error_msg, file=sys.stderr)
        sys.exit(1)


# --- 1. Initial Setup ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB, # Use variable from config
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

logger = setup_logger(
    logger_name=APP_TITLE,
    log_file=LOG_FILE,
    level=LOG_LEVEL,
    log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting up.")

# Load CSS
try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path):
        load_css(css_file_path)
        logger.info("Successfully loaded style.css.")
    else:
        logger.error(f"style.css not found at '{os.path.abspath(css_file_path)}'. Custom styles may not apply.")
except Exception as e_css:
    logger.error(f"Failed to load style.css: {e_css}", exc_info=True)

# Initialize Database Tables
try:
    create_db_tables()
    logger.info("Database tables checked/created successfully.")
except Exception as e_db_init:
    logger.critical(f"Failed to initialize database tables: {e_db_init}", exc_info=True)
    st.error(f"Database Initialization Error: {e_db_init}. The application may not function correctly.")
    # Consider st.stop() if DB is essential for all operations

# Initialize Services
auth_service = AuthService()
data_service = DataService()
analysis_service = AnalysisService()


# --- 2. Session State Initialization ---
def init_session_state_key(key: str, default_value: Any):
    if key not in st.session_state:
        st.session_state[key] = default_value

init_session_state_key('authenticated_user', None)
init_session_state_key('auth_flow_page', 'login')
init_session_state_key('password_reset_token', None)
init_session_state_key('user_preferences', {})
init_session_state_key('current_theme', st.session_state.user_preferences.get('default_theme', 'dark'))
init_session_state_key('selected_user_file_id', None)
init_session_state_key('uploaded_file_name', None)
init_session_state_key('current_file_content_for_processing', None)
init_session_state_key('user_column_mapping', None)
init_session_state_key('column_mapping_confirmed', False)
init_session_state_key('last_processed_file_id_for_mapping_ui', None)
init_session_state_key('initial_mapping_override_for_ui', None)
init_session_state_key('processed_data', None)
init_session_state_key('filtered_data', None)
init_session_state_key('kpi_results', None)
init_session_state_key('kpi_confidence_intervals', {})
init_session_state_key('benchmark_daily_returns', None)
init_session_state_key('max_drawdown_period_details', None)
init_session_state_key('all_drawdown_periods', None) # For advanced drawdown table
init_session_state_key('last_analysis_run_signature', None)
init_session_state_key('risk_free_rate', st.session_state.user_preferences.get('default_risk_free_rate', RISK_FREE_RATE))
init_session_state_key('selected_benchmark_ticker', st.session_state.user_preferences.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER))
init_session_state_key('selected_benchmark_display_name', next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == st.session_state.selected_benchmark_ticker), "None"))
init_session_state_key('initial_capital', 100000.0)
init_session_state_key('global_date_filter_range', None)
init_session_state_key('global_symbol_filter', "All")
init_session_state_key('global_strategy_filter', "All")
init_session_state_key('trigger_file_save_processing', False)
init_session_state_key('pending_file_to_save_content', None)
init_session_state_key('pending_file_to_save_name', None)
init_session_state_key('trigger_file_load_id', None)
init_session_state_key('last_uploaded_raw_file_id_tracker', None)

# --- 3. Theme Management ---
effective_theme = st.session_state.current_theme
theme_js = f"""
<script>
    const rootHtml = document.documentElement;
    const targetTheme = '{effective_theme}';
    if (rootHtml.getAttribute('data-theme') !== targetTheme) {{
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


# --- 4. Authentication Flow UI (Placeholder - to be moved to components/auth_ui.py) ---
# For brevity, these are direct implementations. Ideally, import from components.auth_ui
def display_login_form():
    with st.container(): # Main container for centering
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True) # Inner container with border
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
            with st.form("login_form_main_v6"):
                username = st.text_input("Username", key="login_username_v6", placeholder="Enter your username")
                password = st.text_input("Password", type="password", key="login_password_v6", placeholder="Enter your password")
                submitted = st.form_submit_button("🔑 Login", use_container_width=True, type="primary")

                if submitted:
                    if not username or not password:
                        display_custom_message("Username and password are required.", "error")
                    else:
                        user = auth_service.authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated_user = {'user_id': user.id, 'username': user.username}
                            st.session_state.auth_flow_page = None
                            user_settings = auth_service.get_user_settings(user.id)
                            if user_settings:
                                st.session_state.user_preferences = {
                                    'default_theme': user_settings.default_theme,
                                    'default_risk_free_rate': user_settings.default_risk_free_rate,
                                    'default_benchmark_ticker': user_settings.default_benchmark_ticker
                                }
                                if st.session_state.current_theme != user_settings.default_theme:
                                    st.session_state.current_theme = user_settings.default_theme
                                st.session_state.risk_free_rate = user_settings.default_risk_free_rate
                                st.session_state.selected_benchmark_ticker = user_settings.default_benchmark_ticker
                                st.session_state.selected_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == user_settings.default_benchmark_ticker), "None")
                            else:
                                st.session_state.user_preferences = {'default_theme': 'dark', 'default_risk_free_rate': RISK_FREE_RATE, 'default_benchmark_ticker': DEFAULT_BENCHMARK_TICKER}
                            st.success(f"Welcome back, {username}!"); st.rerun()
                        else:
                            display_custom_message("Invalid username or password.", "error")

            col_auth_links1, col_auth_links2 = st.columns(2)
            with col_auth_links1:
                if st.button("🔑 Forgot Password?", use_container_width=True, key="forgot_password_link_v6"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.rerun()
            with col_auth_links2:
                if st.button("➕ Register New Account", use_container_width=True, key="goto_register_btn_v6"):
                    st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Register for {APP_TITLE}</h2>", unsafe_allow_html=True)
            st.markdown("""
            <small style='color: var(--text-muted-color);'>Password must:
            <ul>
                <li>Be at least 8 characters long</li>
                <li>Contain at least one uppercase letter (A-Z)</li>
                <li>Contain at least one lowercase letter (a-z)</li>
                <li>Contain at least one digit (0-9)</li>
                <li>Contain at least one special character (e.g., !@#$%^&*)</li>
            </ul></small>
            """, unsafe_allow_html=True)
            with st.form("registration_form_v6"):
                reg_username = st.text_input("Username", key="reg_username_v6")
                reg_email = st.text_input("Email (Required for password reset)", key="reg_email_v6")
                reg_password = st.text_input("Password", type="password", key="reg_password_v6")
                reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm_v6")
                reg_submitted = st.form_submit_button("➕ Register Account", use_container_width=True, type="primary")
                if reg_submitted:
                    if not reg_username or not reg_password or not reg_password_confirm or not reg_email:
                        display_custom_message("Username, Email, password, and confirmation are required.", "error")
                    elif reg_password != reg_password_confirm:
                        display_custom_message("Passwords do not match.", "error")
                    else:
                        registration_result = auth_service.register_user(reg_username, reg_password, reg_email)
                        if registration_result.get("user"):
                            display_custom_message(f"User '{reg_username}' registered successfully! Please login.", "success")
                            st.session_state.auth_flow_page = 'login'; st.rerun()
                        else:
                            display_custom_message(registration_result.get("error", "Registration failed due to an unknown server error."), "error")
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_from_reg_v6"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_forgot_password_request_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Forgot Password</h2>", unsafe_allow_html=True)
            st.write("Enter your email address below. If an account with this email exists, a password reset link will be sent to you.")
            with st.form("forgot_password_request_form_v6"):
                email_input = st.text_input("Your Email Address", key="forgot_pw_email_v6", placeholder="you@example.com")
                submit_request = st.form_submit_button("📧 Send Reset Link", use_container_width=True, type="primary")
                if submit_request:
                    if not email_input:
                        display_custom_message("Please enter your email address.", "error")
                    else:
                        with st.spinner("Processing request..."):
                            result = auth_service.create_password_reset_token(email_input)
                        # Display generic success regardless of email existence for security
                        display_custom_message(result.get("success", "If an account with this email exists, a password reset link has been sent."), "success" if result.get("success") else "info")
                        if result.get("error"): # Log internal errors but don't reveal to user
                            logger.error(f"Forgot password error (user-facing msg was generic): {result.get('error')}")
            if st.button("Back to Login", use_container_width=True, key="forgot_pw_back_to_login_v6"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_reset_password_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Reset Your Password</h2>", unsafe_allow_html=True)
            token = st.session_state.get('password_reset_token')
            if not token:
                display_custom_message("Invalid or missing password reset token. Please request a new link via the 'Forgot Password' option.", "error")
                if st.button("Request New Link", use_container_width=True, key="reset_form_req_new_link_v6"):
                    st.session_state.auth_flow_page = 'forgot_password_request'
                    st.session_state.password_reset_token = None
                    st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True); return

            st.markdown("""<small style='color: var(--text-muted-color);'>New password must meet complexity rules (8+ chars, upper, lower, digit, special).</small>""", unsafe_allow_html=True)
            with st.form("reset_password_form_v6"):
                new_password = st.text_input("New Password", type="password", key="reset_pw_new_v6")
                confirm_new_password = st.text_input("Confirm New Password", type="password", key="reset_pw_confirm_v6")
                submit_reset = st.form_submit_button("🔑 Reset Password", use_container_width=True, type="primary")
                if submit_reset:
                    if not new_password or not confirm_new_password:
                        display_custom_message("Please enter and confirm your new password.", "error")
                    elif new_password != confirm_new_password:
                        display_custom_message("New passwords do not match.", "error")
                    else:
                        with st.spinner("Resetting your password..."):
                            result = auth_service.reset_password_with_token(token, new_password)
                        if result.get("success"):
                            display_custom_message(result["success"] + " You can now log in with your new password.", "success")
                            st.session_state.auth_flow_page = 'login'
                            st.session_state.password_reset_token = None
                            # Use st.page_link for cleaner navigation if Streamlit version supports it well,
                            # otherwise, a button + rerun is fine.
                            if st.button("Go to Login Page", key="reset_pw_goto_login_success_v6"):
                                st.rerun()
                        else:
                            display_custom_message(result.get("error", "Could not reset password due to a server error."), "error")
            if st.button("Cancel Reset", use_container_width=True, type="secondary", key="reset_pw_cancel_v6"):
                st.session_state.auth_flow_page = 'login'
                st.session_state.password_reset_token = None
                st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)

# Handle password reset token from URL (this logic should run early)
query_params = st.query_params
if "page" in query_params and str(query_params.get("page", "")).strip() == "reset_password_form" and "token" in query_params:
    token_from_url = str(query_params.get("token", "")).strip()
    if isinstance(token_from_url, list): token_from_url = token_from_url[0]

    if token_from_url and st.session_state.auth_flow_page != 'reset_password_form':
        logger.info(f"Password reset token '{token_from_url[:8]}...' received from URL.")
        user_id_from_token_verify = auth_service.verify_password_reset_token(token_from_url)
        if user_id_from_token_verify:
            st.session_state.password_reset_token = token_from_url
            st.session_state.auth_flow_page = 'reset_password_form'
            logger.info(f"Token valid. Switching to reset password form for user ID {user_id_from_token_verify}.")
        else:
            logger.warning(f"Invalid or expired token received from URL: {token_from_url[:8]}...")
            st.session_state.auth_flow_page = 'login'
            display_custom_message("The password reset link is invalid or has expired. Please request a new one.", "error")
        
        st.experimental_set_query_params() # Clear query params
        st.rerun()


if st.session_state.authenticated_user is None:
    st.sidebar.empty()
    if st.session_state.auth_flow_page == 'login': display_login_form()
    elif st.session_state.auth_flow_page == 'register': display_registration_form()
    elif st.session_state.auth_flow_page == 'forgot_password_request': display_forgot_password_request_form()
    elif st.session_state.auth_flow_page == 'reset_password_form': display_reset_password_form()
    else:
        st.session_state.auth_flow_page = 'login'
        display_login_form()
    st.stop()

# --- Authenticated App Logic ---
current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- 5. Main Application Layout & Data Lifecycle ---
# Sidebar Rendering (Logo, User, Logout, Theme, File Management, Filters)
st.sidebar.header(APP_TITLE)
if LOGO_PATH_SIDEBAR and os.path.exists(LOGO_PATH_SIDEBAR):
    try: st.logo(LOGO_PATH_SIDEBAR, icon_image=LOGO_PATH_SIDEBAR)
    except Exception: 
        try: st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')
        except Exception as e_logo: logger.error(f"Error loading sidebar logo: {e_logo}")
elif LOGO_PATH_SIDEBAR:
    logger.warning(f"Sidebar logo path specified but not found: {LOGO_PATH_SIDEBAR}")

st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_button_app_v6", use_container_width=True):
    logger.info(f"User '{current_username}' logging out.")
    keys_to_clear = list(st.session_state.keys())
    for key_logout in keys_to_clear:
        if key_logout not in ['current_theme']: del st.session_state[key_logout]
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None
    st.session_state.user_preferences = {}; st.success("Logged out."); st.rerun()

st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_app_v6", use_container_width=True):
    new_theme = "dark" if st.session_state.current_theme == "light" else "light"
    st.session_state.current_theme = new_theme
    auth_service.update_user_settings(current_user_id, {'default_theme': new_theme})
    st.session_state.user_preferences['default_theme'] = new_theme
    logger.info(f"User {current_user_id} theme updated to {new_theme}."); st.rerun()
st.sidebar.markdown("---")

# File Management
st.sidebar.subheader("📁 Your Trading Journals")
user_files_list = data_service.list_user_files(current_user_id)
file_options_dict = {f"{f.original_file_name} ({f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files_list}
file_options_dict["✨ Upload New File..."] = "upload_new"
default_file_label = st.session_state.get('last_selected_file_label', "✨ Upload New File...")
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    matching_label = next((label for label, id_val in file_options_dict.items() if id_val == st.session_state.selected_user_file_id), None)
    if matching_label: default_file_label = matching_label
else: # If no file is selected, or "upload_new" was last, default to upload
    default_file_label = "✨ Upload New File..."
if default_file_label not in file_options_dict: default_file_label = "✨ Upload New File..." # Final fallback

selected_file_label = st.sidebar.selectbox("Select/Upload Journal:", options=list(file_options_dict.keys()),
                                           index=list(file_options_dict.keys()).index(default_file_label),
                                           key="main_file_selector_v6")
st.session_state.last_selected_file_label = selected_file_label # Store for persistence
selected_file_id_dropdown = file_options_dict.get(selected_file_label)

if selected_file_id_dropdown == "upload_new":
    newly_uploaded_file = st.sidebar.file_uploader("Upload New (CSV)", type=["csv"], key="main_file_uploader_v6")
    if newly_uploaded_file is not None:
        current_raw_id = id(newly_uploaded_file)
        if st.session_state.last_uploaded_raw_file_id_tracker != current_raw_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file.getvalue()
            st.session_state.pending_file_to_save_name = newly_uploaded_file.name
            st.session_state.trigger_file_save_processing = True
            st.session_state.last_uploaded_raw_file_id_tracker = current_raw_id
            st.rerun()
elif selected_file_id_dropdown is not None and st.session_state.selected_user_file_id != selected_file_id_dropdown:
    st.session_state.trigger_file_load_id = selected_file_id_dropdown
    st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
    st.session_state.initial_mapping_override_for_ui = None; st.session_state.last_processed_file_id_for_mapping_ui = None
    st.session_state.processed_data = None; st.rerun()

if selected_file_id_dropdown != "upload_new" and selected_file_id_dropdown is not None:
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label.split(' (')[0]}'", key=f"del_file_{selected_file_id_dropdown}_v6"):
        if data_service.delete_user_file(selected_file_id_dropdown, current_user_id, True):
            st.sidebar.success(f"File '{selected_file_label}' deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_dropdown:
                keys_to_reset = ['selected_user_file_id', 'uploaded_file_name', 'current_file_content_for_processing', 'user_column_mapping', 'column_mapping_confirmed', 'initial_mapping_override_for_ui', 'last_processed_file_id_for_mapping_ui', 'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'all_drawdown_periods', 'last_analysis_run_signature', 'last_selected_file_label']
                for k_del in keys_to_reset:
                    if k_del in st.session_state: del st.session_state[k_del]
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")

# Sidebar Filters (after file management)
sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
sidebar_filters = sidebar_manager.render_sidebar_controls()
if sidebar_filters: # Update global state from sidebar controls
    # This logic ensures that changes in sidebar filters correctly update the session state
    # and can trigger re-analysis by changing the `current_analysis_inputs_signature`.
    changed_filters = False
    if st.session_state.risk_free_rate != sidebar_filters.get('risk_free_rate'):
        st.session_state.risk_free_rate = sidebar_filters.get('risk_free_rate'); changed_filters = True
    if st.session_state.selected_benchmark_ticker != sidebar_filters.get('selected_benchmark_ticker'):
        st.session_state.selected_benchmark_ticker = sidebar_filters.get('selected_benchmark_ticker')
        st.session_state.selected_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == st.session_state.selected_benchmark_ticker), "None")
        changed_filters = True
    if st.session_state.initial_capital != sidebar_filters.get('initial_capital'):
        st.session_state.initial_capital = sidebar_filters.get('initial_capital'); changed_filters = True
    if st.session_state.global_date_filter_range != sidebar_filters.get('selected_date_range'):
        st.session_state.global_date_filter_range = sidebar_filters.get('selected_date_range'); changed_filters = True
    if st.session_state.global_symbol_filter != sidebar_filters.get('selected_symbol'):
        st.session_state.global_symbol_filter = sidebar_filters.get('selected_symbol'); changed_filters = True
    if st.session_state.global_strategy_filter != sidebar_filters.get('selected_strategy'):
        st.session_state.global_strategy_filter = sidebar_filters.get('selected_strategy'); changed_filters = True
    
    if changed_filters:
        st.session_state.last_analysis_run_signature = None # Invalidate analysis to force re-run
        logger.info("Sidebar filters changed, analysis signature invalidated.")
        # A rerun might be implicitly triggered by widget changes, or explicitly if needed here.
        # st.rerun() # Usually not needed if widgets cause rerun, but can ensure state update propagation.

# --- Data Processing Lifecycle ---
# 1. Save File
if st.session_state.trigger_file_save_processing and st.session_state.pending_file_to_save_content:
    # ... (same as before) ...
    file_bytes = st.session_state.pending_file_to_save_content; original_name = st.session_state.pending_file_to_save_name
    st.session_state.trigger_file_save_processing = False; st.session_state.pending_file_to_save_content = None; st.session_state.pending_file_to_save_name = None
    temp_file_obj = BytesIO(file_bytes); temp_file_obj.name = original_name
    with st.spinner(f"Saving '{original_name}'..."): saved_file_record = data_service.save_user_file(current_user_id, temp_file_obj)
    if saved_file_record:
        st.session_state.selected_user_file_id = saved_file_record.id; st.session_state.trigger_file_load_id = saved_file_record.id
        st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
        st.session_state.initial_mapping_override_for_ui = None; st.session_state.last_processed_file_id_for_mapping_ui = None
        st.session_state.processed_data = None; logger.info(f"File '{original_name}' saved (ID: {saved_file_record.id}). Triggering load."); st.rerun()
    else: display_custom_message(f"Failed to save file '{original_name}'.", "error")

# 2. Load File Content
if st.session_state.trigger_file_load_id is not None:
    # ... (same as before) ...
    file_id_to_load = st.session_state.trigger_file_load_id; st.session_state.trigger_file_load_id = None
    with st.spinner(f"Loading file ID {file_id_to_load}..."):
        file_content_bytesio = data_service.get_user_file_content(file_id_to_load, current_user_id)
        user_file_record = data_service.get_user_file_record_by_id(file_id_to_load, current_user_id) # Assumed method
    if file_content_bytesio and user_file_record:
        st.session_state.current_file_content_for_processing = file_content_bytesio
        st.session_state.uploaded_file_name = user_file_record.original_file_name
        st.session_state.selected_user_file_id = file_id_to_load
        st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
        st.session_state.initial_mapping_override_for_ui = None; st.session_state.processed_data = None
        logger.info(f"File '{user_file_record.original_file_name}' (ID: {file_id_to_load}) loaded."); st.rerun()
    elif file_id_to_load is not None:
        display_custom_message(f"Failed to load content for file ID {file_id_to_load}.", "error")
        st.session_state.selected_user_file_id = None; st.session_state.uploaded_file_name = None; st.session_state.current_file_content_for_processing = None


# 3. Column Mapping
if st.session_state.current_file_content_for_processing and not st.session_state.column_mapping_confirmed and st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    # ... (same as before, ensuring initial_mapping_override_for_ui is used) ...
    current_file_id_map = st.session_state.selected_user_file_id
    if st.session_state.last_processed_file_id_for_mapping_ui != current_file_id_map:
        with st.spinner("Loading saved mapping..."): st.session_state.initial_mapping_override_for_ui = data_service.get_user_column_mapping(current_user_id, current_file_id_map)
        st.session_state.last_processed_file_id_for_mapping_ui = current_file_id_map
    csv_headers_ui = []
    st.session_state.current_file_content_for_processing.seek(0)
    try: csv_headers_ui = pd.read_csv(st.session_state.current_file_content_for_processing, nrows=0).columns.tolist(); st.session_state.current_file_content_for_processing.seek(0)
    except Exception as e: display_custom_message(f"Error reading CSV headers: {e}", "error"); st.stop()
    if not csv_headers_ui: display_custom_message("Could not extract CSV headers.", "error"); st.stop()
    
    # Ensure ColumnMapperUI is instantiated in the main content area, not sidebar
    with st.container(): # Or st.empty() if you want to replace it after mapping
        column_mapper = ColumnMapperUI(st.session_state.uploaded_file_name, st.session_state.current_file_content_for_processing, csv_headers_ui, CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS, CRITICAL_CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_CATEGORIES, st.session_state.initial_mapping_override_for_ui)
        mapping_result = column_mapper.render()
        if mapping_result:
            if data_service.save_user_column_mapping(current_user_id, current_file_id_map, mapping_result): logger.info("Mapping saved.")
            else: display_custom_message("Error saving mapping.", "error")
            st.session_state.user_column_mapping = mapping_result; st.session_state.column_mapping_confirmed = True
            st.session_state.processed_data = None; st.session_state.last_analysis_run_signature = None; st.rerun()
        else: display_custom_message("Please complete column mapping.", "info", icon="⚙️"); st.stop()


# 4. Main Analysis
if st.session_state.current_file_content_for_processing and st.session_state.user_column_mapping and st.session_state.column_mapping_confirmed:
    # ... (same as before, using AnalysisService for orchestration) ...
    current_sig_parts = (st.session_state.selected_user_file_id, tuple(sorted(st.session_state.user_column_mapping.items())) if st.session_state.user_column_mapping else None, st.session_state.risk_free_rate, st.session_state.selected_benchmark_ticker, st.session_state.initial_capital, tuple(sorted(st.session_state.global_date_filter_range)) if st.session_state.global_date_filter_range else None, st.session_state.global_symbol_filter, st.session_state.global_strategy_filter)
    current_analysis_signature = hash(current_sig_parts)

    if st.session_state.last_analysis_run_signature != current_analysis_signature or st.session_state.processed_data is None:
        logger.info("Analysis inputs changed or data not processed. Re-running analysis.")
        with st.spinner("🔬 Performing comprehensive analysis... Please wait."):
            analysis_results_package = analysis_service.get_full_analysis_package(
                user_file_content=st.session_state.current_file_content_for_processing,
                user_column_mapping=st.session_state.user_column_mapping,
                original_file_name=st.session_state.uploaded_file_name,
                filters={
                    'selected_date_range': st.session_state.global_date_filter_range,
                    'selected_symbol': st.session_state.global_symbol_filter,
                    'selected_strategy': st.session_state.global_strategy_filter
                },
                risk_free_rate=st.session_state.risk_free_rate,
                benchmark_ticker=st.session_state.selected_benchmark_ticker,
                initial_capital=st.session_state.initial_capital
            ) # This method needs to be implemented in AnalysisService

        if analysis_results_package and 'error' not in analysis_results_package:
            st.session_state.processed_data = analysis_results_package.get('processed_data')
            st.session_state.filtered_data = analysis_results_package.get('filtered_data')
            st.session_state.kpi_results = analysis_results_package.get('kpi_results')
            st.session_state.kpi_confidence_intervals = analysis_results_package.get('kpi_confidence_intervals', {})
            st.session_state.benchmark_daily_returns = analysis_results_package.get('benchmark_daily_returns')
            st.session_state.max_drawdown_period_details = analysis_results_package.get('max_drawdown_period_details')
            st.session_state.all_drawdown_periods = analysis_results_package.get('all_drawdown_periods')
            st.session_state.last_analysis_run_signature = current_analysis_signature
            logger.info("Global analysis re-run complete. Session state updated.")
            if st.session_state.processed_data is None or st.session_state.processed_data.empty:
                 display_custom_message("Data processing was successful, but the resulting dataset is empty. This might be due to the nature of the input file or strict processing rules.", "warning")
            if st.session_state.filtered_data is not None and st.session_state.filtered_data.empty and \
               st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
                 display_custom_message("No data matches the current filter criteria. Adjust filters in the sidebar to see results.", "info")

        elif analysis_results_package: # Error from service
            error_msg = analysis_results_package.get('error', "Unknown error during analysis.")
            display_custom_message(f"Analysis Error: {error_msg}", "error")
            # Reset key states to prevent pages from trying to use faulty data
            st.session_state.processed_data = None; st.session_state.filtered_data = None
            st.session_state.kpi_results = {"error": error_msg}
        else: # Service failed to return anything
             display_custom_message("Core analysis service failed to return results.", "error")
             st.session_state.processed_data = None; st.session_state.filtered_data = None
             st.session_state.kpi_results = {"error": "Analysis service did not respond."}
    # If signature matches and data exists, no re-run needed. Pages will use existing session state.

elif not st.session_state.current_file_content_for_processing and st.session_state.authenticated_user:
    # This is the landing state for an authenticated user before any file interaction.
    # The default page (User Guide or Overview) will be shown by Streamlit's multipage logic.
    # We can add a general welcome/instruction message in the main area if no specific page is selected,
    # but typically the default page handles this.
    # Example: if st.get_option(" streamlit.scriptrunner.page_script_hash") is None: # Check if on main app.py page
    if not any(st.session_state.get(key) for key in ['selected_user_file_id', 'current_file_content_for_processing']):
        # This is a simplified check. A more robust way is to see which page is active if Streamlit provides that.
        # For now, if no file is active, and we are on app.py (not a sub-page), show welcome.
        # This part is tricky with multipage apps as app.py always runs.
        # The pages themselves should handle the "no data" state.
        pass # Let the default page (e.g., User Guide) handle this.

# Render Scroll Buttons (globally)
scroll_buttons = ScrollButtons()
scroll_buttons.render()

logger.info(f"App '{APP_TITLE}' run cycle finished.")
# Streamlit automatically renders the selected page from the `pages/` directory after this script.
