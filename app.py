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
        APP_TITLE, LOGO_PATH_FOR_BROWSER_TAB, LOGO_PATH_SIDEBAR,
        RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS,
        LOG_FILE, LOG_LEVEL, LOG_FORMAT,
        CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS,
        CONCEPTUAL_COLUMN_CATEGORIES, EXPECTED_COLUMNS, APP_BASE_URL
    )
    from kpi_definitions import KPI_CONFIG
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message
    from components import (
        SidebarManager, ColumnMapperUI, ScrollButtons
    )
    # from components.auth_ui import ( # Placeholder for future refactoring
    # display_login_form, display_registration_form,
    # display_forgot_password_request_form, display_reset_password_form
    # )
    from services import (
        DataService, AnalysisService, AuthService,
        create_db_tables, get_benchmark_data_static
    )
except ImportError as e:
    critical_error_msg = f"Fatal Error: A critical module could not be imported. App cannot start. Details: {e}. Check PYTHONPATH and file locations."
    try:
        import streamlit as st_fallback
        st_fallback.set_page_config(page_title="App Load Error", layout="centered")
        st_fallback.error(critical_error_msg)
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
    menu_items={}
)

logger = setup_logger(
    logger_name=APP_TITLE,
    log_file=LOG_FILE,
    level=LOG_LEVEL,
    log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting up.")

try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path):
        load_css(css_file_path)
        logger.info("Successfully loaded style.css.")
    else:
        logger.error(f"style.css not found at '{os.path.abspath(css_file_path)}'. Custom styles may not apply.")
except Exception as e_css:
    logger.error(f"Failed to load style.css: {e_css}", exc_info=True)

try:
    create_db_tables()
    logger.info("Database tables checked/created successfully.")
except Exception as e_db_init:
    logger.critical(f"Failed to initialize database tables: {e_db_init}", exc_info=True)
    st.error(f"Database Initialization Error: {e_db_init}. The application may not function correctly.")

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
init_session_state_key('all_drawdown_periods', None)
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
init_session_state_key('last_selected_file_label', "✨ Upload New File...") # For selectbox persistence


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
def display_login_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
            with st.form("login_form_main_v6_app"): # Ensure unique key
                username = st.text_input("Username", key="login_username_v6_app", placeholder="Enter your username")
                password = st.text_input("Password", type="password", key="login_password_v6_app", placeholder="Enter your password")
                submitted = st.form_submit_button("🔑 Login", use_container_width=True, type="primary")
                if submitted:
                    if not username or not password: display_custom_message("Username and password are required.", "error")
                    else:
                        user = auth_service.authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated_user = {'user_id': user.id, 'username': user.username}
                            st.session_state.auth_flow_page = None
                            user_settings = auth_service.get_user_settings(user.id)
                            if user_settings:
                                st.session_state.user_preferences = {'default_theme': user_settings.default_theme, 'default_risk_free_rate': user_settings.default_risk_free_rate, 'default_benchmark_ticker': user_settings.default_benchmark_ticker}
                                if st.session_state.current_theme != user_settings.default_theme: st.session_state.current_theme = user_settings.default_theme
                                st.session_state.risk_free_rate = user_settings.default_risk_free_rate
                                st.session_state.selected_benchmark_ticker = user_settings.default_benchmark_ticker
                                st.session_state.selected_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == user_settings.default_benchmark_ticker), "None")
                            else: st.session_state.user_preferences = {'default_theme': 'dark', 'default_risk_free_rate': RISK_FREE_RATE, 'default_benchmark_ticker': DEFAULT_BENCHMARK_TICKER}
                            st.success(f"Welcome back, {username}!"); st.rerun()
                        else: display_custom_message("Invalid username or password.", "error")
            col_auth_links1, col_auth_links2 = st.columns(2)
            with col_auth_links1:
                if st.button("🔑 Forgot Password?", use_container_width=True, key="forgot_password_link_v6_app"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.rerun()
            with col_auth_links2:
                if st.button("➕ Register New Account", use_container_width=True, key="goto_register_btn_v6_app"):
                    st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Register for {APP_TITLE}</h2>", unsafe_allow_html=True)
            st.markdown("""<small style='color: var(--text-muted-color);'>Password must: <ul><li>Be at least 8 characters long</li><li>Contain at least one uppercase letter (A-Z)</li><li>Contain at least one lowercase letter (a-z)</li><li>Contain at least one digit (0-9)</li><li>Contain at least one special character (e.g., !@#$%^&*)</li></ul></small>""", unsafe_allow_html=True)
            with st.form("registration_form_v6_app"):
                reg_username = st.text_input("Username", key="reg_username_v6_app")
                reg_email = st.text_input("Email (Required for password reset)", key="reg_email_v6_app")
                reg_password = st.text_input("Password", type="password", key="reg_password_v6_app")
                reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm_v6_app")
                reg_submitted = st.form_submit_button("➕ Register Account", use_container_width=True, type="primary")
                if reg_submitted:
                    if not reg_username or not reg_password or not reg_password_confirm or not reg_email: display_custom_message("Username, Email, password, and confirmation are required.", "error")
                    elif reg_password != reg_password_confirm: display_custom_message("Passwords do not match.", "error")
                    else:
                        registration_result = auth_service.register_user(reg_username, reg_password, reg_email)
                        if registration_result.get("user"): display_custom_message(f"User '{reg_username}' registered successfully! Please login.", "success"); st.session_state.auth_flow_page = 'login'; st.rerun()
                        else: display_custom_message(registration_result.get("error", "Registration failed."), "error")
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_from_reg_v6_app"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_forgot_password_request_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh; padding: 2rem;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center; color: var(--text-heading-color);'>Forgot Password</h2>", unsafe_allow_html=True)
            st.write("Enter your email address. If an account exists, a password reset link will be sent.")
            with st.form("forgot_password_request_form_v6_app"):
                email_input = st.text_input("Your Email Address", key="forgot_pw_email_v6_app", placeholder="you@example.com")
                submit_request = st.form_submit_button("📧 Send Reset Link", use_container_width=True, type="primary")
                if submit_request:
                    if not email_input: display_custom_message("Please enter your email address.", "error")
                    else:
                        with st.spinner("Processing request..."): result = auth_service.create_password_reset_token(email_input)
                        display_custom_message(result.get("success", "If an account with this email exists, a password reset link has been sent."), "success" if result.get("success") else "info")
                        if result.get("error"): logger.error(f"Forgot password error: {result.get('error')}")
            if st.button("Back to Login", use_container_width=True, key="forgot_pw_back_to_login_v6_app"):
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
                display_custom_message("Invalid or missing reset token. Please request a new link.", "error")
                if st.button("Request New Link", use_container_width=True, key="reset_form_req_new_link_v6_app"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.session_state.password_reset_token = None; st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True); return
            st.markdown("""<small style='color: var(--text-muted-color);'>New password must meet complexity rules.</small>""", unsafe_allow_html=True)
            with st.form("reset_password_form_v6_app"):
                new_password = st.text_input("New Password", type="password", key="reset_pw_new_v6_app")
                confirm_new_password = st.text_input("Confirm New Password", type="password", key="reset_pw_confirm_v6_app")
                submit_reset = st.form_submit_button("🔑 Reset Password", use_container_width=True, type="primary")
                if submit_reset:
                    if not new_password or not confirm_new_password: display_custom_message("Please enter and confirm your new password.", "error")
                    elif new_password != confirm_new_password: display_custom_message("New passwords do not match.", "error")
                    else:
                        with st.spinner("Resetting password..."): result = auth_service.reset_password_with_token(token, new_password)
                        if result.get("success"):
                            display_custom_message(result["success"] + " You can now log in.", "success")
                            st.session_state.auth_flow_page = 'login'; st.session_state.password_reset_token = None
                            if st.button("Go to Login", key="reset_pw_goto_login_v6_app"): st.rerun()
                        else: display_custom_message(result.get("error", "Could not reset password."), "error")
            if st.button("Cancel", use_container_width=True, type="secondary", key="reset_pw_cancel_v6_app"):
                st.session_state.auth_flow_page = 'login'; st.session_state.password_reset_token = None; st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)

# Handle password reset token from URL
query_params = st.query_params
if "page" in query_params and str(query_params.get("page", "")).strip() == "reset_password_form" and "token" in query_params:
    token_from_url = str(query_params.get("token", "")).strip()
    # Ensure token_from_url is a string, not a list.
    if isinstance(token_from_url, list): token_from_url = token_from_url[0] if token_from_url else ""

    if token_from_url and st.session_state.auth_flow_page != 'reset_password_form':
        logger.info(f"Password reset token '{token_from_url[:8]}...' received from URL.")
        user_id_verified = auth_service.verify_password_reset_token(token_from_url)
        if user_id_verified:
            st.session_state.password_reset_token = token_from_url
            st.session_state.auth_flow_page = 'reset_password_form'
            logger.info(f"Token valid for user ID {user_id_verified}. Switching to reset form.")
        else:
            logger.warning(f"Invalid/expired token from URL: {token_from_url[:8]}...")
            st.session_state.auth_flow_page = 'login' # Default to login
            display_custom_message("Password reset link is invalid or expired.", "error")
        st.experimental_set_query_params() # Clear query params
        st.rerun()

if st.session_state.authenticated_user is None:
    st.sidebar.empty()
    auth_page_map = {
        'login': display_login_form,
        'register': display_registration_form,
        'forgot_password_request': display_forgot_password_request_form,
        'reset_password_form': display_reset_password_form
    }
    auth_function = auth_page_map.get(st.session_state.auth_flow_page, display_login_form)
    auth_function()
    st.stop()

# --- Authenticated App Logic ---
current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- 5. Main Application Layout & Data Lifecycle ---
st.sidebar.header(APP_TITLE)
if LOGO_PATH_SIDEBAR and os.path.exists(LOGO_PATH_SIDEBAR):
    try: st.logo(LOGO_PATH_SIDEBAR, icon_image=LOGO_PATH_SIDEBAR)
    except Exception:
        try: st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')
        except Exception as e_logo: logger.error(f"Error loading sidebar logo: {e_logo}")
elif LOGO_PATH_SIDEBAR: logger.warning(f"Sidebar logo path specified but not found: {LOGO_PATH_SIDEBAR}")

st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_app_v6", use_container_width=True):
    logger.info(f"User '{current_username}' logging out.")
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        if key not in ['current_theme']: del st.session_state[key]
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None
    st.session_state.user_preferences = {}; st.success("Logged out."); st.rerun()

st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_app_v6", use_container_width=True):
    new_theme = "dark" if st.session_state.current_theme == "light" else "light"
    st.session_state.current_theme = new_theme
    auth_service.update_user_settings(current_user_id, {'default_theme': new_theme})
    st.session_state.user_preferences['default_theme'] = new_theme; st.rerun()
st.sidebar.markdown("---")

st.sidebar.subheader("📁 Your Trading Journals")
user_files = data_service.list_user_files(current_user_id)
file_options = {f"{f.original_file_name} ({f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files}
file_options["✨ Upload New File..."] = "upload_new"
default_label = st.session_state.get('last_selected_file_label', "✨ Upload New File...")
if default_label not in file_options: default_label = "✨ Upload New File..."

selected_label = st.sidebar.selectbox("Select/Upload Journal:", options=list(file_options.keys()),
                                      index=list(file_options.keys()).index(default_label), key="main_file_select_v6")
st.session_state.last_selected_file_label = selected_label
selected_id = file_options.get(selected_label)

if selected_id == "upload_new":
    new_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="main_uploader_v6")
    if new_file:
        raw_id = id(new_file)
        if st.session_state.last_uploaded_raw_file_id_tracker != raw_id:
            st.session_state.pending_file_to_save_content = new_file.getvalue()
            st.session_state.pending_file_to_save_name = new_file.name
            st.session_state.trigger_file_save_processing = True
            st.session_state.last_uploaded_raw_file_id_tracker = raw_id; st.rerun()
elif selected_id and st.session_state.selected_user_file_id != selected_id:
    st.session_state.trigger_file_load_id = selected_id
    st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
    st.session_state.initial_mapping_override_for_ui = None; st.session_state.last_processed_file_id_for_mapping_ui = None
    st.session_state.processed_data = None; st.rerun()

if selected_id != "upload_new" and selected_id:
    if st.sidebar.button(f"🗑️ Delete '{selected_label.split(' (')[0]}'", key=f"del_file_{selected_id}_v6"):
        if data_service.delete_user_file(selected_id, current_user_id, True):
            st.sidebar.success(f"File '{selected_label}' deleted.")
            if st.session_state.selected_user_file_id == selected_id:
                keys_to_reset = ['selected_user_file_id', 'uploaded_file_name', 'current_file_content_for_processing', 'user_column_mapping', 'column_mapping_confirmed', 'initial_mapping_override_for_ui', 'last_processed_file_id_for_mapping_ui', 'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'all_drawdown_periods', 'last_analysis_run_signature', 'last_selected_file_label']
                for k in keys_to_reset:
                    if k in st.session_state: del st.session_state[k]
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")

sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
sidebar_filters = sidebar_manager.render_sidebar_controls()
if sidebar_filters:
    changed = False
    if st.session_state.risk_free_rate != sidebar_filters.get('risk_free_rate'):
        st.session_state.risk_free_rate = sidebar_filters.get('risk_free_rate'); changed = True
    if st.session_state.selected_benchmark_ticker != sidebar_filters.get('selected_benchmark_ticker'):
        st.session_state.selected_benchmark_ticker = sidebar_filters.get('selected_benchmark_ticker')
        st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == st.session_state.selected_benchmark_ticker), "None")
        changed = True
    if st.session_state.initial_capital != sidebar_filters.get('initial_capital'):
        st.session_state.initial_capital = sidebar_filters.get('initial_capital'); changed = True
    if st.session_state.global_date_filter_range != sidebar_filters.get('selected_date_range'):
        st.session_state.global_date_filter_range = sidebar_filters.get('selected_date_range'); changed = True
    if st.session_state.global_symbol_filter != sidebar_filters.get('selected_symbol'):
        st.session_state.global_symbol_filter = sidebar_filters.get('selected_symbol'); changed = True
    if st.session_state.global_strategy_filter != sidebar_filters.get('selected_strategy'):
        st.session_state.global_strategy_filter = sidebar_filters.get('selected_strategy'); changed = True
    if changed: st.session_state.last_analysis_run_signature = None; logger.info("Sidebar filters changed.")

# --- Data Processing Lifecycle ---
if st.session_state.trigger_file_save_processing and st.session_state.pending_file_to_save_content:
    file_bytes, original_name = st.session_state.pending_file_to_save_content, st.session_state.pending_file_to_save_name
    st.session_state.trigger_file_save_processing = False; st.session_state.pending_file_to_save_content = None; st.session_state.pending_file_to_save_name = None
    temp_obj = BytesIO(file_bytes); temp_obj.name = original_name
    with st.spinner(f"Saving '{original_name}'..."): record = data_service.save_user_file(current_user_id, temp_obj)
    if record:
        st.session_state.selected_user_file_id = record.id; st.session_state.trigger_file_load_id = record.id
        st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
        st.session_state.initial_mapping_override_for_ui = None; st.session_state.last_processed_file_id_for_mapping_ui = None
        st.session_state.processed_data = None; logger.info(f"File '{original_name}' saved (ID: {record.id})."); st.rerun()
    else: display_custom_message(f"Failed to save file '{original_name}'.", "error")

if st.session_state.trigger_file_load_id:
    file_id = st.session_state.trigger_file_load_id; st.session_state.trigger_file_load_id = None
    with st.spinner(f"Loading file ID {file_id}..."):
        content_io = data_service.get_user_file_content(file_id, current_user_id)
        record = data_service.get_user_file_record_by_id(file_id, current_user_id)
    if content_io and record:
        st.session_state.current_file_content_for_processing = content_io
        st.session_state.uploaded_file_name = record.original_file_name
        st.session_state.selected_user_file_id = file_id
        st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
        st.session_state.initial_mapping_override_for_ui = None; st.session_state.processed_data = None
        logger.info(f"File '{record.original_file_name}' loaded."); st.rerun()
    elif file_id: display_custom_message(f"Failed to load file ID {file_id}.", "error"); st.session_state.selected_user_file_id = None

if st.session_state.current_file_content_for_processing and not st.session_state.column_mapping_confirmed and st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    file_id_map = st.session_state.selected_user_file_id
    if st.session_state.last_processed_file_id_for_mapping_ui != file_id_map:
        with st.spinner("Loading mapping..."): st.session_state.initial_mapping_override_for_ui = data_service.get_user_column_mapping(current_user_id, file_id_map)
        st.session_state.last_processed_file_id_for_mapping_ui = file_id_map
    headers = []
    st.session_state.current_file_content_for_processing.seek(0)
    try: headers = pd.read_csv(st.session_state.current_file_content_for_processing, nrows=0).columns.tolist(); st.session_state.current_file_content_for_processing.seek(0)
    except Exception as e: display_custom_message(f"Error reading headers: {e}", "error"); st.stop()
    if not headers: display_custom_message("No CSV headers.", "error"); st.stop()
    
    with st.container():
        mapper = ColumnMapperUI(st.session_state.uploaded_file_name, st.session_state.current_file_content_for_processing, headers, CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS, CRITICAL_CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_CATEGORIES, st.session_state.initial_mapping_override_for_ui)
        result = mapper.render()
        if result:
            if data_service.save_user_column_mapping(current_user_id, file_id_map, result): logger.info("Mapping saved.")
            else: display_custom_message("Error saving mapping.", "error")
            st.session_state.user_column_mapping = result; st.session_state.column_mapping_confirmed = True
            st.session_state.processed_data = None; st.session_state.last_analysis_run_signature = None; st.rerun()
        else: display_custom_message("Please complete column mapping.", "info", icon="⚙️"); st.stop()

if st.session_state.current_file_content_for_processing and st.session_state.user_column_mapping and st.session_state.column_mapping_confirmed:
    sig_parts = (st.session_state.selected_user_file_id, tuple(sorted(st.session_state.user_column_mapping.items())) if st.session_state.user_column_mapping else None, st.session_state.risk_free_rate, st.session_state.selected_benchmark_ticker, st.session_state.initial_capital, tuple(sorted(st.session_state.global_date_filter_range)) if st.session_state.global_date_filter_range else None, st.session_state.global_symbol_filter, st.session_state.global_strategy_filter)
    current_sig = hash(sig_parts)
    if st.session_state.last_analysis_run_signature != current_sig or st.session_state.processed_data is None:
        logger.info("Analysis inputs changed or data not processed. Re-running analysis.")
        with st.spinner("🔬 Performing comprehensive analysis..."):
            # This method needs to be implemented in AnalysisService
            # It should take all necessary inputs and return a package of results
            analysis_pkg = analysis_service.get_full_analysis_package(
                user_file_content=st.session_state.current_file_content_for_processing,
                user_column_mapping=st.session_state.user_column_mapping,
                original_file_name=st.session_state.uploaded_file_name,
                filters={'selected_date_range': st.session_state.global_date_filter_range, 'selected_symbol': st.session_state.global_symbol_filter, 'selected_strategy': st.session_state.global_strategy_filter},
                risk_free_rate=st.session_state.risk_free_rate,
                benchmark_ticker=st.session_state.selected_benchmark_ticker,
                initial_capital=st.session_state.initial_capital
            )
        if analysis_pkg and 'error' not in analysis_pkg:
            st.session_state.processed_data = analysis_pkg.get('processed_data')
            st.session_state.filtered_data = analysis_pkg.get('filtered_data')
            st.session_state.kpi_results = analysis_pkg.get('kpi_results')
            st.session_state.kpi_confidence_intervals = analysis_pkg.get('kpi_confidence_intervals', {})
            st.session_state.benchmark_daily_returns = analysis_pkg.get('benchmark_daily_returns')
            st.session_state.max_drawdown_period_details = analysis_pkg.get('max_drawdown_period_details')
            st.session_state.all_drawdown_periods = analysis_pkg.get('all_drawdown_periods')
            st.session_state.last_analysis_run_signature = current_sig
            logger.info("Global analysis re-run complete.")
            if st.session_state.processed_data is None or st.session_state.processed_data.empty: display_custom_message("Data processing resulted in an empty dataset.", "warning")
            if st.session_state.filtered_data is not None and st.session_state.filtered_data.empty and st.session_state.processed_data is not None and not st.session_state.processed_data.empty: display_custom_message("No data matches current filters.", "info")
        elif analysis_pkg:
            display_custom_message(f"Analysis Error: {analysis_pkg.get('error', 'Unknown error')}", "error")
            st.session_state.processed_data = None; st.session_state.filtered_data = None; st.session_state.kpi_results = {"error": analysis_pkg.get('error', 'Unknown error')}
        else:
            display_custom_message("Core analysis service failed.", "error")
            st.session_state.processed_data = None; st.session_state.filtered_data = None; st.session_state.kpi_results = {"error": "Analysis service did not respond."}

elif not st.session_state.current_file_content_for_processing and st.session_state.authenticated_user:
    # Determine if we are on the main app page (not a sub-page)
    # This is a heuristic. A more robust method might involve checking st.get_script_run_ctx().page_script_hash
    # or similar if available and stable across Streamlit versions.
    # For now, if no file is active, assume we might be on a landing state.
    # The individual pages handle their "no data" state, so this is for the main app.py context before a page is selected.
    # This welcome message is more appropriate if app.py itself is the current "page" being rendered.
    # In a multi-page app, Streamlit switches to the selected page from `pages/`.
    # So, this block might only be relevant if no page is selected by default (which is usually User Guide).
    # Let's assume the default page handles its own welcome message if data isn't ready.
    pass

scroll_buttons = ScrollButtons()
scroll_buttons.render()
logger.info(f"App '{APP_TITLE}' run cycle finished.")
