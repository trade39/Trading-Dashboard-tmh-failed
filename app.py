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
        DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS, EXPECTED_COLUMNS,
        APP_BASE_URL
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
    st.error(f"Fatal Error: A critical module could not be imported. App cannot start. Details: {e}")
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Fatal Error during initial imports: {e}", exc_info=True)
    # Fallback definitions
    APP_TITLE="TradingAppError"; RISK_FREE_RATE=0.02; DEFAULT_BENCHMARK_TICKER="SPY"; AVAILABLE_BENCHMARKS={"None":""}; APP_BASE_URL=""
    class DataServiceFallback: pass
    class AnalysisServiceFallback: pass
    class AuthServiceFallback: pass
    DataService = DataServiceFallback
    AnalysisService = AnalysisServiceFallback
    AuthService = AuthServiceFallback
    def create_db_tables(): pass
    def get_benchmark_data_static(*args, **kwargs): return None
    class SidebarManager:
        def __init__(self,*args): pass
        def render_sidebar_controls(self): return {}
    class ColumnMapperUI:
        def __init__(self,*args,**kwargs): pass
        def render(self): return None
    class ScrollButtons:
        def __init__(self,*args,**kwargs): pass
        def render(self): pass
    def load_css(f): pass
    def display_custom_message(m,t="error"): st.error(m) # Ensure default for t
    def setup_logger(**kwargs): return logging.getLogger(APP_TITLE)
    st.stop()


# --- Page Config, Logger, Services, DB Init (as before) ---
PAGE_CONFIG_APP_TITLE = APP_TITLE
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png"
st.set_page_config(page_title=PAGE_CONFIG_APP_TITLE, page_icon=LOGO_PATH_FOR_BROWSER_TAB, layout="wide", initial_sidebar_state="expanded", menu_items={})
logger = setup_logger(logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT)
logger.info(f"Application '{APP_TITLE}' starting.")
data_service = DataService(); analysis_service_instance = AnalysisService(); auth_service = AuthService()
try: create_db_tables(); logger.info("DB tables checked/created.")
except Exception as e: logger.critical(f"Failed to init DB tables: {e}", exc_info=True); st.error(f"DB Init Error: {e}.")

if 'user_preferences' not in st.session_state: st.session_state.user_preferences = {}
effective_theme = st.session_state.user_preferences.get('default_theme', st.session_state.get('current_theme', 'dark'))
if 'current_theme' not in st.session_state or st.session_state.current_theme != effective_theme:
    st.session_state.current_theme = effective_theme
theme_js = f"""<script> const currentTheme = '{st.session_state.current_theme}'; document.documentElement.setAttribute('data-theme', currentTheme);
    if (currentTheme === "dark") {{ document.body.classList.add('dark-mode'); document.body.classList.remove('light-mode'); }}
    else {{ document.body.classList.add('light-mode'); document.body.classList.remove('dark-mode'); }} </script>"""
st.components.v1.html(theme_js, height=0)
try: css_file_path = "style.css";
    if os.path.exists(css_file_path): load_css(css_file_path)
    else: logger.error(f"style.css not found at '{css_file_path}'.")
except Exception as e_css: logger.error(f"Failed to load style.css: {e_css}", exc_info=True)

if 'authenticated_user' not in st.session_state: st.session_state.authenticated_user = None
if 'auth_flow_page' not in st.session_state: st.session_state.auth_flow_page = 'login'
if 'password_reset_token' not in st.session_state: st.session_state.password_reset_token = None

# --- Handle Password Reset Token from URL ---
query_params = st.query_params
if "page" in query_params and query_params.get("page") == "reset_password_form" and "token" in query_params:
    token_from_url = query_params.get("token")
    if isinstance(token_from_url, list): token_from_url = token_from_url[0]

    if token_from_url and st.session_state.auth_flow_page != 'reset_password_form':
        st.session_state.password_reset_token = token_from_url
        st.session_state.auth_flow_page = 'reset_password_form'
        logger.info(f"Password reset token '{token_from_url[:8]}...' received. Switching to reset form.")
        st.query_params.clear() # Clear after processing
        st.rerun()

if 'selected_user_file_id' not in st.session_state: st.session_state.selected_user_file_id = None
if 'current_file_content_for_processing' not in st.session_state: st.session_state.current_file_content_for_processing = None
if 'pending_file_to_save_content' not in st.session_state: st.session_state.pending_file_to_save_content = None
if 'pending_file_to_save_name' not in st.session_state: st.session_state.pending_file_to_save_name = None
if 'trigger_file_save_processing' not in st.session_state: st.session_state.trigger_file_save_processing = False
if 'last_uploaded_raw_file_id_tracker' not in st.session_state: st.session_state.last_uploaded_raw_file_id_tracker = None
if 'trigger_file_load_id' not in st.session_state: st.session_state.trigger_file_load_id = None
if 'last_processed_mapping_for_file_id' not in st.session_state: st.session_state.last_processed_mapping_for_file_id = None


def display_login_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("login_form_main_v3_reset"):
                st.markdown(f"<h2 style='text-align: center;'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
                username = st.text_input("Username", key="login_username_main_v3_reset")
                password = st.text_input("Password", type="password", key="login_password_main_v3_reset")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    if not username or not password: st.error("Username and password are required.")
                    else:
                        user = auth_service.authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated_user = {'user_id': user.id, 'username': user.username}
                            st.session_state.auth_flow_page = None
                            user_settings = auth_service.get_user_settings(user.id)
                            if user_settings:
                                st.session_state.user_preferences = {'default_theme': user_settings.default_theme, 'default_risk_free_rate': user_settings.default_risk_free_rate, 'default_benchmark_ticker': user_settings.default_benchmark_ticker}
                                if st.session_state.current_theme != user_settings.default_theme: st.session_state.current_theme = user_settings.default_theme
                            else: st.session_state.user_preferences = {'default_theme': 'dark', 'default_risk_free_rate': RISK_FREE_RATE, 'default_benchmark_ticker': DEFAULT_BENCHMARK_TICKER}
                            st.success(f"Welcome back, {username}!"); st.rerun()
                        else: st.error("Invalid username or password.")
            col_auth_links1, col_auth_links2 = st.columns(2)
            with col_auth_links1:
                if st.button("Forgot Password?", use_container_width=True, key="forgot_password_link_v1"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.rerun()
            with col_auth_links2:
                if st.button("Register New Account", use_container_width=True, key="goto_register_btn_main_v5_reset"):
                    st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
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
            with st.form("registration_form_main_v4_complexity"):
                reg_username = st.text_input("Username", key="reg_username_main_v4")
                reg_email = st.text_input("Email (Required for password reset)", key="reg_email_main_v4")
                reg_password = st.text_input("Password", type="password", key="reg_password_main_v4")
                reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm_main_v4")
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
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn_main_v5_complexity"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_forgot_password_request_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center;'>Forgot Password</h2>", unsafe_allow_html=True)
            st.write("Enter your email. If an account exists, a reset link will be sent.")
            with st.form("forgot_password_request_form_v1"):
                email_input = st.text_input("Your Email Address", key="forgot_pw_email_v1")
                submit_request = st.form_submit_button("Send Reset Link", use_container_width=True, type="primary")
                if submit_request:
                    if not email_input: st.error("Please enter your email address.")
                    else:
                        with st.spinner("Processing..."): result = auth_service.create_password_reset_token(email_input)
                        if result.get("success"): st.success(result["success"])
                        else: st.error(result.get("error", "Could not process request."))
            if st.button("Back to Login", use_container_width=True, key="forgot_pw_back_to_login_v1"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_reset_password_form():
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            st.markdown(f"<h2 style='text-align: center;'>Reset Your Password</h2>", unsafe_allow_html=True)
            token = st.session_state.get('password_reset_token')
            if not token:
                st.error("Invalid or missing reset token. Please request a new link.");
                if st.button("Request New Link",use_container_width=True,key="reset_form_req_new_link_v1"):
                    st.session_state.auth_flow_page = 'forgot_password_request'; st.session_state.password_reset_token = None; st.rerun()
                st.markdown("</div></div>", unsafe_allow_html=True); return
            
            st.markdown("""<small>New password must meet complexity rules (8+ chars, upper, lower, digit, special).</small>""", unsafe_allow_html=True)
            with st.form("reset_password_form_v1"):
                new_password = st.text_input("New Password", type="password", key="reset_pw_new_v1")
                confirm_new_password = st.text_input("Confirm New Password", type="password", key="reset_pw_confirm_v1")
                submit_reset = st.form_submit_button("Reset Password", use_container_width=True, type="primary")
                if submit_reset:
                    if not new_password or not confirm_new_password: st.error("Please enter and confirm your new password.")
                    elif new_password != confirm_new_password: st.error("New passwords do not match.")
                    else:
                        with st.spinner("Resetting..."): result = auth_service.reset_password_with_token(token, new_password)
                        if result.get("success"):
                            st.success(result["success"]); st.info("You can now log in.");
                            st.session_state.auth_flow_page = 'login'; st.session_state.password_reset_token = None
                            if st.button("Go to Login", key="reset_pw_goto_login_success_v1"): st.rerun()
                        else: st.error(result.get("error", "Could not reset password."))
            if st.button("Cancel", use_container_width=True, type="secondary", key="reset_pw_cancel_v1"):
                st.session_state.auth_flow_page = 'login'; st.session_state.password_reset_token = None; st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)

if st.session_state.authenticated_user is None:
    st.sidebar.empty()
    if st.session_state.auth_flow_page == 'login': display_login_form()
    elif st.session_state.auth_flow_page == 'register': display_registration_form()
    elif st.session_state.auth_flow_page == 'forgot_password_request': display_forgot_password_request_form()
    elif st.session_state.auth_flow_page == 'reset_password_form': display_reset_password_form()
    else: st.session_state.auth_flow_page = 'login'; display_login_form()
    st.stop()

current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- Main App Logic (File Processing, Sidebar, KPIs, etc. - as before) ---
# This section assumes the code from "User Settings Persistence" update is present here.
# For brevity, only showing a small part of it.
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
default_session_state_main_app['selected_benchmark_display_name'] = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == default_session_state_main_app['selected_benchmark_ticker']), "None")
for key, value in default_session_state_main_app.items():
    if key not in st.session_state: st.session_state[key] = value
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"; logo_base64 = None
if os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        with open(LOGO_PATH_SIDEBAR, "rb") as image_file: logo_base64 = base64.b64encode(image_file.read()).decode()
    except Exception as e_logo: logger.error(f"Error encoding logo: {e_logo}", exc_info=True)
if logo_base64: st.logo(f"data:image/png;base64,{logo_base64}", icon_image=f"data:image/png;base64,{logo_base64}")
elif os.path.exists(LOGO_PATH_SIDEBAR): st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')
st.sidebar.header(APP_TITLE); st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_button_main_app_v5_settings", use_container_width=True):
    logger.info(f"User '{current_username}' logging out."); keys_to_clear = list(st.session_state.keys())
    for key_logout in keys_to_clear:
        if key_logout not in ['current_theme']: del st.session_state[key_logout]
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None; st.session_state.user_preferences = {}
    st.success("Logged out."); st.rerun()
st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_auth_v5_settings", use_container_width=True):
    new_theme = "dark" if st.session_state.current_theme == "light" else "light"; st.session_state.current_theme = new_theme
    auth_service.update_user_settings(current_user_id, {'default_theme': new_theme})
    st.session_state.user_preferences['default_theme'] = new_theme; logger.info(f"User {current_user_id} theme updated to {new_theme}."); st.rerun()
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Your Trading Journals")
user_files = data_service.list_user_files(current_user_id)
file_options = {f"{f.original_file_name} ({f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files}
file_options["✨ Upload New File..."] = "upload_new"; default_file_selection_label = "✨ Upload New File..."
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    matching_label = next((label for label, id_val in file_options.items() if id_val == st.session_state.selected_user_file_id), None)
    if matching_label: default_file_selection_label = matching_label
selected_file_label_in_sidebar = st.sidebar.selectbox("Select/Upload Journal:", options=list(file_options.keys()), index=list(file_options.keys()).index(default_file_selection_label), key="select_user_file_v7_settings")
selected_file_id_from_sidebar_dropdown = file_options.get(selected_file_label_in_sidebar)
if selected_file_id_from_sidebar_dropdown == "upload_new":
    newly_uploaded_file_object = st.sidebar.file_uploader("Upload New (CSV)", type=["csv"], key="app_wide_file_uploader_auth_v7_settings")
    if newly_uploaded_file_object is not None:
        current_raw_file_id = id(newly_uploaded_file_object)
        if st.session_state.get('last_uploaded_raw_file_id_tracker') != current_raw_file_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file_object.getvalue(); st.session_state.pending_file_to_save_name = newly_uploaded_file_object.name
            st.session_state.trigger_file_save_processing = True; st.session_state.last_uploaded_raw_file_id_tracker = current_raw_file_id; st.rerun()
elif selected_file_id_from_sidebar_dropdown is not None and st.session_state.selected_user_file_id != selected_file_id_from_sidebar_dropdown:
    st.session_state.trigger_file_load_id = selected_file_id_from_sidebar_dropdown; st.session_state.last_processed_mapping_for_file_id = None; st.rerun()
if selected_file_id_from_sidebar_dropdown != "upload_new" and selected_file_id_from_sidebar_dropdown is not None:
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label_in_sidebar.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_sidebar_dropdown}_v6_settings"):
        if data_service.delete_user_file(selected_file_id_from_sidebar_dropdown, current_user_id, True):
            st.sidebar.success(f"File '{selected_file_label_in_sidebar}' deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_from_sidebar_dropdown:
                keys_to_clear = ['selected_user_file_id', 'current_file_content_for_processing', 'processed_data', 'filtered_data', 'kpi_results', 'uploaded_file_name', 'column_mapping_confirmed', 'user_column_mapping', 'last_uploaded_file_for_mapping_id', 'csv_headers_for_mapping', 'last_processed_mapping_for_file_id', 'last_processed_file_id']
                for k in keys_to_clear: st.session_state[k] = None
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")
sidebar_manager = SidebarManager(st.session_state.get('processed_data')); current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters
if current_sidebar_filters:
    settings_changed = False; updated_settings_payload: Dict[str, Any] = {}
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar: st.session_state.risk_free_rate = rfr_from_sidebar; st.session_state.kpi_results = None; updated_settings_payload['default_risk_free_rate'] = rfr_from_sidebar; settings_changed = True
    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', DEFAULT_BENCHMARK_TICKER)
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar; st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == benchmark_ticker_from_sidebar), "None")
        st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None; updated_settings_payload['default_benchmark_ticker'] = benchmark_ticker_from_sidebar; settings_changed = True
    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar: st.session_state.initial_capital = initial_capital_from_sidebar; st.session_state.kpi_results = None
    if settings_changed and updated_settings_payload:
        auth_service.update_user_settings(current_user_id, updated_settings_payload)
        for key, value in updated_settings_payload.items(): st.session_state.user_preferences[key] = value
        logger.info(f"User {current_user_id} preferences updated: {updated_settings_payload}")
active_file_content_to_process = st.session_state.get('current_file_content_for_processing')
active_file_name_for_processing = st.session_state.get('uploaded_file_name')
@log_execution_time
def get_and_process_data_with_profiling(file_obj_bytesio: BytesIO, mapping: Dict[str, str], name_for_log: str):
    file_obj_bytesio.seek(0); return data_service.get_processed_trading_data(file_obj_bytesio, user_column_mapping=mapping, original_file_name=name_for_log)
if active_file_content_to_process and active_file_name_for_processing and current_user_id and st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    current_file_id_for_mapping = st.session_state.selected_user_file_id
    if st.session_state.get('last_processed_mapping_for_file_id') != current_file_id_for_mapping:
        st.session_state.last_processed_mapping_for_file_id = current_file_id_for_mapping; loaded_mapping = data_service.get_user_column_mapping(current_user_id, current_file_id_for_mapping)
        if loaded_mapping:
            st.session_state.user_column_mapping = loaded_mapping; st.session_state.column_mapping_confirmed = True; st.session_state.csv_headers_for_mapping = None; st.session_state.uploaded_file_bytes_for_mapper = None; st.session_state.last_processed_file_id = None
        else:
            st.session_state.user_column_mapping = None; st.session_state.column_mapping_confirmed = False
            try:
                active_file_content_to_process.seek(0); header_peek_io = BytesIO(active_file_content_to_process.getvalue()); df_peek = pd.read_csv(header_peek_io, nrows=5); st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
                active_file_content_to_process.seek(0); st.session_state.uploaded_file_bytes_for_mapper = BytesIO(active_file_content_to_process.getvalue())
            except Exception as e_header: display_custom_message(f"Error reading headers: {e_header}.", "error"); st.session_state.csv_headers_for_mapping = None; st.session_state.current_file_content_for_processing = None; st.stop()
    if not st.session_state.get('column_mapping_confirmed', False) and st.session_state.get('csv_headers_for_mapping'):
        st.session_state.processed_data = None; st.session_state.filtered_data = None; st.session_state.kpi_results = None
        column_mapper = ColumnMapperUI(uploaded_file_name=active_file_name_for_processing, uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper, csv_headers=st.session_state.csv_headers_for_mapping, conceptual_columns_map=CONCEPTUAL_COLUMNS, conceptual_column_types=CONCEPTUAL_COLUMN_TYPES, conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS, critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS, conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES)
        user_mapping_result = column_mapper.render()
        if user_mapping_result is not None:
            if not data_service.save_user_column_mapping(current_user_id, current_file_id_for_mapping, user_mapping_result): display_custom_message("Error: Could not save mapping.", "error")
            st.session_state.user_column_mapping = user_mapping_result; st.session_state.column_mapping_confirmed = True; st.session_state.last_processed_file_id = None; st.session_state.last_processed_mapping_for_file_id = current_file_id_for_mapping; st.rerun()
        else: display_custom_message("Please complete column mapping.", "info", icon="⚙️"); st.stop()
    if st.session_state.get('column_mapping_confirmed') and st.session_state.get('user_column_mapping'):
        if st.session_state.last_processed_file_id != current_file_id_for_mapping or st.session_state.processed_data is None:
            with st.spinner(f"Processing '{active_file_name_for_processing}'..."): active_file_content_to_process.seek(0); st.session_state.processed_data = get_and_process_data_with_profiling(active_file_content_to_process, st.session_state.user_column_mapping, active_file_name_for_processing)
            st.session_state.last_processed_file_id = current_file_id_for_mapping
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']: st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data
            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty: display_custom_message(f"Processed '{active_file_name_for_processing}'.", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty: display_custom_message(f"Processing '{active_file_name_for_processing}' resulted in empty data.", "warning")
            else: display_custom_message(f"Failed to process '{active_file_name_for_processing}'.", "error"); st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None; st.session_state.last_processed_mapping_for_file_id = None; st.session_state.current_file_content_for_processing = None; st.session_state.selected_user_file_id = None; st.rerun()
elif not active_file_content_to_process and st.session_state.authenticated_user:
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        keys_to_clear = ['processed_data', 'filtered_data', 'kpi_results', 'uploaded_file_name', 'last_processed_file_id', 'user_column_mapping', 'column_mapping_confirmed', 'csv_headers_for_mapping', 'last_uploaded_file_for_mapping_id', 'benchmark_daily_returns', 'max_drawdown_period_details', 'uploaded_file_bytes_for_mapper', 'current_file_content_for_processing', 'last_processed_mapping_for_file_id']
        for k in keys_to_clear: st.session_state[k] = None
        if st.session_state.get('selected_user_file_id') is not None: st.session_state.selected_user_file_id = None
if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty and st.session_state.get('sidebar_filters'):
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."): st.session_state.filtered_data = filter_data_with_profiling(st.session_state.processed_data, st.session_state.sidebar_filters, EXPECTED_COLUMNS)
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']: st.session_state[key_to_reset] = None
if st.session_state.get('filtered_data') is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        if st.session_state.benchmark_daily_returns is None or st.session_state.last_fetched_benchmark_ticker != selected_ticker or st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape:
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date'); min_d_str, max_d_str = None, None
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
            else: st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None
    elif st.session_state.benchmark_daily_returns is not None: st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None
    current_kpi_state_id_parts = [st.session_state.filtered_data.shape, st.session_state.risk_free_rate, st.session_state.initial_capital, st.session_state.selected_benchmark_ticker]
    if st.session_state.get('benchmark_daily_returns') is not None and not st.session_state.benchmark_daily_returns.empty:
        try: current_kpi_state_id_parts.append(pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum())
        except Exception: current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else: current_kpi_state_id_parts.append(None)
    current_kpi_state_id = tuple(current_kpi_state_id_parts)
    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        with st.spinner("Calculating KPIs..."):
            kpi_res = get_core_kpis_with_profiling(st.session_state.filtered_data, st.session_state.risk_free_rate, st.session_state.benchmark_daily_returns, st.session_state.initial_capital)
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res; st.session_state.last_kpi_calc_state_id = current_kpi_state_id
                date_col_dd, cum_pnl_col_dd = EXPECTED_COLUMNS.get('date'), 'cumulative_pnl'; equity_series_dd = pd.Series(dtype=float)
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
                    if len(pnl_s_ci) >= 10: st.session_state.kpi_confidence_intervals = analysis_service_instance.get_bootstrapped_kpi_cis(st.session_state.filtered_data) or {}
                    else: st.session_state.kpi_confidence_intervals = {}
                else: st.session_state.kpi_confidence_intervals = {}
            else: display_custom_message(f"KPI calculation error: {kpi_res.get('error', 'Unknown') if kpi_res else 'Service fail'}.", "error"); st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None
elif st.session_state.get('filtered_data') is not None and st.session_state.filtered_data.empty:
    if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty: display_custom_message("No data matches filters.", "info")
    st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None
def main_page_layout():
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-section'><h1 class='welcome-title'>Trading Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='welcome-subtitle'>Powered by {PAGE_CONFIG_APP_TITLE}</p></div>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Unlock insights from your trading data.</p>", unsafe_allow_html=True)
    if not st.session_state.get('current_file_content_for_processing') and not st.session_state.get('processed_data'):
        st.info("No trading journal loaded. Select or upload one via the sidebar.", icon="📄")
    st.markdown("<h2 class='features-title' style='text-align: center; color: var(--secondary-color); margin-top: 2rem;'>Get Started</h2>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3, gap="large")
    with col1: st.markdown("<div class='feature-item'><h4>📄 Manage Files</h4><p>Upload or select journals via sidebar.</p></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='feature-item'><h4>📊 Analyze</h4><p>Explore metrics once data is processed.</p></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='feature-item'><h4>💡 Discover</h4><p>Use filters for deeper insights.</p></div>", unsafe_allow_html=True)
    st.markdown("<br><div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py"
    if os.path.exists(user_guide_page_path):
        if st.button("📘 Read User Guide", key="welcome_guide_btn_auth_v3"): st.switch_page(user_guide_page_path)
    else: st.markdown("<p>User guide not found.</p>", unsafe_allow_html=True); logger.warning(f"User Guide not found: {user_guide_page_path}")
    st.markdown("</div></div>", unsafe_allow_html=True)
processed_data_main_final = st.session_state.get('processed_data')
condition_for_main_layout_final = not st.session_state.get('current_file_content_for_processing') and \
    not (st.session_state.get('column_mapping_confirmed') and processed_data_main_final is not None and not processed_data_main_final.empty)
if condition_for_main_layout_final: main_page_layout()
elif (st.session_state.get('processed_data') is None or st.session_state.get('processed_data').empty) and st.session_state.get('current_file_content_for_processing') and not st.session_state.get('column_mapping_confirmed'):
    if not st.session_state.get('csv_headers_for_mapping'): display_custom_message("Preparing for column mapping...", "info")
scroll_buttons_component = ScrollButtons(); scroll_buttons_component.render()
logger.info(f"App run cycle finished for user '{current_username}'. File ID: {st.session_state.get('selected_user_file_id')}")

