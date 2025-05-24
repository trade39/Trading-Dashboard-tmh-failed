# app.py - Main Entry Point
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
    
    from services import ( 
        DataService, 
        AnalysisService, 
        AuthService,
        create_db_tables, 
        get_benchmark_data_static 
    )
except ImportError as e:
    st.error(f"Fatal Error: A critical module could not be imported. App cannot start. Details: {e}")
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Fatal Error during initial imports: {e}", exc_info=True)
    APP_TITLE = "TradingAppError" 
    try:
        logger = logging.getLogger(APP_TITLE)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr); formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"); handler.setFormatter(formatter); logger.addHandler(handler); logger.setLevel("ERROR")
    except Exception: pass
    st.stop()

# --- Page Config ---
PAGE_CONFIG_APP_TITLE = APP_TITLE
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png"
st.set_page_config(
    page_title=PAGE_CONFIG_APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4',
        'Report a bug': "https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4/issues",
        'About': f"## {PAGE_CONFIG_APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)

# --- Initialize Logger, Services, and Database Tables ---
logger = setup_logger(logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")

data_service = DataService()
analysis_service_instance = AnalysisService()
auth_service = AuthService()

try:
    create_db_tables() 
    logger.info("Database tables checked/created successfully via centralized function.")
except Exception as db_init_e:
    logger.critical(f"Failed to initialize database tables via centralized function: {db_init_e}", exc_info=True)
    st.error(f"Database Initialization Error: {db_init_e}. The application might not function correctly.")

# --- Theme Management & CSS ---
if 'current_theme' not in st.session_state: st.session_state.current_theme = "dark"
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

# --- Session state for file management (Phase 2) ---
if 'selected_user_file_id' not in st.session_state: st.session_state.selected_user_file_id = None
if 'current_file_content_for_processing' not in st.session_state: st.session_state.current_file_content_for_processing = None
if 'pending_file_to_save_content' not in st.session_state: st.session_state.pending_file_to_save_content = None
if 'pending_file_to_save_name' not in st.session_state: st.session_state.pending_file_to_save_name = None
if 'trigger_file_save_processing' not in st.session_state: st.session_state.trigger_file_save_processing = False # Renamed for clarity
if 'last_uploaded_raw_file_id_tracker' not in st.session_state: st.session_state.last_uploaded_raw_file_id_tracker = None
if 'trigger_file_load_id' not in st.session_state: st.session_state.trigger_file_load_id = None


def display_login_form():
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
                            st.success(f"Welcome back, {username}!"); st.rerun()
                        else: st.error("Invalid username or password.")
            if st.button("Don't have an account? Register", use_container_width=True, key="goto_register_btn_main_v4"):
                st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
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
                            st.success(f"User '{reg_username}' registered successfully! Please login."); st.session_state.auth_flow_page = 'login'; st.rerun()
                        else:
                            if auth_service.get_user_by_username(reg_username): st.error(f"Username '{reg_username}' already exists.")
                            else: st.error("Registration failed. Username/email might be taken or internal error.")
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn_main_v4"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.authenticated_user is None:
    st.sidebar.empty()
    if st.session_state.auth_flow_page == 'login': display_login_form()
    elif st.session_state.auth_flow_page == 'register': display_registration_form()
    else: st.session_state.auth_flow_page = 'login'; display_login_form()
    st.stop()

# --- USER IS AUTHENTICATED ---
current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- Process Pending File Save (Moved to main area, before other UI) ---
if st.session_state.get('trigger_file_save_processing') and st.session_state.get('pending_file_to_save_content') is not None:
    file_content_to_save_bytes = st.session_state.pending_file_to_save_content
    original_name_to_save = st.session_state.pending_file_to_save_name
    
    st.session_state.trigger_file_save_processing = False # Clear the trigger
    st.session_state.pending_file_to_save_content = None
    st.session_state.pending_file_to_save_name = None

    temp_uploaded_file_for_service = BytesIO(file_content_to_save_bytes)
    temp_uploaded_file_for_service.name = original_name_to_save 

    # Use spinner in the main application area
    main_area_placeholder = st.empty() # Create a placeholder in the main area
    with main_area_placeholder.status(f"Saving '{original_name_to_save}' to your account...", expanded=True):
        # st.spinner is not needed if using st.status
        # with st.spinner(f"Saving '{original_name_to_save}' to your account... Please wait."):
        st.write(f"Uploading and saving '{original_name_to_save}'...")
        saved_user_file_record = data_service.save_user_file(current_user_id, temp_uploaded_file_for_service)
        if saved_user_file_record:
            st.write("File saved to database record.")
            st.session_state.selected_user_file_id = saved_user_file_record.id
            st.session_state.current_file_content_for_processing = None 
            st.session_state.processed_data = None 
            st.session_state.column_mapping_confirmed = False 
            st.session_state.uploaded_file_name = saved_user_file_record.original_file_name
            logger.info(f"File '{saved_user_file_record.original_file_name}' saved for user {current_user_id}. Triggering rerun to load.")
            st.rerun() # Rerun to load the newly selected file and clear spinner
        else:
            st.error(f"Could not save the file '{original_name_to_save}'. Please try uploading again.")
            # No rerun here, error message stays.

# --- Process Pending File Load (Moved to main area) ---
if st.session_state.get('trigger_file_load_id') is not None:
    file_id_to_load = st.session_state.trigger_file_load_id
    st.session_state.trigger_file_load_id = None 

    user_files_for_load_check = data_service.list_user_files(current_user_id)
    selected_file_record_for_load = next((f for f in user_files_for_load_check if f.id == file_id_to_load), None)
    
    if selected_file_record_for_load:
        main_area_placeholder_load = st.empty()
        with main_area_placeholder_load.status(f"Loading '{selected_file_record_for_load.original_file_name}'...", expanded=True):
            st.write(f"Fetching content for '{selected_file_record_for_load.original_file_name}'...")
            file_content_bytesio = data_service.get_user_file_content(file_id_to_load, current_user_id)
            if file_content_bytesio:
                st.session_state.current_file_content_for_processing = file_content_bytesio
                st.session_state.selected_user_file_id = file_id_to_load
                st.session_state.uploaded_file_name = selected_file_record_for_load.original_file_name
                st.session_state.processed_data = None 
                st.session_state.column_mapping_confirmed = False 
                logger.info(f"File ID {file_id_to_load} loaded for user {current_user_id}. Triggering rerun for processing.")
                st.rerun()
            else:
                st.error(f"Could not load the content for file: {selected_file_record_for_load.original_file_name}.")
                st.session_state.current_file_content_for_processing = None
                st.session_state.selected_user_file_id = None
    elif file_id_to_load is not None: # Only show error if a load was actually triggered
        st.error(f"Selected file (ID: {file_id_to_load}) not found or access denied.")
        st.session_state.selected_user_file_id = None


# --- Initialize Main App Session State (if not already done) ---
default_session_state_main_app = { /* ... as before ... */ }
# ... (session state initialization as before) ...
default_session_state_main_app = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None,
    'kpi_results': None, 'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE, 'uploaded_file_name': None,
    'uploaded_file_bytes_for_mapper': None, 'last_processed_file_id': None,
    'user_column_mapping': None, 'column_mapping_confirmed': False,
    'csv_headers_for_mapping': None, 'last_uploaded_file_for_mapping_id': None,
    'last_applied_filters': None, 'sidebar_filters': None, 
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER,
    'benchmark_daily_returns': None, 'initial_capital': 100000.0,
    'last_fetched_benchmark_ticker': None, 'last_benchmark_data_filter_shape': None,
    'last_kpi_calc_state_id': None, 'max_drawdown_period_details': None
}
default_session_state_main_app['selected_benchmark_display_name'] = next(
    (name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == default_session_state_main_app['selected_benchmark_ticker']), "None"
)
for key, value in default_session_state_main_app.items():
    if key not in st.session_state: st.session_state[key] = value


# --- Sidebar for Authenticated User ---
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"
# ... (logo rendering as before) ...
logo_base64 = None
if os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        with open(LOGO_PATH_SIDEBAR, "rb") as image_file: logo_base64 = base64.b64encode(image_file.read()).decode()
    except Exception as e_logo: logger.error(f"Error encoding logo: {e_logo}", exc_info=True)

if logo_base64: st.logo(f"data:image/png;base64,{logo_base64}", icon_image=f"data:image/png;base64,{logo_base64}")
elif os.path.exists(LOGO_PATH_SIDEBAR): st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')

st.sidebar.header(APP_TITLE)
st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_button_main_app_v4", use_container_width=True):
    logger.info(f"User '{current_username}' logging out.")
    keys_to_clear_on_logout = list(st.session_state.keys()) 
    for key_logout in keys_to_clear_on_logout: 
        if key_logout not in ['current_theme']: del st.session_state[key_logout]
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None
    st.success("You have been logged out."); st.rerun()

st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_auth_v4", use_container_width=True):
    st.session_state.current_theme = "dark" if st.session_state.current_theme == "light" else "light"; st.rerun()
st.sidebar.markdown("---")

st.sidebar.subheader("📁 Your Trading Journals")
user_files = data_service.list_user_files(current_user_id) 
file_options = {f"{f.original_file_name} (Uploaded: {f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files}
file_options["✨ Upload New File..."] = "upload_new"

default_file_selection_label = "✨ Upload New File..."
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new" and st.session_state.selected_user_file_id in file_options.values():
    default_file_selection_label = next((label for label, id_val in file_options.items() if id_val == st.session_state.selected_user_file_id), "✨ Upload New File...")

selected_file_label_in_sidebar = st.sidebar.selectbox(
    "Select a journal or upload new:", options=list(file_options.keys()),
    index=list(file_options.keys()).index(default_file_selection_label),
    key="select_user_file_v4" 
)
selected_file_id_from_sidebar_dropdown = file_options.get(selected_file_label_in_sidebar)

if selected_file_id_from_sidebar_dropdown == "upload_new":
    newly_uploaded_file_object = st.sidebar.file_uploader(
        "Upload New Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader_auth_v4_trigger",
        help="Your uploaded CSV will be saved to your account."
    )
    if newly_uploaded_file_object is not None:
        current_raw_file_id = id(newly_uploaded_file_object)
        if st.session_state.get('last_uploaded_raw_file_id_tracker') != current_raw_file_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file_object.getvalue()
            st.session_state.pending_file_to_save_name = newly_uploaded_file_object.name
            st.session_state.trigger_file_save_processing = True # Use new trigger name
            st.session_state.last_uploaded_raw_file_id_tracker = current_raw_file_id
            logger.debug(f"New file '{newly_uploaded_file_object.name}' detected. Setting trigger_file_save_processing.")
            st.rerun() 
elif selected_file_id_from_sidebar_dropdown is not None and \
     st.session_state.selected_user_file_id != selected_file_id_from_sidebar_dropdown:
    # User selected a different existing file from dropdown
    st.session_state.trigger_file_load_id = selected_file_id_from_sidebar_dropdown
    logger.debug(f"User selected existing file ID {selected_file_id_from_sidebar_dropdown}. Setting trigger_file_load_id.")
    st.rerun()

if selected_file_id_from_sidebar_dropdown != "upload_new" and selected_file_id_from_sidebar_dropdown is not None:
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label_in_sidebar.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_sidebar_dropdown}_v3"):
        if data_service.delete_user_file(selected_file_id_from_sidebar_dropdown, current_user_id, permanent_delete_local_file=True):
            st.sidebar.success(f"File '{selected_file_label_in_sidebar}' marked as deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_from_sidebar_dropdown:
                st.session_state.selected_user_file_id = None; st.session_state.current_file_content_for_processing = None
                st.session_state.processed_data = None; st.session_state.uploaded_file_name = None
                st.session_state.column_mapping_confirmed = False
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")

sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters

if current_sidebar_filters:
    # ... (filter update logic as before) ...
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar: st.session_state.risk_free_rate = rfr_from_sidebar; st.session_state.kpi_results = None
    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == benchmark_ticker_from_sidebar), "None")
        st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None
    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar: st.session_state.initial_capital = initial_capital_from_sidebar; st.session_state.kpi_results = None

# --- Data Processing Pipeline ---
active_file_content_to_process = st.session_state.get('current_file_content_for_processing')
active_file_name_for_processing = st.session_state.get('uploaded_file_name')
active_processing_file_identifier = st.session_state.selected_user_file_id if (st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new") else active_file_name_for_processing

@log_execution_time
def get_and_process_data_with_profiling(file_obj, mapping, name):
    # ... (implementation as before) ...
    if hasattr(file_obj, 'getvalue') and not isinstance(file_obj, BytesIO):
        file_bytes_io = BytesIO(file_obj.getvalue()); file_bytes_io.seek(0)
        return data_service.get_processed_trading_data(file_bytes_io, user_column_mapping=mapping, original_file_name=name)
    elif isinstance(file_obj, BytesIO):
        file_obj.seek(0)
        return data_service.get_processed_trading_data(file_obj, user_column_mapping=mapping, original_file_name=name)
    logger.error("get_and_process_data_with_profiling: file_obj is not compatible."); return None

if active_file_content_to_process and active_file_name_for_processing:
    # ... (column mapping and data processing logic as before) ...
    if st.session_state.last_uploaded_file_for_mapping_id != active_processing_file_identifier or not st.session_state.column_mapping_confirmed:
        logger.info(f"File '{active_file_name_for_processing}' (ID: {active_processing_file_identifier}) needs mapping.")
        st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
        st.session_state.processed_data = None; st.session_state.filtered_data = None
        st.session_state.last_uploaded_file_for_mapping_id = active_processing_file_identifier
        try:
            active_file_content_to_process.seek(0)
            df_peek = pd.read_csv(BytesIO(active_file_content_to_process.getvalue()), nrows=5)
            st.session_state.csv_headers_for_mapping = df_peek.columns.tolist(); active_file_content_to_process.seek(0)
            st.session_state.uploaded_file_bytes_for_mapper = active_file_content_to_process
        except Exception as e_header:
            logger.error(f"Could not read CSV headers from active file '{active_file_name_for_processing}': {e_header}", exc_info=True)
            display_custom_message(f"Error reading '{active_file_name_for_processing}': {e_header}.", "error")
            st.session_state.csv_headers_for_mapping = None; st.stop()
    if st.session_state.csv_headers_for_mapping and not st.session_state.column_mapping_confirmed:
        column_mapper = ColumnMapperUI(uploaded_file_name=active_file_name_for_processing, uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper, csv_headers=st.session_state.csv_headers_for_mapping, conceptual_columns_map=CONCEPTUAL_COLUMNS, conceptual_column_types=CONCEPTUAL_COLUMN_TYPES, conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS, critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS, conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES)
        user_mapping_result = column_mapper.render()
        if user_mapping_result is not None: st.session_state.user_column_mapping = user_mapping_result; st.session_state.column_mapping_confirmed = True; st.session_state.last_processed_file_id = None; st.rerun()
        else: st.stop()
    if st.session_state.column_mapping_confirmed and st.session_state.user_column_mapping:
        if st.session_state.last_processed_file_id != active_processing_file_identifier or st.session_state.processed_data is None:
            with st.spinner(f"Processing '{active_file_name_for_processing}'..."): 
                active_file_content_to_process.seek(0)
                st.session_state.processed_data = get_and_process_data_with_profiling(active_file_content_to_process, st.session_state.user_column_mapping, active_file_name_for_processing)
            st.session_state.last_processed_file_id = active_processing_file_identifier
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']: st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data
            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty: display_custom_message(f"Successfully processed '{active_file_name_for_processing}'.", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty: display_custom_message(f"Processing of '{active_file_name_for_processing}' resulted in empty data.", "warning"); st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
            else: display_custom_message(f"Failed to process '{active_file_name_for_processing}'.", "error"); st.session_state.column_mapping_confirmed = False; st.session_state.user_column_mapping = None
elif not active_file_content_to_process and st.session_state.authenticated_user:
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        logger.info("No active file. Clearing previous processed data.")
        keys_to_clear_no_active = ['processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals', 'uploaded_file_name', 'last_processed_file_id', 'user_column_mapping', 'column_mapping_confirmed', 'csv_headers_for_mapping', 'last_uploaded_file_for_mapping_id', 'benchmark_daily_returns', 'max_drawdown_period_details']
        for key_val in keys_to_clear_no_active:
            if key_val in st.session_state: st.session_state[key_val] = None

# --- Data Filtering, Benchmark Fetching, KPI Calculation (as before) ---
# ... (This entire section remains the same) ...
@log_execution_time
def filter_data_with_profiling(df, filters, col_map): return data_service.filter_data(df, filters, col_map)

if st.session_state.processed_data is not None and not st.session_state.processed_data.empty and st.session_state.sidebar_filters:
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."): st.session_state.filtered_data = filter_data_with_profiling(st.session_state.processed_data, st.session_state.sidebar_filters, EXPECTED_COLUMNS)
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']: st.session_state[key_to_reset] = None

if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None: refetch_benchmark = True
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker: refetch_benchmark = True
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape: refetch_benchmark = True
        if refetch_benchmark:
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date')
            min_d_str_to_fetch, max_d_str_to_fetch = None, None
            if date_col_conceptual in st.session_state.filtered_data.columns:
                dates_for_bm_filtered = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                if not dates_for_bm_filtered.empty:
                    min_d_filtered, max_d_filtered = dates_for_bm_filtered.min(), dates_for_bm_filtered.max()
                    if pd.notna(min_d_filtered) and pd.notna(max_d_filtered) and (max_d_filtered.date() - min_d_filtered.date()).days >= 0: min_d_str_to_fetch, max_d_str_to_fetch = min_d_filtered.strftime('%Y-%m-%d'), max_d_filtered.strftime('%Y-%m-%d')
            if min_d_str_to_fetch and max_d_str_to_fetch:
                with st.spinner(f"Fetching benchmark: {selected_ticker}..."): st.session_state.benchmark_daily_returns = get_benchmark_data_static(selected_ticker, min_d_str_to_fetch, max_d_str_to_fetch)
                st.session_state.last_fetched_benchmark_ticker = selected_ticker; st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty: display_custom_message(f"Could not fetch benchmark data for {selected_ticker}.", "warning")
            else: logger.warning(f"Cannot fetch benchmark {selected_ticker}: invalid date range."); st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None
    elif st.session_state.benchmark_daily_returns is not None: st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None

@log_execution_time
def get_core_kpis_with_profiling(df, rfr, benchmark_returns, capital): return analysis_service_instance.get_core_kpis(df, rfr, benchmark_returns, capital)
@log_execution_time
def get_advanced_drawdown_analysis_with_profiling(equity_series): return analysis_service_instance.get_advanced_drawdown_analysis(equity_series)

if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    current_kpi_state_id_parts = [st.session_state.filtered_data.shape, st.session_state.risk_free_rate, st.session_state.initial_capital, st.session_state.selected_benchmark_ticker]
    if st.session_state.benchmark_daily_returns is not None and not st.session_state.benchmark_daily_returns.empty:
        try: current_kpi_state_id_parts.append(pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum())
        except Exception as e_hash_bm: logger.warning(f"Hashing benchmark failed: {e_hash_bm}. Using shape."); current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else: current_kpi_state_id_parts.append(None)
    current_kpi_state_id = tuple(current_kpi_state_id_parts)
    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info("Recalculating KPIs...")
        with st.spinner("Calculating metrics..."): # This spinner is for KPI calculation
            kpi_res = get_core_kpis_with_profiling(st.session_state.filtered_data, st.session_state.risk_free_rate, st.session_state.benchmark_daily_returns, st.session_state.initial_capital)
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res; st.session_state.last_kpi_calc_state_id = current_kpi_state_id
                date_col_dd, cum_pnl_col_dd = EXPECTED_COLUMNS.get('date'), 'cumulative_pnl'
                equity_series_dd = pd.Series(dtype=float)
                if date_col_dd and cum_pnl_col_dd and date_col_dd in st.session_state.filtered_data and cum_pnl_col_dd in st.session_state.filtered_data:
                    df_eq_dd = st.session_state.filtered_data[[date_col_dd, cum_pnl_col_dd]].copy(); df_eq_dd[date_col_dd] = pd.to_datetime(df_eq_dd[date_col_dd], errors='coerce'); df_eq_dd.dropna(subset=[date_col_dd], inplace=True)
                    if not df_eq_dd.empty: equity_series_dd = df_eq_dd.set_index(date_col_dd)[cum_pnl_col_dd].sort_index().dropna()
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
            else:
                error_msg = kpi_res.get('error', 'Unknown error') if kpi_res else 'KPI service failed'
                display_custom_message(f"KPI calculation error: {error_msg}.", "error")
                st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty: display_custom_message("No data matches filters.", "info")
    st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None

# --- Welcome Page / Main Content Display Logic ---
def main_page_layout():
    # ... (implementation as before) ...
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-section'><h1 class='welcome-title'>Trading Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='welcome-subtitle'>Powered by {PAGE_CONFIG_APP_TITLE}</p></div>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Unlock insights from your trading data.</p>", unsafe_allow_html=True)
    st.markdown("<h2 class='features-title' style='text-align: center; color: var(--secondary-color);'>Get Started</h2>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3, gap="large")
    with col1: st.markdown("<div class='feature-item'><h4>📄 Manage Files</h4><p>Upload new journals or select existing ones from your saved files via the sidebar.</p></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='feature-item'><h4>📊 Analyze</h4><p>Explore performance metrics once data is loaded.</p></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='feature-item'><h4>💡 Discover</h4><p>Use advanced tools for insights.</p></div>", unsafe_allow_html=True)
    st.markdown("<br><div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py"
    if os.path.exists(user_guide_page_path):
        if st.button("📘 Read User Guide", key="welcome_guide_btn_auth_v3"): # Incremented key
             st.switch_page(user_guide_page_path)
    else: st.markdown("<p>User guide not found.</p>", unsafe_allow_html=True); logger.warning(f"User Guide not found: {user_guide_page_path}")
    st.markdown("</div></div>", unsafe_allow_html=True)

if not active_file_content_to_process and not (st.session_state.get('column_mapping_confirmed') and st.session_state.get('processed_data') is not None):
    main_page_layout() 

scroll_buttons_component = ScrollButtons()
scroll_buttons_component.render()
logger.info(f"App run cycle finished for user '{current_username}'.")
