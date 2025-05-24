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
if 'selected_user_file_id' not in st.session_state: st.session_state.selected_user_file_id = None
if 'current_file_content_for_processing' not in st.session_state: st.session_state.current_file_content_for_processing = None
if 'file_to_save' not in st.session_state: st.session_state.file_to_save = None # New state for pending save

def display_login_form():
    # ... (implementation as before) ...
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True) 
        with auth_area_container:
            with st.form("login_form_main"): 
                st.markdown(f"<h2 style='text-align: center;'>Login to {APP_TITLE}</h2>", unsafe_allow_html=True)
                username = st.text_input("Username", key="login_username_main")
                password = st.text_input("Password", type="password", key="login_password_main")
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
            if st.button("Don't have an account? Register", use_container_width=True, key="goto_register_btn_main_v2"):
                st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def display_registration_form():
    # ... (implementation as before) ...
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("registration_form_main"): 
                st.markdown(f"<h2 style='text-align: center;'>Register for {APP_TITLE}</h2>", unsafe_allow_html=True)
                reg_username = st.text_input("Username", key="reg_username_main")
                reg_email = st.text_input("Email (Optional)", key="reg_email_main")
                reg_password = st.text_input("Password", type="password", key="reg_password_main")
                reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm_main")
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
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn_main_v2"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.authenticated_user is None:
    st.sidebar.empty()
    if st.session_state.auth_flow_page == 'login': display_login_form()
    elif st.session_state.auth_flow_page == 'register': display_registration_form()
    else: st.session_state.auth_flow_page = 'login'; display_login_form()
    st.stop()

current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

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


# Sidebar for authenticated user
# ... (Logo, User Info, Logout, Theme Toggle as before) ...
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"
logo_base64 = None
if os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        with open(LOGO_PATH_SIDEBAR, "rb") as image_file: logo_base64 = base64.b64encode(image_file.read()).decode()
    except Exception as e_logo: logger.error(f"Error encoding logo: {e_logo}", exc_info=True)

if logo_base64: st.logo(f"data:image/png;base64,{logo_base64}", icon_image=f"data:image/png;base64,{logo_base64}")
elif os.path.exists(LOGO_PATH_SIDEBAR): st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')

st.sidebar.header(APP_TITLE)
st.sidebar.markdown(f"Logged in as: **{current_username}**")
if st.sidebar.button("🔒 Logout", key="logout_button_main_app_v3", use_container_width=True):
    logger.info(f"User '{current_username}' logging out.")
    keys_to_clear_on_logout = list(st.session_state.keys()) 
    for key_logout in keys_to_clear_on_logout: 
        if key_logout not in ['current_theme']: del st.session_state[key_logout]
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None
    st.success("You have been logged out."); st.rerun()

st.sidebar.markdown("---")
toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_auth_v3", use_container_width=True):
    st.session_state.current_theme = "dark" if st.session_state.current_theme == "light" else "light"; st.rerun()
st.sidebar.markdown("---")


st.sidebar.subheader("📁 Your Trading Journals")
user_files = data_service.list_user_files(current_user_id)
file_options = {f"{f.original_file_name} (Uploaded: {f.upload_timestamp.strftime('%Y-%m-%d %H:%M')})": f.id for f in user_files}
file_options["✨ Upload New File..."] = "upload_new"

default_file_selection_label = "✨ Upload New File..."
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id in file_options.values():
    default_file_selection_label = next((label for label, id_val in file_options.items() if id_val == st.session_state.selected_user_file_id), "✨ Upload New File...")

selected_file_label = st.sidebar.selectbox(
    "Select a journal or upload new:", options=list(file_options.keys()),
    index=list(file_options.keys()).index(default_file_selection_label),
    key="select_user_file_v3"
)
selected_file_id_from_dropdown = file_options.get(selected_file_label)

# --- MODIFIED FILE UPLOAD AND SAVE LOGIC ---
if selected_file_id_from_dropdown == "upload_new":
    newly_uploaded_file_object = st.sidebar.file_uploader(
        "Upload New Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader_auth_v3",
        help="Your uploaded CSV will be saved to your account."
    )
    if newly_uploaded_file_object:
        # Store the uploaded file object in session state to be processed after the rerun
        st.session_state.file_to_save = newly_uploaded_file_object
        # Clear the uploader by rerunning (it will be None in the next run unless a new file is chosen)
        st.rerun() 

# Process pending file save if one exists in session state
if st.session_state.get('file_to_save') is not None:
    file_to_process_save = st.session_state.file_to_save
    st.session_state.file_to_save = None # Clear it immediately to prevent reprocessing

    with st.sidebar.spinner("Saving your file..."): # Spinner is now outside the uploader's direct if block
        saved_user_file_record = data_service.save_user_file(current_user_id, file_to_process_save)
        if saved_user_file_record:
            st.sidebar.success(f"File '{saved_user_file_record.original_file_name}' saved!")
            st.session_state.selected_user_file_id = saved_user_file_record.id
            st.session_state.current_file_content_for_processing = None
            st.session_state.processed_data = None
            st.session_state.column_mapping_confirmed = False
            st.session_state.uploaded_file_name = saved_user_file_record.original_file_name
            st.sidebar.info("File saved. It will be loaded for analysis.")
            st.rerun() # Rerun to load the newly selected file
        else:
            st.sidebar.error("Could not save the file. Please try again.")
# --- END OF MODIFIED FILE UPLOAD AND SAVE LOGIC ---


if selected_file_id_from_dropdown != "upload_new" and selected_file_id_from_dropdown is not None:
    if st.session_state.selected_user_file_id != selected_file_id_from_dropdown or st.session_state.current_file_content_for_processing is None:
        with st.sidebar.spinner("Loading selected file..."):
            file_content_bytesio = data_service.get_user_file_content(selected_file_id_from_dropdown, current_user_id)
            if file_content_bytesio:
                st.session_state.current_file_content_for_processing = file_content_bytesio
                st.session_state.selected_user_file_id = selected_file_id_from_dropdown
                selected_file_record = next((f for f in user_files if f.id == selected_file_id_from_dropdown), None)
                st.session_state.uploaded_file_name = selected_file_record.original_file_name if selected_file_record else "Selected File"
                st.session_state.processed_data = None 
                st.session_state.column_mapping_confirmed = False 
                st.sidebar.info(f"Loaded '{st.session_state.uploaded_file_name}' for analysis.")
                st.rerun()
            else:
                st.sidebar.error("Could not load the selected file.")
                st.session_state.current_file_content_for_processing = None
                st.session_state.selected_user_file_id = None

if selected_file_id_from_dropdown != "upload_new" and selected_file_id_from_dropdown is not None:
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_dropdown}_v2"):
        if data_service.delete_user_file(selected_file_id_from_dropdown, current_user_id, permanent_delete_local_file=True):
            st.sidebar.success(f"File '{selected_file_label}' marked as deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_from_dropdown:
                st.session_state.selected_user_file_id = None; st.session_state.current_file_content_for_processing = None
                st.session_state.processed_data = None; st.session_state.uploaded_file_name = None
                st.session_state.column_mapping_confirmed = False
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")

sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters

if current_sidebar_filters: # Update session state based on sidebar controls
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
active_processing_file_identifier = st.session_state.selected_user_file_id if st.session_state.selected_user_file_id else active_file_name_for_processing

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
        with st.spinner("Calculating metrics..."):
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

def main_page_layout():
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
        if st.button("📘 Read User Guide", key="welcome_guide_btn_auth_v2"): st.switch_page(user_guide_page_path)
    else: st.markdown("<p>User guide not found.</p>", unsafe_allow_html=True); logger.warning(f"User Guide not found: {user_guide_page_path}")
    st.markdown("</div></div>", unsafe_allow_html=True)

if not active_file_content_to_process and not (st.session_state.get('column_mapping_confirmed') and st.session_state.get('processed_data') is not None):
    main_page_layout() 

scroll_buttons_component = ScrollButtons()
scroll_buttons_component.render()
logger.info(f"App run cycle finished for user '{current_username}'.")

