# app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime # Ensure datetime is imported directly if used for type hints or specific operations
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

    from services import ( # Assuming services are correctly exported from services/__init__.py
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
    APP_TITLE = "TradingAppError" # Fallback
    # Minimal fallbacks to allow app to show the error
    class DataService: pass
    class AnalysisService: pass
    class AuthService: pass
    def create_db_tables(): pass
    def get_benchmark_data_static(*args, **kwargs): return None
    class SidebarManager: def __init__(self, *args): pass; def render_sidebar_controls(self): return {}
    class ColumnMapperUI: def __init__(self, *args, **kwargs): pass; def render(self): return None
    class ScrollButtons: def __init__(self, *args, **kwargs): pass; def render(self): pass
    def load_css(f): pass
    def display_custom_message(m, t): st.error(m)
    def setup_logger(**kwargs): return logging.getLogger(APP_TITLE)

    st.stop()

# --- Page Config ---
PAGE_CONFIG_APP_TITLE = APP_TITLE
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png" # Ensure this asset exists
st.set_page_config(
    page_title=PAGE_CONFIG_APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4', # Replace with your repo
        'Report a bug': "https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4/issues", # Replace
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
    create_db_tables() # This should be called once, ideally at app startup
    logger.info("Database tables checked/created successfully via centralized function.")
except Exception as db_init_e:
    logger.critical(f"Failed to initialize database tables: {db_init_e}", exc_info=True)
    st.error(f"Database Initialization Error: {db_init_e}. The application might not function correctly.")
    # Depending on severity, you might st.stop() here

# --- Theme Management & CSS ---
if 'current_theme' not in st.session_state: st.session_state.current_theme = "dark" # Default to dark
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
if 'auth_flow_page' not in st.session_state: st.session_state.auth_flow_page = 'login' # Default to login

# --- Session state for file management & mapping ---
if 'selected_user_file_id' not in st.session_state: st.session_state.selected_user_file_id = None
if 'current_file_content_for_processing' not in st.session_state: st.session_state.current_file_content_for_processing = None
if 'pending_file_to_save_content' not in st.session_state: st.session_state.pending_file_to_save_content = None
if 'pending_file_to_save_name' not in st.session_state: st.session_state.pending_file_to_save_name = None
if 'trigger_file_save_processing' not in st.session_state: st.session_state.trigger_file_save_processing = False
if 'last_uploaded_raw_file_id_tracker' not in st.session_state: st.session_state.last_uploaded_raw_file_id_tracker = None
if 'trigger_file_load_id' not in st.session_state: st.session_state.trigger_file_load_id = None
if 'last_processed_mapping_for_file_id' not in st.session_state: st.session_state.last_processed_mapping_for_file_id = None # For mapping persistence


def display_login_form():
    # ... (implementation as before)
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("login_form_main_v3"): # Key can remain, form is self-contained
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
                            st.session_state.auth_flow_page = None # Clear auth flow page
                            logger.info(f"User '{username}' logged in successfully.")
                            st.success(f"Welcome back, {username}!"); st.rerun()
                        else: st.error("Invalid username or password.")
            if st.button("Don't have an account? Register", use_container_width=True, key="goto_register_btn_main_v4"):
                st.session_state.auth_flow_page = 'register'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def display_registration_form():
    # ... (implementation as before)
    with st.container():
        st.markdown("<div style='display: flex; justify-content: center; margin-top: 5vh;'>", unsafe_allow_html=True)
        auth_area_container = st.container(border=True)
        with auth_area_container:
            with st.form("registration_form_main_v3"): # Key can remain
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
                            if auth_service.get_user_by_username(reg_username): st.error(f"Username '{reg_username}' already exists.") # Check if username taken
                            else: st.error("Registration failed. Username/email might be taken or internal error.")
            if st.button("Already have an account? Login", use_container_width=True, key="goto_login_btn_main_v4"):
                st.session_state.auth_flow_page = 'login'; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.authenticated_user is None:
    st.sidebar.empty() # Clear sidebar for auth pages
    if st.session_state.auth_flow_page == 'login': display_login_form()
    elif st.session_state.auth_flow_page == 'register': display_registration_form()
    else: st.session_state.auth_flow_page = 'login'; display_login_form() # Default to login
    st.stop()

# --- USER IS AUTHENTICATED ---
current_user_id = st.session_state.authenticated_user['user_id']
current_username = st.session_state.authenticated_user['username']

# --- Process Pending File Save ---
if st.session_state.get('trigger_file_save_processing') and st.session_state.get('pending_file_to_save_content') is not None:
    # ... (implementation as before, no changes needed here for user scoping itself)
    file_content_to_save_bytes = st.session_state.pending_file_to_save_content
    original_name_to_save = st.session_state.pending_file_to_save_name
    st.session_state.trigger_file_save_processing = False
    st.session_state.pending_file_to_save_content = None
    st.session_state.pending_file_to_save_name = None
    temp_uploaded_file_for_service = BytesIO(file_content_to_save_bytes)
    temp_uploaded_file_for_service.name = original_name_to_save
    main_area_placeholder_save = st.empty()
    with main_area_placeholder_save.status(f"Saving '{original_name_to_save}' to your account...", expanded=True):
        st.write(f"Uploading and saving '{original_name_to_save}'...")
        saved_user_file_record = data_service.save_user_file(current_user_id, temp_uploaded_file_for_service)
        if saved_user_file_record:
            st.write("File saved to database record.")
            st.session_state.selected_user_file_id = saved_user_file_record.id
            st.session_state.trigger_file_load_id = saved_user_file_record.id
            st.session_state.current_file_content_for_processing = None
            st.session_state.processed_data = None
            st.session_state.column_mapping_confirmed = False
            st.session_state.user_column_mapping = None
            st.session_state.uploaded_file_name = None
            st.session_state.last_processed_mapping_for_file_id = None # Reset for new file
            logger.info(f"File '{saved_user_file_record.original_file_name}' (ID: {saved_user_file_record.id}) saved for user {current_user_id}. Triggering load.")
            st.rerun()
        else:
            st.error(f"Could not save the file '{original_name_to_save}'. Please try uploading again.")


# --- Process Pending File Load ---
if st.session_state.get('trigger_file_load_id') is not None:
    # ... (implementation as before, no changes needed here for user scoping itself)
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
                st.session_state.user_column_mapping = None
                st.session_state.last_processed_file_id = None
                st.session_state.last_processed_mapping_for_file_id = None # Reset for new file load
                logger.info(f"File ID {file_id_to_load} ('{selected_file_record_for_load.original_file_name}') loaded for user {current_user_id}. Triggering rerun.")
                st.rerun()
            else:
                st.error(f"Could not load content for file: {selected_file_record_for_load.original_file_name}.")
                st.session_state.current_file_content_for_processing = None
                if st.session_state.selected_user_file_id == file_id_to_load:
                     st.session_state.selected_user_file_id = None
                     st.session_state.uploaded_file_name = None
    elif file_id_to_load is not None:
        st.error(f"Selected file (ID: {file_id_to_load}) not found or access denied."); st.session_state.selected_user_file_id = None; st.session_state.current_file_content_for_processing = None; st.session_state.uploaded_file_name = None


# --- Initialize Main App Session State ---
default_session_state_main_app = {
    # ... (other states as before) ...
    'user_column_mapping': None,
    'column_mapping_confirmed': False,
    'csv_headers_for_mapping': None,
    'last_uploaded_file_for_mapping_id': None, # This might be superseded by last_processed_mapping_for_file_id
    'uploaded_file_bytes_for_mapper': None,
    'last_processed_mapping_for_file_id': None # New state for mapping persistence
}
# ... (rest of default_session_state_main_app and loop as before) ...
for key, value in default_session_state_main_app.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Sidebar for Authenticated User ---
# ... (Logo and Logout button as before) ...
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
if st.sidebar.button("🔒 Logout", key="logout_button_main_app_v4", use_container_width=True):
    logger.info(f"User '{current_username}' logging out.")
    keys_to_clear_on_logout = list(st.session_state.keys()) # Get all keys
    # Preserve theme, clear everything else
    for key_logout in keys_to_clear_on_logout:
        if key_logout not in ['current_theme']: # Add any other persistent states if needed
            del st.session_state[key_logout]
    # Explicitly reset auth and file states
    st.session_state.auth_flow_page = 'login'; st.session_state.authenticated_user = None
    st.session_state.selected_user_file_id = None; st.session_state.current_file_content_for_processing = None
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
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    matching_label = next((label for label, id_val in file_options.items() if id_val == st.session_state.selected_user_file_id), None)
    if matching_label: default_file_selection_label = matching_label
selected_file_label_in_sidebar = st.sidebar.selectbox(
    "Select a journal or upload new:", options=list(file_options.keys()),
    index=list(file_options.keys()).index(default_file_selection_label),
    key="select_user_file_v6_mapping" # Incremented key
)
selected_file_id_from_sidebar_dropdown = file_options.get(selected_file_label_in_sidebar)

if selected_file_id_from_sidebar_dropdown == "upload_new":
    # ... (file uploader logic as before) ...
    newly_uploaded_file_object = st.sidebar.file_uploader(
        "Upload New Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader_auth_v6_mapping",
        help="Your uploaded CSV will be saved to your account."
    )
    if newly_uploaded_file_object is not None:
        current_raw_file_id = id(newly_uploaded_file_object)
        if st.session_state.get('last_uploaded_raw_file_id_tracker') != current_raw_file_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file_object.getvalue()
            st.session_state.pending_file_to_save_name = newly_uploaded_file_object.name
            st.session_state.trigger_file_save_processing = True
            st.session_state.last_uploaded_raw_file_id_tracker = current_raw_file_id
            logger.debug(f"New file '{newly_uploaded_file_object.name}' detected for upload. Setting trigger_file_save_processing.")
            st.rerun()
elif selected_file_id_from_sidebar_dropdown is not None and \
     st.session_state.selected_user_file_id != selected_file_id_from_sidebar_dropdown:
    st.session_state.trigger_file_load_id = selected_file_id_from_sidebar_dropdown
    # When a different existing file is selected, reset the mapping process trigger for it
    st.session_state.last_processed_mapping_for_file_id = None
    logger.debug(f"User selected different existing file ID {selected_file_id_from_sidebar_dropdown}. Setting trigger_file_load_id and resetting mapping process trigger.")
    st.rerun()

if selected_file_id_from_sidebar_dropdown != "upload_new" and selected_file_id_from_sidebar_dropdown is not None:
    # ... (delete button logic as before) ...
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label_in_sidebar.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_sidebar_dropdown}_v5_mapping"):
        if data_service.delete_user_file(selected_file_id_from_sidebar_dropdown, current_user_id, permanent_delete_local_file=True):
            st.sidebar.success(f"File '{selected_file_label_in_sidebar}' marked as deleted.")
            if st.session_state.selected_user_file_id == selected_file_id_from_sidebar_dropdown:
                # Reset all relevant states
                st.session_state.selected_user_file_id = None
                st.session_state.current_file_content_for_processing = None
                st.session_state.processed_data = None; st.session_state.filtered_data = None; st.session_state.kpi_results = None
                st.session_state.uploaded_file_name = None
                st.session_state.column_mapping_confirmed = False
                st.session_state.user_column_mapping = None
                st.session_state.last_uploaded_file_for_mapping_id = None
                st.session_state.csv_headers_for_mapping = None
                st.session_state.last_processed_mapping_for_file_id = None # Reset this too
            st.rerun()
        else: st.sidebar.error("Failed to delete file.")


sidebar_manager = SidebarManager(st.session_state.get('processed_data')) # Pass current processed_data for dynamic filters
current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters
# ... (update session state from sidebar_filters as before) ...
if current_sidebar_filters:
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar: st.session_state.risk_free_rate = rfr_from_sidebar; st.session_state.kpi_results = None
    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == benchmark_ticker_from_sidebar), "None")
        st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None
    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar: st.session_state.initial_capital = initial_capital_from_sidebar; st.session_state.kpi_results = None


# --- Data Processing Pipeline (with Column Mapping Persistence) ---
active_file_content_to_process = st.session_state.get('current_file_content_for_processing')
active_file_name_for_processing = st.session_state.get('uploaded_file_name')

@log_execution_time
def get_and_process_data_with_profiling(file_obj_bytesio: BytesIO, mapping: Dict[str, str], name_for_log: str):
    # ... (implementation as before, ensure file_obj_bytesio.seek(0) is called)
    file_obj_bytesio.seek(0) # Crucial: reset pointer before reading
    return data_service.get_processed_trading_data(
        file_obj_bytesio, 
        user_column_mapping=mapping, 
        original_file_name=name_for_log # Pass original name for logging within load_and_process_data
    )

if active_file_content_to_process and active_file_name_for_processing and current_user_id and \
   st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    
    current_file_id_for_mapping = st.session_state.selected_user_file_id

    # Check if mapping needs to be loaded or UI shown for the current file
    if st.session_state.get('last_processed_mapping_for_file_id') != current_file_id_for_mapping:
        logger.info(f"Processing mapping for newly selected/loaded file ID: {current_file_id_for_mapping}")
        st.session_state.last_processed_mapping_for_file_id = current_file_id_for_mapping
        
        loaded_mapping = data_service.get_user_column_mapping(current_user_id, current_file_id_for_mapping)
        
        if loaded_mapping:
            st.session_state.user_column_mapping = loaded_mapping
            st.session_state.column_mapping_confirmed = True
            st.session_state.csv_headers_for_mapping = None # Not needed if map is loaded
            st.session_state.uploaded_file_bytes_for_mapper = None # Not needed
            logger.info(f"Loaded saved column mapping for file ID: {current_file_id_for_mapping}")
            # Force data reprocessing with this loaded mapping
            st.session_state.last_processed_file_id = None 
        else:
            logger.info(f"No saved mapping for file ID: {current_file_id_for_mapping}. Will show mapping UI.")
            st.session_state.user_column_mapping = None
            st.session_state.column_mapping_confirmed = False
            # Prepare headers for the UI
            try:
                active_file_content_to_process.seek(0)
                # Use chardet for encoding detection if needed, or assume UTF-8 for simplicity here
                # For robustness, data_processing.py's load_and_process_data handles encoding.
                # Here, for header peeking, a simple read_csv might suffice or fail if encoding is unusual.
                header_peek_io = BytesIO(active_file_content_to_process.getvalue())
                df_peek = pd.read_csv(header_peek_io, nrows=5) 
                st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
                active_file_content_to_process.seek(0) # Reset for ColumnMapperUI
                st.session_state.uploaded_file_bytes_for_mapper = BytesIO(active_file_content_to_process.getvalue()) # Pass a fresh BytesIO
            except Exception as e_header:
                logger.error(f"Could not read CSV headers from '{active_file_name_for_processing}' for mapping: {e_header}", exc_info=True)
                display_custom_message(f"Error reading '{active_file_name_for_processing}' for mapping: {e_header}. Ensure it's a valid CSV.", "error")
                st.session_state.csv_headers_for_mapping = None
                st.session_state.current_file_content_for_processing = None # Clear to avoid reprocessing bad file
                st.stop() # Stop if headers can't be read

    # Display ColumnMapperUI if mapping is not confirmed (i.e., no saved map was found and loaded)
    if not st.session_state.get('column_mapping_confirmed', False) and st.session_state.get('csv_headers_for_mapping'):
        st.session_state.processed_data = None # Clear any old processed data if re-mapping
        st.session_state.filtered_data = None
        st.session_state.kpi_results = None

        column_mapper = ColumnMapperUI(
            uploaded_file_name=active_file_name_for_processing,
            uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper, # Use the fresh BytesIO
            csv_headers=st.session_state.csv_headers_for_mapping,
            conceptual_columns_map=CONCEPTUAL_COLUMNS,
            conceptual_column_types=CONCEPTUAL_COLUMN_TYPES,
            conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS,
            critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS,
            conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES,
            # initial_mapping_override=st.session_state.user_column_mapping # Pass if we want UI to prefill from a loaded (but unconfirmed) map
        )
        user_mapping_result = column_mapper.render()

        if user_mapping_result is not None:
            save_success = data_service.save_user_column_mapping(
                current_user_id, 
                current_file_id_for_mapping, 
                user_mapping_result
            )
            if save_success:
                logger.info(f"Successfully saved column mapping for file ID: {current_file_id_for_mapping}")
            else:
                logger.error(f"Failed to save column mapping for file ID: {current_file_id_for_mapping}")
                display_custom_message("Error: Could not save your column mapping preferences. Please try again.", "error")
            
            st.session_state.user_column_mapping = user_mapping_result
            st.session_state.column_mapping_confirmed = True
            st.session_state.last_processed_file_id = None # Force reprocessing with new/confirmed mapping
            st.session_state.last_processed_mapping_for_file_id = current_file_id_for_mapping # Mark as processed
            logger.info(f"Column mapping confirmed and saved for '{active_file_name_for_processing}'. Rerunning for data processing.")
            st.rerun()
        else:
            display_custom_message("Please complete and confirm column mapping to proceed.", "info", icon="⚙️")
            st.stop()
            
    # Process data if mapping is confirmed and mapping exists
    if st.session_state.get('column_mapping_confirmed') and st.session_state.get('user_column_mapping'):
        # Ensure we use the correct file identifier for checking if reprocessing is needed
        # current_file_id_for_mapping is the st.session_state.selected_user_file_id
        if st.session_state.last_processed_file_id != current_file_id_for_mapping or \
           st.session_state.processed_data is None:
            
            with st.spinner(f"Processing '{active_file_name_for_processing}' with selected mapping..."):
                active_file_content_to_process.seek(0) # Ensure BytesIO is reset
                st.session_state.processed_data = get_and_process_data_with_profiling(
                    active_file_content_to_process, 
                    st.session_state.user_column_mapping,
                    active_file_name_for_processing # Pass name for logging inside
                )
            
            st.session_state.last_processed_file_id = current_file_id_for_mapping
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']:
                st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data 
            
            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
                display_custom_message(f"Successfully processed '{active_file_name_for_processing}'. Apply filters or view analysis.", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty:
                display_custom_message(f"Processing of '{active_file_name_for_processing}' resulted in empty data. Check mapping or file content.", "warning")
            else: 
                display_custom_message(f"Failed to process '{active_file_name_for_processing}'. Review mapping or file. See logs for details.", "error")
                st.session_state.column_mapping_confirmed = False 
                st.session_state.user_column_mapping = None
                st.session_state.last_processed_mapping_for_file_id = None # Allow re-attempt
                st.session_state.current_file_content_for_processing = None
                st.session_state.selected_user_file_id = None
                st.rerun()

elif not active_file_content_to_process and st.session_state.authenticated_user:
    # ... (logic for clearing states when no file is active, as before) ...
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        logger.info("No active file content. Clearing previous processed data and related states.")
        keys_to_clear_no_active = [
            'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals',
            'uploaded_file_name', 'last_processed_file_id', 'user_column_mapping',
            'column_mapping_confirmed', 'csv_headers_for_mapping',
            'last_uploaded_file_for_mapping_id', 'benchmark_daily_returns',
            'max_drawdown_period_details', 'uploaded_file_bytes_for_mapper',
            'current_file_content_for_processing', 'last_processed_mapping_for_file_id' # Also clear this
        ]
        for key_val in keys_to_clear_no_active:
            if key_val in st.session_state: st.session_state[key_val] = None
        if st.session_state.get('selected_user_file_id') is not None:
            st.session_state.selected_user_file_id = None


# --- Data Filtering, Benchmark Fetching, KPI Calculation (as before, ensure robust checks) ---
# ... (rest of the app.py as before, no changes needed in these downstream sections for this specific update) ...
@log_execution_time
def filter_data_with_profiling(df, filters, col_map):
    if df is None or df.empty:
        logger.warning("filter_data_with_profiling called with empty or None DataFrame.")
        return pd.DataFrame() 
    return data_service.filter_data(df, filters, col_map)

if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty and st.session_state.get('sidebar_filters'):
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."):
            st.session_state.filtered_data = filter_data_with_profiling(
                st.session_state.processed_data,
                st.session_state.sidebar_filters,
                EXPECTED_COLUMNS # This should map conceptual keys to their actual names in processed_data
            )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None

if st.session_state.get('filtered_data') is not None and not st.session_state.filtered_data.empty:
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
                try:
                    dates_for_bm_filtered = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                    if not dates_for_bm_filtered.empty:
                        min_d_filtered, max_d_filtered = dates_for_bm_filtered.min(), dates_for_bm_filtered.max()
                        if pd.notna(min_d_filtered) and pd.notna(max_d_filtered) and (max_d_filtered.date() >= min_d_filtered.date()):
                            min_d_str_to_fetch, max_d_str_to_fetch = min_d_filtered.strftime('%Y-%m-%d'), max_d_filtered.strftime('%Y-%m-%d')
                except Exception as e_date_bm: logger.error(f"Error processing dates for benchmark: {e_date_bm}", exc_info=True)
            if min_d_str_to_fetch and max_d_str_to_fetch:
                with st.spinner(f"Fetching benchmark: {st.session_state.selected_benchmark_display_name} ({selected_ticker})..."):
                    st.session_state.benchmark_daily_returns = get_benchmark_data_static(selected_ticker, min_d_str_to_fetch, max_d_str_to_fetch)
                st.session_state.last_fetched_benchmark_ticker = selected_ticker
                st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty:
                    display_custom_message(f"Could not fetch benchmark data for {selected_ticker} for {min_d_str_to_fetch} to {max_d_str_to_fetch}.", "warning")
            else: logger.warning(f"Cannot fetch benchmark {selected_ticker}: invalid date range."); st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None
    elif st.session_state.benchmark_daily_returns is not None:
        st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None

@log_execution_time
def get_core_kpis_with_profiling(df, rfr, benchmark_returns, capital):
    if df is None or df.empty: return {'error': 'Input data for KPIs is empty.'}
    return analysis_service_instance.get_core_kpis(df, rfr, benchmark_returns, capital)

@log_execution_time
def get_advanced_drawdown_analysis_with_profiling(equity_series):
    if equity_series is None or equity_series.empty: return {'error': 'Equity series for drawdown analysis is empty.'}
    return analysis_service_instance.get_advanced_drawdown_analysis(equity_series)

if st.session_state.get('filtered_data') is not None and not st.session_state.filtered_data.empty:
    current_kpi_state_id_parts = [
        st.session_state.filtered_data.shape, st.session_state.risk_free_rate,
        st.session_state.initial_capital, st.session_state.selected_benchmark_ticker
    ]
    if st.session_state.get('benchmark_daily_returns') is not None and not st.session_state.benchmark_daily_returns.empty:
        try: benchmark_hash = pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum(); current_kpi_state_id_parts.append(benchmark_hash)
        except Exception as e_hash_bm: logger.warning(f"Hashing benchmark failed: {e_hash_bm}. Using shape."); current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else: current_kpi_state_id_parts.append(None)
    current_kpi_state_id = tuple(current_kpi_state_id_parts)

    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info(f"Recalculating KPIs. Current state ID: {current_kpi_state_id}, Last: {st.session_state.last_kpi_calc_state_id}")
        with st.spinner("Calculating key performance indicators..."):
            kpi_res = get_core_kpis_with_profiling(st.session_state.filtered_data, st.session_state.risk_free_rate, st.session_state.benchmark_daily_returns, st.session_state.initial_capital)
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res; st.session_state.last_kpi_calc_state_id = current_kpi_state_id
                date_col_dd, cum_pnl_col_dd = EXPECTED_COLUMNS.get('date'), 'cumulative_pnl'
                equity_series_dd = pd.Series(dtype=float)
                if date_col_dd and cum_pnl_col_dd and date_col_dd in st.session_state.filtered_data and cum_pnl_col_dd in st.session_state.filtered_data:
                    df_eq_dd = st.session_state.filtered_data[[date_col_dd, cum_pnl_col_dd]].copy()
                    df_eq_dd[date_col_dd] = pd.to_datetime(df_eq_dd[date_col_dd], errors='coerce')
                    df_eq_dd.dropna(subset=[date_col_dd, cum_pnl_col_dd], inplace=True)
                    if not df_eq_dd.empty: equity_series_dd = df_eq_dd.set_index(date_col_dd)[cum_pnl_col_dd].sort_index()
                if not equity_series_dd.empty and len(equity_series_dd) >= 5:
                    adv_dd_res = get_advanced_drawdown_analysis_with_profiling(equity_series_dd)
                    st.session_state.max_drawdown_period_details = adv_dd_res.get('max_drawdown_details') if adv_dd_res and 'error' not in adv_dd_res else None
                else: st.session_state.max_drawdown_period_details = None; logger.info("Not enough data for advanced drawdown.")
                pnl_col_ci = EXPECTED_COLUMNS.get('pnl')
                if pnl_col_ci and pnl_col_ci in st.session_state.filtered_data:
                    pnl_s_ci = st.session_state.filtered_data[pnl_col_ci].dropna()
                    if len(pnl_s_ci) >= 10: st.session_state.kpi_confidence_intervals = analysis_service_instance.get_bootstrapped_kpi_cis(st.session_state.filtered_data, ['avg_trade_pnl', 'win_rate', 'sharpe_ratio']) or {}
                    else: st.session_state.kpi_confidence_intervals = {}; logger.info("Not enough PnL data for CIs.")
                else: st.session_state.kpi_confidence_intervals = {}; logger.info("PnL column not found for CIs.")
            else:
                error_msg = kpi_res.get('error', 'Unknown error') if kpi_res else 'KPI service did not return results'
                display_custom_message(f"KPI calculation error: {error_msg}. Check data and filters.", "error")
                st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None
elif st.session_state.get('filtered_data') is not None and st.session_state.filtered_data.empty:
    if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches the current filter criteria.", "info")
    st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None


def main_page_layout():
    # ... (implementation as before) ...
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-section'><h1 class='welcome-title'>Trading Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='welcome-subtitle'>Powered by {PAGE_CONFIG_APP_TITLE}</p></div>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Unlock insights from your trading data.</p>", unsafe_allow_html=True)
    if not st.session_state.get('current_file_content_for_processing') and not st.session_state.get('processed_data'):
        st.info("No trading journal is currently loaded. Please select an existing journal or upload a new one using the sidebar to begin analysis.", icon="📄")
    st.markdown("<h2 class='features-title' style='text-align: center; color: var(--secondary-color); margin-top: 2rem;'>Get Started</h2>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3, gap="large")
    with col1: st.markdown("<div class='feature-item'><h4>📄 Manage Files</h4><p>Upload new journals or select existing ones from your saved files via the sidebar.</p></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='feature-item'><h4>📊 Analyze</h4><p>Explore performance metrics once data is loaded and processed.</p></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='feature-item'><h4>💡 Discover</h4><p>Use filters and benchmark comparisons for deeper insights.</p></div>", unsafe_allow_html=True)
    st.markdown("<br><div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py"
    if os.path.exists(user_guide_page_path):
        if st.button("📘 Read User Guide", key="welcome_guide_btn_auth_v3"): st.switch_page(user_guide_page_path)
    else: st.markdown("<p>User guide not found.</p>", unsafe_allow_html=True); logger.warning(f"User Guide not found: {user_guide_page_path}")
    st.markdown("</div></div>", unsafe_allow_html=True)


# Determine what to display in the main area
processed_data_main = st.session_state.get('processed_data')
condition_for_main_layout_display = not st.session_state.get('current_file_content_for_processing') and \
    not (
        st.session_state.get('column_mapping_confirmed') and \
        processed_data_main is not None and not processed_data_main.empty
    )

if condition_for_main_layout_display:
    main_page_layout()
elif (st.session_state.get('processed_data') is None or st.session_state.get('processed_data').empty) and \
     st.session_state.get('current_file_content_for_processing') and \
     not st.session_state.get('column_mapping_confirmed'): # Added check for not column_mapping_confirmed
    if not st.session_state.get('csv_headers_for_mapping'):
         display_custom_message("Preparing data for column mapping. Please wait or check file.", "info")


scroll_buttons_component = ScrollButtons()
scroll_buttons_component.render()
logger.info(f"App run cycle finished for user '{current_username}'. Active file ID: {st.session_state.get('selected_user_file_id')}, Processed: {st.session_state.get('processed_data') is not None and not st.session_state.get('processed_data').empty}")

