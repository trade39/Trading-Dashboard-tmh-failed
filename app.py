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
if 'trigger_file_save_processing' not in st.session_state: st.session_state.trigger_file_save_processing = False
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

# --- Process Pending File Save ---
if st.session_state.get('trigger_file_save_processing') and st.session_state.get('pending_file_to_save_content') is not None:
    file_content_to_save_bytes = st.session_state.pending_file_to_save_content
    original_name_to_save = st.session_state.pending_file_to_save_name

    # Reset triggers immediately
    st.session_state.trigger_file_save_processing = False
    st.session_state.pending_file_to_save_content = None
    st.session_state.pending_file_to_save_name = None

    temp_uploaded_file_for_service = BytesIO(file_content_to_save_bytes)
    temp_uploaded_file_for_service.name = original_name_to_save

    main_area_placeholder_save = st.empty() # Use a unique placeholder if needed
    with main_area_placeholder_save.status(f"Saving '{original_name_to_save}' to your account...", expanded=True):
        st.write(f"Uploading and saving '{original_name_to_save}'...")
        saved_user_file_record = data_service.save_user_file(current_user_id, temp_uploaded_file_for_service)

        if saved_user_file_record:
            st.write("File saved to database record.")
            
            # Set selected_user_file_id for the sidebar to correctly show the new file selected.
            st.session_state.selected_user_file_id = saved_user_file_record.id
            
            # CRITICAL FIX: Trigger the load process for the newly saved file.
            # This ensures the "Process Pending File Load" block will execute next.
            st.session_state.trigger_file_load_id = saved_user_file_record.id
            
            # Reset other relevant states. The load block will populate current_file_content_for_processing and uploaded_file_name.
            st.session_state.current_file_content_for_processing = None
            st.session_state.processed_data = None
            st.session_state.column_mapping_confirmed = False
            st.session_state.uploaded_file_name = None # Let the load block set this

            logger.info(f"File '{saved_user_file_record.original_file_name}' (ID: {saved_user_file_record.id}) saved for user {current_user_id}. Triggering load sequence.")
            st.rerun()
        else:
            st.error(f"Could not save the file '{original_name_to_save}'. Please try uploading again.")
            # No rerun here, user sees the error and can retry.

# --- Process Pending File Load ---
if st.session_state.get('trigger_file_load_id') is not None:
    file_id_to_load = st.session_state.trigger_file_load_id
    st.session_state.trigger_file_load_id = None # Consume the trigger immediately

    user_files_for_load_check = data_service.list_user_files(current_user_id)
    selected_file_record_for_load = next((f for f in user_files_for_load_check if f.id == file_id_to_load), None)

    if selected_file_record_for_load:
        main_area_placeholder_load = st.empty() # Use a unique placeholder
        with main_area_placeholder_load.status(f"Loading '{selected_file_record_for_load.original_file_name}'...", expanded=True):
            st.write(f"Fetching content for '{selected_file_record_for_load.original_file_name}'...")
            file_content_bytesio = data_service.get_user_file_content(file_id_to_load, current_user_id)

            if file_content_bytesio:
                st.session_state.current_file_content_for_processing = file_content_bytesio
                st.session_state.selected_user_file_id = file_id_to_load # Confirm/update selected ID
                st.session_state.uploaded_file_name = selected_file_record_for_load.original_file_name
                
                # Reset downstream states as new content is loaded
                st.session_state.processed_data = None
                st.session_state.column_mapping_confirmed = False
                st.session_state.user_column_mapping = None # Important to reset mapping too
                st.session_state.last_processed_file_id = None # Ensure reprocessing

                logger.info(f"File ID {file_id_to_load} ('{selected_file_record_for_load.original_file_name}') loaded for user {current_user_id}. Triggering rerun for processing.")
                st.rerun()
            else:
                st.error(f"Could not load the content for file: {selected_file_record_for_load.original_file_name}.")
                st.session_state.current_file_content_for_processing = None
                # If load fails, and this was the selected file, potentially clear selected_user_file_id
                # or allow user to select another. For now, just clearing content.
                if st.session_state.selected_user_file_id == file_id_to_load:
                     st.session_state.selected_user_file_id = None # Clear selection if its content failed to load
                     st.session_state.uploaded_file_name = None

    elif file_id_to_load is not None: # Trigger was set, but file_id didn't match any user files
        st.error(f"Selected file (ID: {file_id_to_load}) not found or access denied.")
        st.session_state.selected_user_file_id = None
        st.session_state.current_file_content_for_processing = None
        st.session_state.uploaded_file_name = None


# --- Initialize Main App Session State (if not already done) ---
default_session_state_main_app = {
    'app_initialized': True,
    'processed_data': None,
    'filtered_data': None,
    'kpi_results': None,
    'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE,
    'uploaded_file_name': None,
    'uploaded_file_bytes_for_mapper': None,
    'last_processed_file_id': None,
    'user_column_mapping': None,
    'column_mapping_confirmed': False,
    'csv_headers_for_mapping': None,
    'last_uploaded_file_for_mapping_id': None,
    'last_applied_filters': None,
    'sidebar_filters': None,
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER,
    'benchmark_daily_returns': None,
    'initial_capital': 100000.0,
    'last_fetched_benchmark_ticker': None,
    'last_benchmark_data_filter_shape': None,
    'last_kpi_calc_state_id': None,
    'max_drawdown_period_details': None
}
default_session_state_main_app['selected_benchmark_display_name'] = next(
    (name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == default_session_state_main_app['selected_benchmark_ticker']), "None"
)

for key, value in default_session_state_main_app.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Sidebar for Authenticated User ---
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

# Determine default selection for the selectbox
# If a file was just uploaded and its ID is in selected_user_file_id, make that the default.
# Otherwise, if selected_user_file_id points to an existing file, use that.
# Fallback to "Upload New File..."
default_file_selection_label = "✨ Upload New File..."
if st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new":
    # Find the label corresponding to the selected_user_file_id
    matching_label = next((label for label, id_val in file_options.items() if id_val == st.session_state.selected_user_file_id), None)
    if matching_label:
        default_file_selection_label = matching_label

selected_file_label_in_sidebar = st.sidebar.selectbox(
    "Select a journal or upload new:", options=list(file_options.keys()),
    index=list(file_options.keys()).index(default_file_selection_label), # Ensure index is valid
    key="select_user_file_v5" # Incremented key to avoid state issues if structure changed
)
selected_file_id_from_sidebar_dropdown = file_options.get(selected_file_label_in_sidebar)


if selected_file_id_from_sidebar_dropdown == "upload_new":
    newly_uploaded_file_object = st.sidebar.file_uploader(
        "Upload New Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader_auth_v5_trigger",
        help="Your uploaded CSV will be saved to your account."
    )
    if newly_uploaded_file_object is not None:
        current_raw_file_id = id(newly_uploaded_file_object) # Get a unique ID for the raw upload
        # Only trigger save if this specific upload hasn't been processed yet
        if st.session_state.get('last_uploaded_raw_file_id_tracker') != current_raw_file_id:
            st.session_state.pending_file_to_save_content = newly_uploaded_file_object.getvalue()
            st.session_state.pending_file_to_save_name = newly_uploaded_file_object.name
            st.session_state.trigger_file_save_processing = True
            st.session_state.last_uploaded_raw_file_id_tracker = current_raw_file_id # Track this upload
            logger.debug(f"New file '{newly_uploaded_file_object.name}' detected for upload. Setting trigger_file_save_processing.")
            st.rerun()
elif selected_file_id_from_sidebar_dropdown is not None and \
     st.session_state.selected_user_file_id != selected_file_id_from_sidebar_dropdown:
    # This means user selected a DIFFERENT existing file from the dropdown
    st.session_state.trigger_file_load_id = selected_file_id_from_sidebar_dropdown
    logger.debug(f"User selected different existing file ID {selected_file_id_from_sidebar_dropdown}. Setting trigger_file_load_id.")
    st.rerun()

if selected_file_id_from_sidebar_dropdown != "upload_new" and selected_file_id_from_sidebar_dropdown is not None:
    # This block is for actions on an ALREADY selected (and presumably loaded or being loaded) file
    if st.sidebar.button(f"🗑️ Delete '{selected_file_label_in_sidebar.split(' (Uploaded:')[0]}'", key=f"delete_file_{selected_file_id_from_sidebar_dropdown}_v4"): # Incremented key
        if data_service.delete_user_file(selected_file_id_from_sidebar_dropdown, current_user_id, permanent_delete_local_file=True):
            st.sidebar.success(f"File '{selected_file_label_in_sidebar}' marked as deleted.")
            # If the deleted file was the currently active one, clear relevant states
            if st.session_state.selected_user_file_id == selected_file_id_from_sidebar_dropdown:
                st.session_state.selected_user_file_id = None
                st.session_state.current_file_content_for_processing = None
                st.session_state.processed_data = None
                st.session_state.uploaded_file_name = None
                st.session_state.column_mapping_confirmed = False
                st.session_state.user_column_mapping = None
                st.session_state.last_uploaded_file_for_mapping_id = None
                st.session_state.csv_headers_for_mapping = None

            st.rerun()
        else: st.sidebar.error("Failed to delete file.")

sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters

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

# --- Data Processing Pipeline ---
active_file_content_to_process = st.session_state.get('current_file_content_for_processing')
active_file_name_for_processing = st.session_state.get('uploaded_file_name')
# Use selected_user_file_id as the primary identifier if available and valid, otherwise fallback to name (less reliable for uniqueness)
active_processing_file_identifier = st.session_state.selected_user_file_id if (st.session_state.selected_user_file_id and st.session_state.selected_user_file_id != "upload_new") else active_file_name_for_processing

@log_execution_time
def get_and_process_data_with_profiling(file_obj, mapping, name):
    # Ensure file_obj is BytesIO and reset its pointer
    if hasattr(file_obj, 'getvalue') and not isinstance(file_obj, BytesIO):
        # This case might happen if somehow a Streamlit UploadedFile object is passed directly
        file_bytes_io = BytesIO(file_obj.getvalue())
    elif isinstance(file_obj, BytesIO):
        file_bytes_io = file_obj
    else:
        logger.error("get_and_process_data_with_profiling: file_obj is not compatible (must be BytesIO or have getvalue).")
        return None
    
    file_bytes_io.seek(0) # IMPORTANT: Always reset pointer before reading
    return data_service.get_processed_trading_data(file_bytes_io, user_column_mapping=mapping, original_file_name=name)

if active_file_content_to_process and active_file_name_for_processing:
    # Check if mapping is needed for the current file content
    if st.session_state.last_uploaded_file_for_mapping_id != active_processing_file_identifier or \
       not st.session_state.column_mapping_confirmed:
        
        logger.info(f"File '{active_file_name_for_processing}' (ID: {active_processing_file_identifier}) requires column mapping or re-mapping.")
        # Reset states for new mapping session
        st.session_state.column_mapping_confirmed = False
        st.session_state.user_column_mapping = None
        st.session_state.processed_data = None # Clear old processed data if re-mapping
        st.session_state.filtered_data = None
        st.session_state.kpi_results = None # KPIs depend on processed data

        st.session_state.last_uploaded_file_for_mapping_id = active_processing_file_identifier
        
        try:
            # Ensure the BytesIO object is correctly prepared for header reading
            active_file_content_to_process.seek(0)
            # Create a fresh BytesIO for the header peek to avoid issues with shared objects if any
            header_peek_io = BytesIO(active_file_content_to_process.getvalue())
            df_peek = pd.read_csv(header_peek_io, nrows=5)
            st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
            
            # Prepare the full content for the mapper UI (it might need the full file for previews)
            active_file_content_to_process.seek(0) # Reset again for the mapper
            st.session_state.uploaded_file_bytes_for_mapper = active_file_content_to_process 
            
        except Exception as e_header:
            logger.error(f"Could not read CSV headers from active file '{active_file_name_for_processing}': {e_header}", exc_info=True)
            display_custom_message(f"Error reading '{active_file_name_for_processing}' for mapping: {e_header}. Please ensure it's a valid CSV.", "error")
            st.session_state.csv_headers_for_mapping = None
            # Potentially clear active_file_content_to_process to prevent further errors, or let user re-select
            st.session_state.current_file_content_for_processing = None 
            st.stop() # Stop further execution in this run if headers can't be read

    # If headers are available and mapping is not yet confirmed, show mapper
    if st.session_state.csv_headers_for_mapping and not st.session_state.column_mapping_confirmed:
        column_mapper = ColumnMapperUI(
            uploaded_file_name=active_file_name_for_processing,
            # Pass a fresh BytesIO for the mapper to avoid shared state issues with the original active_file_content_to_process
            uploaded_file_bytes=BytesIO(st.session_state.uploaded_file_bytes_for_mapper.getvalue()), 
            csv_headers=st.session_state.csv_headers_for_mapping,
            conceptual_columns_map=CONCEPTUAL_COLUMNS,
            conceptual_column_types=CONCEPTUAL_COLUMN_TYPES,
            conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS,
            critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS,
            conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES
        )
        user_mapping_result = column_mapper.render()
        if user_mapping_result is not None:
            st.session_state.user_column_mapping = user_mapping_result
            st.session_state.column_mapping_confirmed = True
            st.session_state.last_processed_file_id = None # Force reprocessing with new mapping
            logger.info(f"Column mapping confirmed for '{active_file_name_for_processing}'. Rerunning for processing.")
            st.rerun()
        else:
            # Mapper is still active or user hasn't confirmed; stop to prevent processing without mapping.
            display_custom_message("Please complete and confirm column mapping to proceed.", "info", icon="⚙️")
            st.stop()

    # If mapping is confirmed and we have a user mapping, proceed to process data
    if st.session_state.column_mapping_confirmed and st.session_state.user_column_mapping:
        # Process data if it's a new file, new mapping, or data hasn't been processed yet
        if st.session_state.last_processed_file_id != active_processing_file_identifier or \
           st.session_state.processed_data is None: # Check for None explicitly
            
            with st.spinner(f"Processing '{active_file_name_for_processing}' with selected mapping..."):
                # Ensure active_file_content_to_process is reset before passing
                active_file_content_to_process.seek(0)
                st.session_state.processed_data = get_and_process_data_with_profiling(
                    active_file_content_to_process, # This is already a BytesIO from the load stage
                    st.session_state.user_column_mapping,
                    active_file_name_for_processing
                )
            
            st.session_state.last_processed_file_id = active_processing_file_identifier
            # Reset downstream data that depends on processed_data
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']:
                st.session_state[key_to_reset] = None
            
            # Initialize filtered_data with the newly processed_data
            st.session_state.filtered_data = st.session_state.processed_data 
            
            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
                display_custom_message(f"Successfully processed '{active_file_name_for_processing}'. Apply filters or view analysis.", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty:
                display_custom_message(f"Processing of '{active_file_name_for_processing}' resulted in empty data. Check mapping or file content.", "warning")
                # Keep mapping confirmed, but user might need to adjust if data is unexpectedly empty
            else: # processed_data is None
                display_custom_message(f"Failed to process '{active_file_name_for_processing}'. Review mapping or file. See logs for details.", "error")
                # If processing fails, it's good to allow re-mapping
                st.session_state.column_mapping_confirmed = False 
                st.session_state.user_column_mapping = None
                st.session_state.current_file_content_for_processing = None # Clear content to force re-load/re-select
                st.session_state.selected_user_file_id = None # Clear selection
                st.rerun() # Rerun to reflect cleared state

elif not active_file_content_to_process and st.session_state.authenticated_user:
    # This block handles the case where no file content is loaded (e.g., after logout/login, or if a load failed)
    # If there was previously processed data, clear it to avoid showing stale info.
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        logger.info("No active file content. Clearing previous processed data and related states.")
        keys_to_clear_no_active = [
            'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals',
            'uploaded_file_name', 'last_processed_file_id', 'user_column_mapping',
            'column_mapping_confirmed', 'csv_headers_for_mapping',
            'last_uploaded_file_for_mapping_id', 'benchmark_daily_returns',
            'max_drawdown_period_details', 'uploaded_file_bytes_for_mapper',
            'current_file_content_for_processing' # also clear this
        ]
        for key_val in keys_to_clear_no_active:
            if key_val in st.session_state:
                st.session_state[key_val] = None
        # selected_user_file_id should also be None if no content is active
        if st.session_state.get('selected_user_file_id') is not None:
            st.session_state.selected_user_file_id = None
            # st.rerun() # Rerun if we cleared selected_user_file_id to update sidebar
            # Only rerun if necessary, to avoid too many reruns if already in a stable "no data" state.

# --- Data Filtering, Benchmark Fetching, KPI Calculation (as before, ensure robust checks) ---
@log_execution_time
def filter_data_with_profiling(df, filters, col_map):
    if df is None or df.empty:
        logger.warning("filter_data_with_profiling called with empty or None DataFrame.")
        return pd.DataFrame() # Return empty DataFrame
    return data_service.filter_data(df, filters, col_map)

if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty and st.session_state.get('sidebar_filters'):
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."):
            st.session_state.filtered_data = filter_data_with_profiling(
                st.session_state.processed_data,
                st.session_state.sidebar_filters,
                EXPECTED_COLUMNS
            )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        # Reset KPIs as filters changed
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']: # benchmark_daily_returns might also need re-fetch if date range changes significantly
            st.session_state[key_to_reset] = None

if st.session_state.get('filtered_data') is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None: refetch_benchmark = True
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker: refetch_benchmark = True
        # Check if filtered data shape changed, indicating date range might have changed
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape: refetch_benchmark = True

        if refetch_benchmark:
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date')
            min_d_str_to_fetch, max_d_str_to_fetch = None, None
            if date_col_conceptual in st.session_state.filtered_data.columns:
                try:
                    # Ensure date column is datetime
                    dates_for_bm_filtered = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                    if not dates_for_bm_filtered.empty:
                        min_d_filtered, max_d_filtered = dates_for_bm_filtered.min(), dates_for_bm_filtered.max()
                        if pd.notna(min_d_filtered) and pd.notna(max_d_filtered) and (max_d_filtered.date() >= min_d_filtered.date()): # Ensure valid range
                            min_d_str_to_fetch, max_d_str_to_fetch = min_d_filtered.strftime('%Y-%m-%d'), max_d_filtered.strftime('%Y-%m-%d')
                except Exception as e_date_bm:
                    logger.error(f"Error processing dates for benchmark fetching: {e_date_bm}", exc_info=True)

            if min_d_str_to_fetch and max_d_str_to_fetch:
                with st.spinner(f"Fetching benchmark: {st.session_state.selected_benchmark_display_name} ({selected_ticker})..."):
                    st.session_state.benchmark_daily_returns = get_benchmark_data_static(selected_ticker, min_d_str_to_fetch, max_d_str_to_fetch)
                st.session_state.last_fetched_benchmark_ticker = selected_ticker
                st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape # Store shape after filtering
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty:
                    display_custom_message(f"Could not fetch benchmark data for {selected_ticker} for the period {min_d_str_to_fetch} to {max_d_str_to_fetch}.", "warning")
            else:
                logger.warning(f"Cannot fetch benchmark {selected_ticker}: invalid or insufficient date range from filtered data.")
                st.session_state.benchmark_daily_returns = None # Ensure it's None if not fetched
            st.session_state.kpi_results = None # Force KPI recalc after benchmark change

    elif st.session_state.benchmark_daily_returns is not None: # No ticker selected, clear benchmark
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None # Force KPI recalc

@log_execution_time
def get_core_kpis_with_profiling(df, rfr, benchmark_returns, capital):
    if df is None or df.empty: return {'error': 'Input data for KPIs is empty.'}
    return analysis_service_instance.get_core_kpis(df, rfr, benchmark_returns, capital)

@log_execution_time
def get_advanced_drawdown_analysis_with_profiling(equity_series):
    if equity_series is None or equity_series.empty: return {'error': 'Equity series for drawdown analysis is empty.'}
    return analysis_service_instance.get_advanced_drawdown_analysis(equity_series)

if st.session_state.get('filtered_data') is not None and not st.session_state.filtered_data.empty:
    # Construct a unique ID for the current state affecting KPIs
    current_kpi_state_id_parts = [
        st.session_state.filtered_data.shape,
        st.session_state.risk_free_rate,
        st.session_state.initial_capital,
        st.session_state.selected_benchmark_ticker
    ]
    if st.session_state.get('benchmark_daily_returns') is not None and not st.session_state.benchmark_daily_returns.empty:
        try:
            # Hash the benchmark series for a more robust change detection
            benchmark_hash = pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum()
            current_kpi_state_id_parts.append(benchmark_hash)
        except Exception as e_hash_bm:
            logger.warning(f"Hashing benchmark daily returns failed: {e_hash_bm}. Using shape as fallback for state ID.")
            current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else:
        current_kpi_state_id_parts.append(None) # Placeholder if no benchmark
    current_kpi_state_id = tuple(current_kpi_state_id_parts)

    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info(f"Recalculating KPIs. Current state ID: {current_kpi_state_id}, Last state ID: {st.session_state.last_kpi_calc_state_id}")
        with st.spinner("Calculating key performance indicators..."):
            kpi_res = get_core_kpis_with_profiling(
                st.session_state.filtered_data,
                st.session_state.risk_free_rate,
                st.session_state.benchmark_daily_returns, # Pass the actual series
                st.session_state.initial_capital
            )
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res
                st.session_state.last_kpi_calc_state_id = current_kpi_state_id

                # Advanced Drawdown Analysis
                date_col_dd, cum_pnl_col_dd = EXPECTED_COLUMNS.get('date'), 'cumulative_pnl' # Assuming 'cumulative_pnl' is standard
                equity_series_dd = pd.Series(dtype=float)
                if date_col_dd and cum_pnl_col_dd and \
                   date_col_dd in st.session_state.filtered_data and \
                   cum_pnl_col_dd in st.session_state.filtered_data:
                    df_eq_dd = st.session_state.filtered_data[[date_col_dd, cum_pnl_col_dd]].copy()
                    df_eq_dd[date_col_dd] = pd.to_datetime(df_eq_dd[date_col_dd], errors='coerce')
                    df_eq_dd.dropna(subset=[date_col_dd, cum_pnl_col_dd], inplace=True) # Drop NA before setting index
                    if not df_eq_dd.empty:
                        equity_series_dd = df_eq_dd.set_index(date_col_dd)[cum_pnl_col_dd].sort_index()
                
                if not equity_series_dd.empty and len(equity_series_dd) >= 5: # Need some data for drawdown
                    adv_dd_res = get_advanced_drawdown_analysis_with_profiling(equity_series_dd)
                    st.session_state.max_drawdown_period_details = adv_dd_res.get('max_drawdown_details') if adv_dd_res and 'error' not in adv_dd_res else None
                else:
                    st.session_state.max_drawdown_period_details = None
                    logger.info("Not enough data or missing columns for advanced drawdown analysis.")

                # Confidence Intervals
                pnl_col_ci = EXPECTED_COLUMNS.get('pnl')
                if pnl_col_ci and pnl_col_ci in st.session_state.filtered_data:
                    pnl_s_ci = st.session_state.filtered_data[pnl_col_ci].dropna()
                    if len(pnl_s_ci) >= 10: # Need enough samples for bootstrap
                        st.session_state.kpi_confidence_intervals = analysis_service_instance.get_bootstrapped_kpi_cis(
                            st.session_state.filtered_data, ['avg_trade_pnl', 'win_rate', 'sharpe_ratio'] # Example KPIs
                        ) or {}
                    else: st.session_state.kpi_confidence_intervals = {}; logger.info("Not enough PnL data points for confidence intervals.")
                else: st.session_state.kpi_confidence_intervals = {}; logger.info("PnL column not found for confidence intervals.")
            else:
                error_msg = kpi_res.get('error', 'Unknown error during KPI calculation') if kpi_res else 'KPI service did not return results'
                display_custom_message(f"KPI calculation error: {error_msg}. Check data and filters.", "error")
                st.session_state.kpi_results = None # Ensure it's None on error
                st.session_state.kpi_confidence_intervals = {}
                st.session_state.max_drawdown_period_details = None
elif st.session_state.get('filtered_data') is not None and st.session_state.filtered_data.empty:
    # This case means filters resulted in no data
    if st.session_state.get('processed_data') is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches the current filter criteria.", "info")
    # Clear results if filtered data is empty
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}
    st.session_state.max_drawdown_period_details = None


def main_page_layout():
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-section'><h1 class='welcome-title'>Trading Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='welcome-subtitle'>Powered by {PAGE_CONFIG_APP_TITLE}</p></div>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Unlock insights from your trading data.</p>", unsafe_allow_html=True)
    
    # Message for when no data is loaded yet
    if not st.session_state.get('current_file_content_for_processing') and not st.session_state.get('processed_data'):
        st.info("No trading journal is currently loaded. Please select an existing journal or upload a new one using the sidebar to begin analysis.", icon="📄")
    
    st.markdown("<h2 class='features-title' style='text-align: center; color: var(--secondary-color); margin-top: 2rem;'>Get Started</h2>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3, gap="large")
    with col1: st.markdown("<div class='feature-item'><h4>📄 Manage Files</h4><p>Upload new journals or select existing ones from your saved files via the sidebar.</p></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='feature-item'><h4>📊 Analyze</h4><p>Explore performance metrics once data is loaded and processed.</p></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='feature-item'><h4>💡 Discover</h4><p>Use filters and benchmark comparisons for deeper insights.</p></div>", unsafe_allow_html=True)
    
    st.markdown("<br><div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py" # Assuming Streamlit multipage app structure
    if os.path.exists(user_guide_page_path):
        if st.button("📘 Read User Guide", key="welcome_guide_btn_auth_v3"):
             st.switch_page(user_guide_page_path) # Correct way to switch pages
    else: st.markdown("<p>User guide not found at expected location.</p>", unsafe_allow_html=True); logger.warning(f"User Guide not found: {user_guide_page_path}")
    st.markdown("</div></div>", unsafe_allow_html=True)

# Determine what to display in the main area
# Show welcome/initial page if no file content is active AND (no mapping is confirmed OR no data is processed OR processed_data is empty)
processed_data = st.session_state.get('processed_data')
condition_for_main_layout = not st.session_state.get('current_file_content_for_processing') and \
    not (
        st.session_state.get('column_mapping_confirmed') and \
        processed_data is not None and not processed_data.empty
    )

if condition_for_main_layout:
    main_page_layout()
# Corrected condition for the elif block
elif (st.session_state.get('processed_data') is None or st.session_state.get('processed_data').empty) and \
     st.session_state.get('current_file_content_for_processing'):
    # This case means file is loaded, but not yet successfully processed (or processing resulted in empty data).
    # The column mapper UI or error messages within the processing pipeline should handle display.
    # This block can provide a fallback message if those internal components don't display anything.
    if not st.session_state.get('csv_headers_for_mapping') and not st.session_state.get('column_mapping_confirmed'):
         display_custom_message("Preparing data for column mapping. Please wait or check file.", "info")


scroll_buttons_component = ScrollButtons()
scroll_buttons_component.render()
logger.info(f"App run cycle finished for user '{current_username}'. Active file: {st.session_state.get('uploaded_file_name')}, Processed: {st.session_state.get('processed_data') is not None and not st.session_state.get('processed_data').empty}")
