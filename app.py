# app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import datetime # Keep this import
import base64
from io import BytesIO

# --- config.py import for APP_TITLE ---
# This ensures PAGE_CONFIG_APP_TITLE is defined before st.set_page_config
try:
    from config import APP_TITLE as PAGE_CONFIG_APP_TITLE
    LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png" # Define your logo path
except ImportError:
    PAGE_CONFIG_APP_TITLE = "Trading Dashboard" # Fallback title
    LOGO_PATH_FOR_BROWSER_TAB = None # Fallback icon
    # Basic logging if config fails early
    logging.basicConfig(level=logging.INFO)
    logging.error("CRITICAL: config.py not found or APP_TITLE missing. Using fallback page title.")

st.set_page_config(
    page_title=PAGE_CONFIG_APP_TITLE,
    page_icon=LOGO_PATH_FOR_BROWSER_TAB if LOGO_PATH_FOR_BROWSER_TAB and os.path.exists(LOGO_PATH_FOR_BROWSER_TAB) else "📊", # Fallback emoji if logo not found
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4',
        'Report a bug': "https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4/issues",
        'About': f"## {PAGE_CONFIG_APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)

# --- Utility Modules ---
try:
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message, log_execution_time
except ImportError as e:
    st.error(f"Fatal Error: Could not import utility modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing utils: {e}", exc_info=True)
    st.stop()

# --- Component Modules ---
try:
    from components.sidebar_manager import SidebarManager
    from components.column_mapper_ui import ColumnMapperUI
    from components.scroll_buttons import ScrollButtons
except ImportError as e:
    st.error(f"Fatal Error: Could not import component modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing components: {e}", exc_info=True)
    st.stop()

# --- Service Modules ---
try:
    from services.data_service import DataService, get_benchmark_data_static # DataService needed for user settings
    from services.analysis_service import AnalysisService
except ImportError as e:
    st.error(f"Fatal Error: Could not import service modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing services: {e}", exc_info=True)
    st.stop()

# --- Core Application Modules (Configs) ---
try:
    from config import (
        APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS,
        CONCEPTUAL_COLUMN_CATEGORIES,
        RISK_FREE_RATE, LOG_FILE, LOG_LEVEL, LOG_FORMAT,
        DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS, EXPECTED_COLUMNS
    )
    from kpi_definitions import KPI_CONFIG
except ImportError as e:
    st.error(f"Fatal Error: Could not import configuration (config.py or kpi_definitions.py). App cannot start. Details: {e}")
    # Fallback for essential configs if main config fails
    APP_TITLE = PAGE_CONFIG_APP_TITLE # Use the already defined one
    LOG_FILE = "logs/error_app.log"; LOG_LEVEL = "ERROR"; LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    RISK_FREE_RATE = 0.02; CONCEPTUAL_COLUMNS = {"date": "Date", "pnl": "PnL"}; CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    CONCEPTUAL_COLUMN_TYPES = {}; CONCEPTUAL_COLUMN_SYNONYMS = {}; KPI_CONFIG = {}; CONCEPTUAL_COLUMN_CATEGORIES = {}
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}; DEFAULT_BENCHMARK_TICKER = "SPY"; AVAILABLE_BENCHMARKS = {}
    st.stop()

# Initialize logger (must be done after config is loaded)
logger = setup_logger(
    logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")

# Instantiate DataService early for user settings
data_service = DataService()
analysis_service_instance = AnalysisService()
logger.info("DataService and AnalysisService instantiated successfully.")


# --- Theme Management (Load from DB, then apply) ---
if 'app_theme_loaded' not in st.session_state:
    try:
        # This was the original line 97, where the error occurred.
        default_theme_from_db = data_service.get_user_setting("default_app_theme", default="dark")
        st.session_state.current_theme = default_theme_from_db
        logger.info(f"Initial theme loaded from DB: {st.session_state.current_theme}")
    except Exception as e_theme_load:
        # Log the error and fallback to a default theme
        logger.error(f"Failed to load theme setting from DataService on app startup: {e_theme_load}. Defaulting to 'dark' theme.", exc_info=True)
        st.session_state.current_theme = "dark" # Fallback theme
    st.session_state.app_theme_loaded = True # Ensure this is set so we don't retry every time if DB is down

# Ensure current_theme is always set before applying JS, even if the above block was skipped or failed.
if 'current_theme' not in st.session_state:
    logger.warning("current_theme was not set after DB attempt or fallback during initial load. Setting to 'dark' as final safety measure.")
    st.session_state.current_theme = "dark"


# Apply theme using JavaScript (must be done on every run if theme can change)
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


# Load CSS
try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path): load_css(css_file_path)
    else: logger.warning(f"style.css not found at '{css_file_path}'. Custom styles might not apply.")
except Exception as e: logger.error(f"Failed to load style.css: {e}", exc_info=True)

# Initialize session state variables (idempotent)
default_session_state = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None,
    'kpi_results': None, 'kpi_confidence_intervals': {},
    # risk_free_rate and initial_capital will be loaded from DB by SidebarManager or defaulted there
    'uploaded_file_name': None,
    'uploaded_file_bytes_for_mapper': None, 'last_processed_file_id': None,
    'user_column_mapping': None, 'column_mapping_confirmed': False,
    'csv_headers_for_mapping': None, 'last_uploaded_file_for_mapping_id': None,
    'last_applied_filters': None, 'sidebar_filters': None, 'active_tab': "📈 Overview",
    'selected_benchmark_ticker': data_service.get_user_setting("selected_benchmark_ticker", DEFAULT_BENCHMARK_TICKER), # This call might also need a try-except if DataService is generally unstable
    'benchmark_daily_returns': None,
    'last_fetched_benchmark_ticker': None, 'last_benchmark_data_filter_shape': None,
    'last_kpi_calc_state_id': None,
    'max_drawdown_period_details': None
}
for key, value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = value

# Initialize selected_benchmark_display_name based on selected_benchmark_ticker
if 'selected_benchmark_display_name' not in st.session_state:
    st.session_state.selected_benchmark_display_name = next(
        (name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == st.session_state.selected_benchmark_ticker), "None"
    )


# --- Sidebar Logo and Header ---
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"
logo_base64 = None
if os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        with open(LOGO_PATH_SIDEBAR, "rb") as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()
        logo_to_display_for_st_logo = f"data:image/png;base64,{logo_base64}"
        st.logo(logo_to_display_for_st_logo, icon_image=logo_to_display_for_st_logo)
    except Exception as e:
        logger.error(f"Error setting st.logo or encoding logo: {e}", exc_info=True)
        # Fallback to st.sidebar.image if st.logo fails
        if logo_base64:
            st.sidebar.image(f"data:image/png;base64,{logo_base64}", use_column_width='auto')
        elif os.path.exists(LOGO_PATH_SIDEBAR):
            st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')
else:
    logger.warning(f"Sidebar logo file NOT FOUND at {LOGO_PATH_SIDEBAR}")

st.sidebar.header(APP_TITLE)
st.sidebar.markdown("---")

# --- Theme Toggle Button (now interacts with DB) ---
toggle_label = "☀️ Switch to Light Mode" if st.session_state.current_theme == "dark" else "🌙 Switch to Dark Mode"
if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app_v2", help="Toggle between light and dark themes."):
    st.session_state.current_theme = "light" if st.session_state.current_theme == "dark" else "dark"
    try:
        data_service.save_user_setting("default_app_theme", st.session_state.current_theme) # Save to DB
        logger.info(f"Theme changed to {st.session_state.current_theme} and saved to DB.")
    except Exception as e_theme_save:
        logger.error(f"Failed to save theme preference to DataService: {e_theme_save}", exc_info=True)
        st.warning("Could not save your theme preference. It will reset on next session.")
    st.rerun()
st.sidebar.markdown("---")


# --- File Uploader and Sidebar Controls ---
uploaded_file = st.sidebar.file_uploader("Upload Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader_v2")

# Instantiate SidebarManager (pass data_service for it to load/save its own settings)
sidebar_manager = SidebarManager(st.session_state.get('processed_data'), data_service)
current_sidebar_filters = sidebar_manager.render_sidebar_controls() # This will now load/save its settings
st.session_state.sidebar_filters = current_sidebar_filters

# --- Update session state from sidebar filters (for global app use) ---
if current_sidebar_filters:
    # Risk-Free Rate (already handled by SidebarManager saving to its own session state key if needed)
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.get('risk_free_rate', RISK_FREE_RATE) != rfr_from_sidebar:
        st.session_state.risk_free_rate = rfr_from_sidebar
        st.session_state.kpi_results = None # Force KPI recalc
        logger.info(f"Global risk_free_rate updated from sidebar to: {rfr_from_sidebar}")


    # Benchmark Ticker
    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next(
            (name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == benchmark_ticker_from_sidebar), "None"
        )
        st.session_state.benchmark_daily_returns = None # Force re-fetch
        st.session_state.kpi_results = None # Force KPI recalc
        logger.info(f"Global benchmark_ticker updated from sidebar to: {benchmark_ticker_from_sidebar}")
        try:
            data_service.save_user_setting("selected_benchmark_ticker", benchmark_ticker_from_sidebar)
        except Exception as e_benchmark_save:
            logger.error(f"Failed to save benchmark preference: {e_benchmark_save}", exc_info=True)


    # Initial Capital (already handled by SidebarManager saving to its own session state key if needed)
    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.get('initial_capital', 100000.0) != initial_capital_from_sidebar:
        st.session_state.initial_capital = initial_capital_from_sidebar
        st.session_state.kpi_results = None # Force KPI recalc
        logger.info(f"Global initial_capital updated from sidebar to: {initial_capital_from_sidebar}")


# --- Data Loading, Mapping, and Processing Logic ---
@log_execution_time
def get_and_process_data_with_profiling(file_obj, mapping, name):
    # Pass data_service if needed by load_and_process_data, or keep as is
    return data_service.get_processed_trading_data(file_obj, user_column_mapping=mapping, original_file_name=name)

if uploaded_file is not None:
    # Unique ID for the current uploaded file instance to manage mapping state
    current_file_id_for_mapping = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}-mapping_stage"

    if st.session_state.last_uploaded_file_for_mapping_id != current_file_id_for_mapping:
        logger.info(f"New file '{uploaded_file.name}' detected for mapping. Resetting relevant session states.")
        # Reset states related to data processing and display
        for key_to_reset in ['column_mapping_confirmed', 'user_column_mapping',
                             'processed_data', 'filtered_data', 'kpi_results',
                             'kpi_confidence_intervals', 'benchmark_daily_returns',
                             'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None # Reset to None
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_uploaded_file_for_mapping_id = current_file_id_for_mapping
        try:
            # Store bytes for column mapper and potential reprocessing
            st.session_state.uploaded_file_bytes_for_mapper = BytesIO(uploaded_file.getvalue())
            st.session_state.uploaded_file_bytes_for_mapper.seek(0) # Ensure pointer is at the start
            # Peek at headers for the mapper
            df_peek = pd.read_csv(BytesIO(st.session_state.uploaded_file_bytes_for_mapper.getvalue()), nrows=5)
            st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
            st.session_state.uploaded_file_bytes_for_mapper.seek(0) # Reset pointer again for actual processing
        except Exception as e_header:
            logger.error(f"Could not read CSV headers/preview for mapping: {e_header}", exc_info=True)
            display_custom_message(f"Error reading from '{uploaded_file.name}': {e_header}. Please ensure it's a valid CSV file.", "error")
            st.session_state.csv_headers_for_mapping = None
            st.session_state.uploaded_file_bytes_for_mapper = None
            st.stop() # Stop if we can't even get headers

    # If headers are available and mapping isn't confirmed, show mapper
    if st.session_state.csv_headers_for_mapping and not st.session_state.column_mapping_confirmed:
        st.session_state.processed_data = None # Clear any old processed data
        st.session_state.filtered_data = None  # Clear any old filtered data

        # Display column mapper UI
        # Ensure uploaded_file_bytes_for_mapper is passed and valid
        if st.session_state.uploaded_file_bytes_for_mapper:
            st.session_state.uploaded_file_bytes_for_mapper.seek(0) # Ensure pointer is at start for ColumnMapperUI
            column_mapper = ColumnMapperUI(
                uploaded_file_name=st.session_state.uploaded_file_name,
                uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper, # Pass the BytesIO object
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
                logger.info("Column mapping confirmed by user. Rerunning to process data.")
                st.rerun() # Rerun to trigger data processing with the new mapping
            else:
                # Mapper is still active or user hasn't confirmed
                st.stop() # Stop further execution until mapping is done
        else:
            display_custom_message("Error: File data for mapping is not available. Please re-upload.", "error")
            st.stop()


    # If mapping is confirmed, proceed to process data
    if st.session_state.column_mapping_confirmed and st.session_state.user_column_mapping:
        current_file_id_proc = f"{st.session_state.uploaded_file_name}-{uploaded_file.size}-{uploaded_file.type}-processing_with_mapping"
        # Process data if it's a new file, or if mapping changed, or if data hasn't been processed yet
        if st.session_state.last_processed_file_id != current_file_id_proc or st.session_state.processed_data is None:
            logger.info(f"Processing data for file: '{st.session_state.uploaded_file_name}' with confirmed mapping.")
            with st.spinner(f"Processing '{st.session_state.uploaded_file_name}'..."):
                file_obj_for_service = st.session_state.uploaded_file_bytes_for_mapper
                if file_obj_for_service: # Should always be true if mapping was confirmed
                    file_obj_for_service.seek(0) # Reset pointer before reading again
                    st.session_state.processed_data = get_and_process_data_with_profiling(
                        file_obj_for_service,
                        st.session_state.user_column_mapping,
                        st.session_state.uploaded_file_name
                    )
                else: # Should not happen if logic is correct
                    logger.error("Critical error: uploaded_file_bytes_for_mapper is None after mapping confirmation.")
                    display_custom_message("Internal error: File data disappeared. Please re-upload.", "error")
                    st.session_state.processed_data = None


            st.session_state.last_processed_file_id = current_file_id_proc
            # Reset downstream states after new data processing
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']:
                st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data # Initially, filtered is same as processed

            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
                display_custom_message(f"Successfully processed '{st.session_state.uploaded_file_name}'. DataFrame shape: {st.session_state.processed_data.shape}", "success", icon="✅")
            elif st.session_state.processed_data is not None and st.session_state.processed_data.empty:
                display_custom_message(f"Processed '{st.session_state.uploaded_file_name}', but the resulting DataFrame is empty. This might be due to data quality issues or very restrictive initial processing.", "warning")
            else: # processed_data is None
                display_custom_message(f"Failed to process '{st.session_state.uploaded_file_name}'. Please check logs, your CSV file, and column mapping.", "error")
                # Reset mapping confirmation to allow user to try again if processing fails
                st.session_state.column_mapping_confirmed = False
                st.session_state.user_column_mapping = None

# Handle case where a file was uploaded, processed, and then removed from uploader
elif st.session_state.get('uploaded_file_name') and uploaded_file is None:
    if st.session_state.processed_data is not None: # If there was data, but file is now gone
        logger.info("File uploader is now empty, but processed data exists. Resetting all data-dependent session states.")
        # More comprehensive reset
        for key_to_reset in default_session_state.keys():
            if key_to_reset == 'current_theme' or key_to_reset == 'app_theme_loaded': continue # Don't reset theme
            if key_to_reset in default_session_state: # Check if key exists in defaults before assigning
                 st.session_state[key_to_reset] = default_session_state[key_to_reset]
            elif key_to_reset in st.session_state: # If not in defaults, but in session, set to None
                 st.session_state[key_to_reset] = None
        st.rerun()


# --- Data Filtering Logic ---
@log_execution_time
def filter_data_with_profiling(df, filters, col_map):
    return data_service.filter_data(df, filters, col_map)

if st.session_state.processed_data is not None and not st.session_state.processed_data.empty and st.session_state.sidebar_filters:
    # Check if filters have changed or if filtered_data is not yet set
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        logger.info("Applying sidebar filters to processed data.")
        with st.spinner("Applying filters..."):
            st.session_state.filtered_data = filter_data_with_profiling(
                st.session_state.processed_data,
                st.session_state.sidebar_filters,
                EXPECTED_COLUMNS # Pass the app's internal expected column names
            )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        # Reset downstream states that depend on filtered_data
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None
        logger.info(f"Filtering complete. Filtered data shape: {st.session_state.filtered_data.shape if st.session_state.filtered_data is not None else 'None'}")


# --- Benchmark Data Fetching Logic ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')

    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        refetch_benchmark = False
        # Condition 1: No benchmark data yet
        if st.session_state.benchmark_daily_returns is None: refetch_benchmark = True
        # Condition 2: Ticker changed
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker: refetch_benchmark = True
        # Condition 3: Filtered data shape changed (implies date range might have changed)
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape: refetch_benchmark = True

        if refetch_benchmark:
            logger.info(f"Refetching benchmark data for ticker: {selected_ticker}")
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date') # Use internal name
            min_d_str_to_fetch, max_d_str_to_fetch = None, None

            if date_col_conceptual in st.session_state.filtered_data.columns:
                # Ensure date column is datetime
                try:
                    dates_for_bm_filtered = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                    if not dates_for_bm_filtered.empty:
                        min_d_filtered, max_d_filtered = dates_for_bm_filtered.min(), dates_for_bm_filtered.max()
                        if pd.notna(min_d_filtered) and pd.notna(max_d_filtered) and (max_d_filtered.date() - min_d_filtered.date()).days >= 0:
                            min_d_str_to_fetch = min_d_filtered.strftime('%Y-%m-%d')
                            max_d_str_to_fetch = max_d_filtered.strftime('%Y-%m-%d')
                        else:
                            logger.warning("Invalid min/max dates for benchmark fetching after processing filtered data.")
                    else:
                        logger.warning("Date column for benchmark range is empty after NaT drop.")
                except Exception as e_date_conv:
                    logger.error(f"Error converting date column for benchmark range: {e_date_conv}")
            else:
                logger.warning(f"Date column '{date_col_conceptual}' not in filtered_data for benchmark range.")


            if min_d_str_to_fetch and max_d_str_to_fetch:
                with st.spinner(f"Fetching benchmark data for {selected_ticker}..."):
                    st.session_state.benchmark_daily_returns = get_benchmark_data_static(
                        selected_ticker, min_d_str_to_fetch, max_d_str_to_fetch
                    )
                st.session_state.last_fetched_benchmark_ticker = selected_ticker
                st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape # Store shape at time of fetch
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty:
                    display_custom_message(f"Could not fetch benchmark data for {selected_ticker} or no data was returned for the period {min_d_str_to_fetch} to {max_d_str_to_fetch}.", "warning")
                else:
                    logger.info(f"Benchmark data for {selected_ticker} fetched. Shape: {st.session_state.benchmark_daily_returns.shape}")
            else:
                logger.warning(f"Cannot fetch benchmark for {selected_ticker} due to invalid/missing date range in filtered data. Min: {min_d_str_to_fetch}, Max: {max_d_str_to_fetch}")
                st.session_state.benchmark_daily_returns = None # Ensure it's None if not fetched
            st.session_state.kpi_results = None # Force KPI recalculation after benchmark change
    # If benchmark is "None" or empty, clear existing benchmark data
    elif selected_ticker == "" or selected_ticker.upper() == "NONE":
        if st.session_state.benchmark_daily_returns is not None:
            logger.info("Benchmark set to None. Clearing benchmark data and dependent KPIs.")
            st.session_state.benchmark_daily_returns = None
            st.session_state.last_fetched_benchmark_ticker = None
            st.session_state.kpi_results = None # Force KPI recalculation


# --- KPI Calculation Logic ---
@log_execution_time
def get_core_kpis_with_profiling(df, rfr, benchmark_returns, capital):
    return analysis_service_instance.get_core_kpis(df, rfr, benchmark_returns, capital)

@log_execution_time
def get_advanced_drawdown_analysis_with_profiling(equity_series):
    return analysis_service_instance.get_advanced_drawdown_analysis(equity_series)


if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    # Create a unique ID based on current data state for KPI caching/recalculation
    current_kpi_state_id_parts = [
        st.session_state.filtered_data.shape,
        st.session_state.get('risk_free_rate', RISK_FREE_RATE), # Use .get for safety
        st.session_state.get('initial_capital', 100000.0),
        st.session_state.get('selected_benchmark_ticker')
    ]
    if st.session_state.benchmark_daily_returns is not None and not st.session_state.benchmark_daily_returns.empty:
        try:
            # Ensure consistent hashing for pandas Series
            benchmark_hash_series = st.session_state.benchmark_daily_returns.sort_index()
            current_kpi_state_id_parts.append(pd.util.hash_pandas_object(benchmark_hash_series, index=True).sum())
        except Exception as e_hash: # Catch any error during hashing
            logger.warning(f"Hashing benchmark_daily_returns failed: {e_hash}. Using its shape as a fallback for KPI state.")
            current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else:
        current_kpi_state_id_parts.append(None) # Placeholder if no benchmark data

    current_kpi_state_id = tuple(current_kpi_state_id_parts)

    # Recalculate KPIs if data state has changed
    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info("Recalculating KPIs, Confidence Intervals, and Max Drawdown Details as data state changed.")
        with st.spinner("Calculating key performance metrics..."):
            kpi_res = get_core_kpis_with_profiling(
                st.session_state.filtered_data,
                st.session_state.get('risk_free_rate', RISK_FREE_RATE),
                st.session_state.benchmark_daily_returns, # Pass directly
                st.session_state.get('initial_capital', 100000.0)
            )

            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res
                st.session_state.last_kpi_calc_state_id = current_kpi_state_id # Update state ID

                # Advanced Drawdown after core KPIs
                date_col = EXPECTED_COLUMNS.get('date')
                cum_pnl_col = 'cumulative_pnl' # This is an engineered column
                equity_series_for_dd = pd.Series(dtype=float)

                if date_col and cum_pnl_col and \
                   date_col in st.session_state.filtered_data.columns and \
                   cum_pnl_col in st.session_state.filtered_data.columns:
                    
                    temp_df_for_equity = st.session_state.filtered_data[[date_col, cum_pnl_col]].copy()
                    # Ensure date_col is datetime
                    if not pd.api.types.is_datetime64_any_dtype(temp_df_for_equity[date_col]):
                        temp_df_for_equity[date_col] = pd.to_datetime(temp_df_for_equity[date_col], errors='coerce')
                    
                    temp_df_for_equity.dropna(subset=[date_col], inplace=True) # Drop rows where date conversion failed
                    
                    if not temp_df_for_equity.empty:
                        equity_series_for_dd = temp_df_for_equity.set_index(date_col)[cum_pnl_col].sort_index().dropna()
                
                if not equity_series_for_dd.empty and len(equity_series_for_dd) >= 5: # Min points for drawdown
                    adv_dd_results = get_advanced_drawdown_analysis_with_profiling(equity_series_for_dd)
                    st.session_state.max_drawdown_period_details = adv_dd_results.get('max_drawdown_details') if adv_dd_results and 'error' not in adv_dd_results else None
                    if adv_dd_results and 'error' in adv_dd_results:
                        logger.warning(f"Advanced drawdown analysis service returned an error: {adv_dd_results['error']}")
                else:
                    st.session_state.max_drawdown_period_details = None
                    if equity_series_for_dd.empty: logger.info("Equity series for drawdown is empty.")
                    else: logger.info(f"Equity series too short for drawdown: {len(equity_series_for_dd)} points.")


                # Confidence Intervals
                pnl_col_for_ci = EXPECTED_COLUMNS.get('pnl')
                if pnl_col_for_ci and pnl_col_for_ci in st.session_state.filtered_data.columns:
                    pnl_series_for_ci = st.session_state.filtered_data[pnl_col_for_ci].dropna()
                    if len(pnl_series_for_ci) >= 10: # Min trades for meaningful CI
                        ci_res = analysis_service_instance.get_bootstrapped_kpi_cis(
                            st.session_state.filtered_data, ['avg_trade_pnl', 'win_rate', 'sharpe_ratio']
                        )
                        st.session_state.kpi_confidence_intervals = ci_res if ci_res and 'error' not in ci_res else {}
                    else:
                        st.session_state.kpi_confidence_intervals = {}
                        logger.info(f"Not enough PnL data points ({len(pnl_series_for_ci)}) for CI calculation.")
                else:
                    st.session_state.kpi_confidence_intervals = {}
                    logger.warning(f"PnL column for CI ('{pnl_col_for_ci}') not found.")
            else: # Error in KPI calculation
                error_msg = kpi_res.get('error', 'Unknown error') if kpi_res else 'KPI calculation service failed'
                display_custom_message(f"KPI calculation error: {error_msg}", "error")
                st.session_state.kpi_results = None # Ensure it's None on error
                st.session_state.kpi_confidence_intervals = {}
                st.session_state.max_drawdown_period_details = None

# If filtered data becomes empty (e.g., due to filters), clear KPIs
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty: # Only show if there was data to begin with
        display_custom_message("No data matches the current filter criteria. Adjust filters or upload a new file.", "info")
    st.session_state.kpi_results = None
    st.session_state.kpi_confidence_intervals = {}
    st.session_state.max_drawdown_period_details = None


# --- Welcome Page Layout Function (Main content if no data or no file uploaded) ---
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
        st.markdown("<div class='feature-item'><h4>📄 Upload Data</h4><p>Begin by uploading your trade journal (CSV) via the sidebar. Our intelligent mapping assistant will guide you through aligning your columns.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-item'><h4>📊 Analyze Performance</h4><p>Dive deep into comprehensive performance metrics, equity curves, and statistical breakdowns once your data is loaded and processed.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-item'><h4>💡 Discover Insights</h4><p>Leverage advanced tools like categorical analysis, strategy comparisons, and AI-driven suggestions available in the dashboard pages.</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py" # Relative path for st.switch_page
    # Check existence based on standard Streamlit multipage structure
    # This check might not be perfectly reliable if pages are structured differently or not yet generated
    # For simplicity, we assume if the file exists in the deployed structure, it's a valid page.
    # A more robust check would involve inspecting st.navigation if available or known page list.
    if os.path.exists(user_guide_page_path): # Basic check
        if st.button("📘 Read the User Guide", key="welcome_user_guide_button_v2", help="Navigate to the User Guide page"):
            st.switch_page(user_guide_page_path) # Use the relative path
    else:
        st.markdown("<p style='text-align: center; font-style: italic;'>User guide page link will appear here once available.</p>", unsafe_allow_html=True)
        logger.warning(f"User Guide page not found at expected relative path: {user_guide_page_path}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Page Navigation and Display Logic ---
# This determines what content is shown on the main app.py (the "home" page)
# If no file is uploaded AND no data has ever been processed, show the welcome/main layout.
if not uploaded_file and st.session_state.processed_data is None:
    main_page_layout()
    logger.info("Displaying main_page_layout as no file uploaded and no processed_data in session.")
    st.stop() # Stop execution for this page if welcome page is shown

# If a file is uploaded, but data hasn't been processed and mapping isn't confirmed,
# the column mapper will be shown (handled by logic above). If mapping fails to get headers, stop.
elif uploaded_file and st.session_state.processed_data is None and not st.session_state.column_mapping_confirmed:
    if st.session_state.csv_headers_for_mapping is None and uploaded_file: # This implies header reading failed
        # Error message already displayed by ColumnMapperUI or header reading logic
        logger.info("Stopping app.py display because CSV headers could not be read for mapping.")
        st.stop()
    # If headers are there, but mapping not confirmed, mapper is active, so app.py doesn't need to show more.
    elif st.session_state.csv_headers_for_mapping:
        logger.info("Column mapper is active. app.py main content display deferred.")
        st.stop()


# If data is processed, but filtered data is empty (and it's not due to KPIs being in error state),
# show an info message. This allows sidebar and other UI to still render.
elif st.session_state.processed_data is not None and \
     (st.session_state.filtered_data is None or st.session_state.filtered_data.empty) and \
     not (st.session_state.kpi_results and 'error' in st.session_state.kpi_results):
    # This condition is tricky. We want to show this message if filters legitimately result in no data,
    # but not if the app is still in a loading/processing state for KPIs.
    # The `kpi_results and 'error' not in kpi_results` part tries to avoid showing this
    # if the main issue is a KPI calculation error that might be resolved.
    if st.session_state.sidebar_filters and uploaded_file: # Check if filters were applied
        # Message already displayed by KPI calculation logic if filtered_data is empty
        pass # display_custom_message("No data matches the current filter criteria. Please adjust your filters in the sidebar.", "info")

# Fallback if no other condition stops:
# This means data is likely processed, filtered (even if to itself), and KPIs are calculated or being calculated.
# The individual pages will handle their own display.
# If app.py is the current "page" (e.g., before navigating to a sub-page, or if it's the only page),
# you might want a default display here. For a multi-page app, this might be minimal if Overview is the default page.

# If we reach here, it means:
# 1. A file was uploaded OR data was previously processed.
# 2. If a file was uploaded, mapping is confirmed and data is processed.
# 3. Filtered data exists (even if it's the same as processed_data).
# In a multi-page app setup where app.py is just the entry point and pages/1_📈_Overview.py
# is the actual landing page content, app.py might not need to display much itself
# beyond setting up the sidebar and global state.
# The welcome layout (main_page_layout) is shown if no data is loaded AT ALL.
# Once data is loaded, users navigate to other pages.

# If you want app.py to show something by default when data IS loaded,
# (e.g., if it's acting as a home/summary page before user navigates), add that content here.
# For now, we assume that if data is loaded, the user will navigate to a specific page
# from the sidebar, and those pages will render their content.
# The "Overview" page is typically the first one.

# Add scroll buttons to all pages if this script is the main one being run
try:
    scroll_buttons_component = ScrollButtons()
    scroll_buttons_component.render()
except Exception as e_scroll:
    logger.error(f"Error rendering scroll buttons: {e_scroll}", exc_info=True)


logger.info(f"App '{APP_TITLE}' run cycle finished for app.py main display logic.")
