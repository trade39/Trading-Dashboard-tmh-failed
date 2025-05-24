# components/sidebar_manager.py
"""
This component encapsulates the logic for creating and managing
sidebar filters and controls. Now uses user preferences for defaults.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import datetime

try:
    from config import EXPECTED_COLUMNS, RISK_FREE_RATE, APP_TITLE, AVAILABLE_BENCHMARKS, DEFAULT_BENCHMARK_TICKER
except ImportError:
    # ... (fallback definitions as before) ...
    print("Warning (sidebar_manager.py): Could not import from root config. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"; EXPECTED_COLUMNS = {"date": "date", "symbol": "symbol", "strategy": "strategy"}
    RISK_FREE_RATE = 0.02; AVAILABLE_BENCHMARKS = {"S&P 500 (SPY)": "SPY", "None": ""}; DEFAULT_BENCHMARK_TICKER = "SPY"

import logging
logger = logging.getLogger(APP_TITLE)

class SidebarManager:
    def __init__(self, processed_data: Optional[pd.DataFrame]):
        self.processed_data = processed_data
        self.filter_values: Dict[str, Any] = {}
        logger.debug("SidebarManager initialized.")

    def _get_date_range_objects(self) -> Optional[Tuple[datetime.date, datetime.date]]:
        # ... (implementation as before) ...
        date_col_name = EXPECTED_COLUMNS.get('date')
        if self.processed_data is not None and date_col_name and date_col_name in self.processed_data.columns and not self.processed_data[date_col_name].empty:
            try:
                df_dates_dt = pd.to_datetime(self.processed_data[date_col_name], errors='coerce').dropna()
                if not df_dates_dt.empty: return df_dates_dt.min().date(), df_dates_dt.max().date()
            except Exception as e: logger.error(f"Error processing date column ('{date_col_name}') for range: {e}", exc_info=True)
        return None


    def render_sidebar_controls(self) -> Dict[str, Any]:
        with st.sidebar:
            # --- User Preferences for Defaults ---
            user_prefs = st.session_state.get('user_preferences', {})
            
            # --- Risk-Free Rate ---
            default_rfr_from_prefs = user_prefs.get('default_risk_free_rate', RISK_FREE_RATE)
            # Ensure session state for rfr input matches user pref or global default if not set yet
            if 'sidebar_rfr_input_v3' not in st.session_state: # Check if widget state exists
                 st.session_state.sidebar_rfr_input_v3 = default_rfr_from_prefs * 100

            rfr_percentage = st.number_input(
                "Annual Risk-Free Rate (%)", min_value=0.0, max_value=100.0,
                value=st.session_state.sidebar_rfr_input_v3, # Use widget state for value
                step=0.01, format="%.2f", key="sidebar_rfr_input_v3",
                help="Enter annualized risk-free rate (e.g., 2 for 2%)."
            )
            self.filter_values['risk_free_rate'] = rfr_percentage / 100.0
            # No need to update st.session_state.risk_free_rate here, app.py will do it based on returned filter_values

            st.markdown("---")
            st.subheader("Data Filters")

            # --- Date Range Filter (as before) ---
            date_range_objs = self._get_date_range_objects()
            selected_date_val_tuple = None
            if date_range_objs:
                min_date_data, max_date_data = date_range_objs
                if min_date_data <= max_date_data:
                    session_default_tuple = st.session_state.get('sidebar_date_range_filter_tuple_val')
                    default_start_val, default_end_val = min_date_data, max_date_data
                    if session_default_tuple and isinstance(session_default_tuple, tuple) and len(session_default_tuple) == 2:
                        s_start, s_end = (d.date() if isinstance(d, datetime.datetime) else d for d in session_default_tuple)
                        if isinstance(s_start, datetime.date) and isinstance(s_end, datetime.date):
                             current_default_start, current_default_end = max(min_date_data, s_start), min(max_date_data, s_end)
                             if current_default_start <= current_default_end: default_start_val, default_end_val = current_default_start, current_default_end
                    if min_date_data < max_date_data :
                        selected_date_val_tuple = st.date_input("Select Date Range", value=(default_start_val, default_end_val), min_value=min_date_data, max_value=max_date_data, key="sidebar_date_range_filter_tuple_input_v3")
                    else: selected_date_val_tuple = (min_date_data, max_date_data); st.info(f"Data for single date: {min_date_data.strftime('%Y-%m-%d')}")
                    st.session_state.sidebar_date_range_filter_tuple_val = selected_date_val_tuple
            else: st.info("Upload data with a valid 'date' column for date filtering.")
            self.filter_values['selected_date_range'] = selected_date_val_tuple

            # --- Symbol Filter (as before) ---
            actual_symbol_col = EXPECTED_COLUMNS.get('symbol'); selected_symbol_val = "All"
            if self.processed_data is not None and actual_symbol_col and actual_symbol_col in self.processed_data.columns:
                try:
                    unique_symbols = ["All"] + sorted(self.processed_data[actual_symbol_col].astype(str).dropna().unique().tolist())
                    if unique_symbols: selected_symbol_val = st.selectbox("Filter by Symbol", unique_symbols, index=0, key="sidebar_symbol_filter_input_v3")
                except Exception as e: logger.error(f"Error populating symbol filter: {e}", exc_info=True)
            self.filter_values['selected_symbol'] = selected_symbol_val

            # --- Strategy Filter (as before) ---
            actual_strategy_col = EXPECTED_COLUMNS.get('strategy'); selected_strategy_val = "All"
            if self.processed_data is not None and actual_strategy_col and actual_strategy_col in self.processed_data.columns:
                try:
                    unique_strategies = ["All"] + sorted(self.processed_data[actual_strategy_col].astype(str).dropna().unique().tolist())
                    if unique_strategies: selected_strategy_val = st.selectbox("Filter by Strategy", unique_strategies, index=0, key="sidebar_strategy_filter_input_v3")
                except Exception as e: logger.error(f"Error populating strategy filter: {e}", exc_info=True)
            self.filter_values['selected_strategy'] = selected_strategy_val

            st.markdown("---")
            st.subheader("Benchmark Selection")
            
            # --- Benchmark Selection Dropdown (uses user preference for default) ---
            benchmark_display_names = list(AVAILABLE_BENCHMARKS.keys())
            default_benchmark_ticker_from_prefs = user_prefs.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER)
            default_benchmark_display_name_from_prefs = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == default_benchmark_ticker_from_prefs), "None")

            # Ensure session state for benchmark selectbox matches user pref or global default
            if 'sidebar_benchmark_select_v1' not in st.session_state: # Check if widget state exists
                st.session_state.sidebar_benchmark_select_v1 = default_benchmark_display_name_from_prefs
            
            # Ensure the current value in session state is valid, otherwise reset to pref/default
            if st.session_state.sidebar_benchmark_select_v1 not in benchmark_display_names:
                 st.session_state.sidebar_benchmark_select_v1 = default_benchmark_display_name_from_prefs


            selected_benchmark_name = st.selectbox(
                "Select Benchmark",
                options=benchmark_display_names,
                index=benchmark_display_names.index(st.session_state.sidebar_benchmark_select_v1), # Use widget state for index
                key="sidebar_benchmark_select_v1", # Key remains same
                help="Select a market index. 'None' disables benchmark comparison."
            )
            selected_benchmark_ticker = AVAILABLE_BENCHMARKS.get(selected_benchmark_name, "")
            self.filter_values['selected_benchmark_ticker'] = selected_benchmark_ticker
            
            # --- Initial Capital Input (as before, not typically a persistent user preference) ---
            st.markdown("---")
            st.subheader("Strategy Settings")
            initial_capital_input = st.number_input(
                "Initial Capital (for % Returns & Benchmarking)",
                min_value=0.0, value=st.session_state.get('initial_capital', 100000.0),
                step=1000.0, format="%.2f", key="sidebar_initial_capital_v1",
                help="Enter initial capital for the strategy."
            )
            self.filter_values['initial_capital'] = initial_capital_input

            logger.debug(f"Sidebar controls rendered. Filter values: {self.filter_values}")
            return self.filter_values
