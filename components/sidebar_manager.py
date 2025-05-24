# components/sidebar_manager.py
"""
This component encapsulates the logic for creating and managing
sidebar filters and controls. Now uses user preferences for defaults
and a refined approach for the RFR input widget to avoid session state warnings.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import datetime

try:
    from config import EXPECTED_COLUMNS, RISK_FREE_RATE, APP_TITLE, AVAILABLE_BENCHMARKS, DEFAULT_BENCHMARK_TICKER
except ImportError:
    # Fallback definitions
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
        date_col_name = EXPECTED_COLUMNS.get('date')
        if self.processed_data is not None and date_col_name and date_col_name in self.processed_data.columns and not self.processed_data[date_col_name].empty:
            try:
                # Ensure the date column is properly converted to datetime objects
                df_dates_dt = pd.to_datetime(self.processed_data[date_col_name], errors='coerce').dropna()
                if not df_dates_dt.empty:
                    min_date = df_dates_dt.min()
                    max_date = df_dates_dt.max()
                    # Ensure they are date objects, not datetime
                    return min_date.date() if hasattr(min_date, 'date') else min_date, \
                           max_date.date() if hasattr(max_date, 'date') else max_date
            except Exception as e:
                logger.error(f"Error processing date column ('{date_col_name}') for range: {e}", exc_info=True)
        return None


    def render_sidebar_controls(self) -> Dict[str, Any]:
        with st.sidebar:
            user_prefs = st.session_state.get('user_preferences', {})
            
            # --- Risk-Free Rate ---
            # The authoritative value for risk_free_rate (decimal) is stored in st.session_state.risk_free_rate
            authoritative_rfr_decimal = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
            current_rfr_percentage_for_widget = authoritative_rfr_decimal * 100.0

            rfr_percentage_from_widget = st.number_input(
                "Annual Risk-Free Rate (%)",
                min_value=0.0, max_value=100.0,
                value=current_rfr_percentage_for_widget, # Set value from the authoritative source
                step=0.01, format="%.2f",
                key="sidebar_rfr_percentage_widget_v2", # Distinct key for the widget
                help="Enter annualized risk-free rate (e.g., 2 for 2%)."
            )
            # This widget's state is now 'sidebar_rfr_percentage_widget_v2'.
            # The app.py loop will read this, convert to decimal, and update st.session_state.risk_free_rate.
            self.filter_values['risk_free_rate'] = rfr_percentage_from_widget / 100.0
            
            st.markdown("---")
            st.subheader("Data Filters")

            # --- Date Range Filter ---
            date_range_objs = self._get_date_range_objects()
            selected_date_val_tuple = None
            if date_range_objs:
                min_date_data, max_date_data = date_range_objs
                if min_date_data and max_date_data and min_date_data <= max_date_data:
                    # Use st.session_state.global_date_filter_range for the default if it exists and is valid
                    current_default_start, current_default_end = min_date_data, max_date_data
                    if st.session_state.get('global_date_filter_range') and \
                       isinstance(st.session_state.global_date_filter_range, tuple) and \
                       len(st.session_state.global_date_filter_range) == 2:
                        
                        s_start_raw, s_end_raw = st.session_state.global_date_filter_range
                        s_start = s_start_raw.date() if isinstance(s_start_raw, datetime.datetime) else s_start_raw
                        s_end = s_end_raw.date() if isinstance(s_end_raw, datetime.datetime) else s_end_raw

                        if isinstance(s_start, datetime.date) and isinstance(s_end, datetime.date):
                            # Ensure the session state default is within the data's bounds
                            s_start_clamped = max(min_date_data, s_start)
                            s_end_clamped = min(max_date_data, s_end)
                            if s_start_clamped <= s_end_clamped:
                                current_default_start, current_default_end = s_start_clamped, s_end_clamped
                    
                    if min_date_data < max_date_data :
                        selected_date_val_tuple = st.date_input(
                            "Select Date Range", 
                            value=(current_default_start, current_default_end), 
                            min_value=min_date_data, 
                            max_value=max_date_data, 
                            key="sidebar_date_range_filter_tuple_input_v3" # Keep key if no issues
                        )
                    else: # Single day of data
                        selected_date_val_tuple = (min_date_data, max_date_data)
                        st.info(f"Data available for a single date: {min_date_data.strftime('%Y-%m-%d')}")
                else:
                     st.info("Date range for filtering could not be determined from data.")
            else: 
                st.info("Upload data with a valid 'date' column for date filtering.")
            self.filter_values['selected_date_range'] = selected_date_val_tuple

            # --- Symbol Filter ---
            actual_symbol_col = EXPECTED_COLUMNS.get('symbol'); selected_symbol_val = "All"
            if self.processed_data is not None and actual_symbol_col and actual_symbol_col in self.processed_data.columns:
                try:
                    unique_symbols = ["All"] + sorted(self.processed_data[actual_symbol_col].astype(str).dropna().unique().tolist())
                    # Default to current session state value if valid, else "All"
                    current_symbol_filter = st.session_state.get('global_symbol_filter', "All")
                    symbol_idx = unique_symbols.index(current_symbol_filter) if current_symbol_filter in unique_symbols else 0
                    if unique_symbols: selected_symbol_val = st.selectbox("Filter by Symbol", unique_symbols, index=symbol_idx, key="sidebar_symbol_filter_input_v3")
                except Exception as e: logger.error(f"Error populating symbol filter: {e}", exc_info=True)
            self.filter_values['selected_symbol'] = selected_symbol_val

            # --- Strategy Filter ---
            actual_strategy_col = EXPECTED_COLUMNS.get('strategy'); selected_strategy_val = "All"
            if self.processed_data is not None and actual_strategy_col and actual_strategy_col in self.processed_data.columns:
                try:
                    unique_strategies = ["All"] + sorted(self.processed_data[actual_strategy_col].astype(str).dropna().unique().tolist())
                    current_strategy_filter = st.session_state.get('global_strategy_filter', "All")
                    strategy_idx = unique_strategies.index(current_strategy_filter) if current_strategy_filter in unique_strategies else 0
                    if unique_strategies: selected_strategy_val = st.selectbox("Filter by Strategy", unique_strategies, index=strategy_idx, key="sidebar_strategy_filter_input_v3")
                except Exception as e: logger.error(f"Error populating strategy filter: {e}", exc_info=True)
            self.filter_values['selected_strategy'] = selected_strategy_val

            st.markdown("---")
            st.subheader("Benchmark Selection")
            
            benchmark_display_names = list(AVAILABLE_BENCHMARKS.keys())
            # Authoritative benchmark ticker is in st.session_state.selected_benchmark_ticker
            authoritative_benchmark_ticker = st.session_state.get('selected_benchmark_ticker', DEFAULT_BENCHMARK_TICKER)
            current_benchmark_display_name = next((name for name, ticker_val in AVAILABLE_BENCHMARKS.items() if ticker_val == authoritative_benchmark_ticker), "None")
            
            benchmark_select_idx = 0
            if current_benchmark_display_name in benchmark_display_names:
                benchmark_select_idx = benchmark_display_names.index(current_benchmark_display_name)

            selected_benchmark_name = st.selectbox(
                "Select Benchmark",
                options=benchmark_display_names,
                index=benchmark_select_idx, 
                key="sidebar_benchmark_select_v2_widget", # Distinct key for widget
                help="Select a market index. 'None' disables benchmark comparison."
            )
            selected_benchmark_ticker = AVAILABLE_BENCHMARKS.get(selected_benchmark_name, "")
            self.filter_values['selected_benchmark_ticker'] = selected_benchmark_ticker
            
            st.markdown("---")
            st.subheader("Strategy Settings")
            # Authoritative initial capital is in st.session_state.initial_capital
            authoritative_initial_capital = st.session_state.get('initial_capital', 100000.0)
            initial_capital_input = st.number_input(
                "Initial Capital (for % Returns & Benchmarking)",
                min_value=0.0, 
                value=authoritative_initial_capital, # Source from authoritative state
                step=1000.0, format="%.2f", 
                key="sidebar_initial_capital_widget_v2", # Distinct key for widget
                help="Enter initial capital for the strategy."
            )
            self.filter_values['initial_capital'] = initial_capital_input

            logger.debug(f"Sidebar controls rendered. Filter values: {self.filter_values}")
            return self.filter_values
