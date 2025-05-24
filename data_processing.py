# data_processing.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
import re
import chardet # For encoding detection
import csv # For delimiter sniffing
from io import StringIO, BytesIO # Added BytesIO

try:
    from config import APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_TYPES
except ImportError:
    APP_TITLE = "TradingDashboard_Default_DP"
    CONCEPTUAL_COLUMNS = {"date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)", "entry_time_str": "Entry Time"}
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    CONCEPTUAL_COLUMN_TYPES = {"date": "datetime", "pnl": "numeric", "entry_time_str": "text"}
    print("Warning (data_processing.py): Could not import from config. Using fallback values.")

logger = logging.getLogger(APP_TITLE)

def _calculate_drawdown_series_for_df(cumulative_pnl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    high_water_mark = cumulative_pnl.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl
    hwm_for_pct = high_water_mark.replace(0, np.nan) 
    drawdown_pct_series = (drawdown_abs_series / hwm_for_pct).fillna(0) * 100
    
    mask_hwm_zero_loss = (high_water_mark == 0) & (drawdown_abs_series > 0)
    drawdown_pct_series[mask_hwm_zero_loss] = 100.0

    return drawdown_abs_series, drawdown_pct_series

def clean_text_column(text_series: pd.Series) -> pd.Series:
    if not isinstance(text_series, pd.Series):
        return pd.Series(text_series, dtype=str)
    processed_series = text_series.astype(str).fillna('').str.strip()
    url_pattern = r"\(?https?://[^\s\)\"]+\)?|www\.[^\s\)\"]+"
    notion_link_pattern = r"\(https://www\.notion\.so/[^)]+\)"
    empty_parens_pattern = r"^\(''\)$"

    def clean_element(text: str) -> Any:
        if pd.isna(text) or text.lower() == 'nan': return pd.NA
        cleaned_text = text
        cleaned_text = re.sub(notion_link_pattern, '', cleaned_text)
        cleaned_text = re.sub(url_pattern, '', cleaned_text)
        cleaned_text = re.sub(empty_parens_pattern, '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text if cleaned_text else pd.NA
    return processed_series.apply(clean_element)

@st.cache_data(ttl=3600, show_spinner="Loading and processing trade data...")
def load_and_process_data(
    uploaded_file_obj: Any, # Can be BytesIO or UploadedFile
    user_column_mapping: Optional[Dict[str, str]] = None,
    on_bad_lines_config: str = 'warn',
    original_file_name: Optional[str] = None, # Added for logging
    detected_encoding_from_app: Optional[str] = None # NEW: Accept detected encoding
) -> Optional[pd.DataFrame]:
    file_name_for_log = original_file_name or getattr(uploaded_file_obj, 'name', "In-memory data")
    logger.info(f"load_and_process_data called for '{file_name_for_log}'. Passed encoding: {detected_encoding_from_app}")

    if uploaded_file_obj is None:
        logger.info("No file object provided to load_and_process_data.")
        return None

    # Ensure we have a BytesIO object to work with, as pd.read_csv expects a file-like object
    if not isinstance(uploaded_file_obj, BytesIO):
        if hasattr(uploaded_file_obj, 'getvalue'):
            try:
                file_bytes = uploaded_file_obj.getvalue()
                file_like_obj = BytesIO(file_bytes)
                file_like_obj.seek(0)
            except Exception as e:
                logger.error(f"Could not convert uploaded file object to BytesIO for '{file_name_for_log}': {e}")
                st.error(f"Internal error preparing file for processing: {e}")
                return None
        else:
            logger.error(f"Uploaded file object for '{file_name_for_log}' is not BytesIO and has no getvalue method.")
            st.error("Invalid file object type received for processing.")
            return None
    else:
        file_like_obj = uploaded_file_obj
        file_like_obj.seek(0)


    df = None
    final_encoding_used = None
    detected_delimiter = None

    # Attempt 1: Use encoding passed from app.py (if available)
    if detected_encoding_from_app:
        try:
            logger.info(f"Attempting to read CSV '{file_name_for_log}' with passed encoding: {detected_encoding_from_app}")
            file_like_obj.seek(0)
            df = pd.read_csv(file_like_obj, encoding=detected_encoding_from_app, on_bad_lines=on_bad_lines_config, skipinitialspace=True)
            final_encoding_used = detected_encoding_from_app
            logger.info(f"Successfully loaded CSV '{file_name_for_log}' with passed encoding '{final_encoding_used}'. Shape: {df.shape}.")
        except (UnicodeDecodeError, pd.errors.ParserError) as e_passed_enc:
            logger.warning(f"Passed encoding '{detected_encoding_from_app}' failed for '{file_name_for_log}': {e_passed_enc}. Will try other methods.")
            df = None # Ensure df is None to trigger further attempts
        except Exception as e_gen_passed:
            logger.error(f"General error reading CSV '{file_name_for_log}' with passed encoding '{detected_encoding_from_app}': {e_gen_passed}", exc_info=True)
            df = None


    # Attempt 2: Try UTF-8 if passed encoding failed or wasn't provided
    if df is None:
        try:
            logger.info(f"Attempting to read CSV '{file_name_for_log}' with UTF-8.")
            file_like_obj.seek(0)
            df = pd.read_csv(file_like_obj, encoding='utf-8', on_bad_lines=on_bad_lines_config, skipinitialspace=True)
            final_encoding_used = 'utf-8'
            logger.info(f"Successfully loaded CSV '{file_name_for_log}' with UTF-8. Shape: {df.shape}.")
        except (UnicodeDecodeError, pd.errors.ParserError) as e_utf8:
            logger.warning(f"UTF-8 decoding failed for '{file_name_for_log}': {e_utf8}. Will try chardet.")
            df = None
        except Exception as e_gen_utf8:
            logger.error(f"General error reading CSV '{file_name_for_log}' with UTF-8: {e_gen_utf8}", exc_info=True)
            df = None

    # Attempt 3: Use chardet if previous attempts failed
    if df is None:
        logger.warning(f"UTF-8 (and/or passed encoding) failed for '{file_name_for_log}'. Attempting to detect encoding with chardet.")
        file_like_obj.seek(0)
        raw_data_sample = file_like_obj.read(100*1024) # Read up to 100KB for chardet
        file_like_obj.seek(0)

        if raw_data_sample:
            chardet_result = chardet.detect(raw_data_sample)
            detected_encoding_chardet = chardet_result.get('encoding')
            confidence = chardet_result.get('confidence', 0)
            logger.info(f"Chardet detected encoding: {detected_encoding_chardet} with confidence: {confidence:.2f} for '{file_name_for_log}'.")

            if detected_encoding_chardet and confidence > 0.5: # Using a confidence threshold
                try:
                    logger.info(f"Retrying CSV read for '{file_name_for_log}' with chardet encoding: {detected_encoding_chardet}")
                    file_like_obj.seek(0)
                    df = pd.read_csv(file_like_obj, encoding=detected_encoding_chardet, on_bad_lines=on_bad_lines_config, skipinitialspace=True)
                    final_encoding_used = detected_encoding_chardet
                    logger.info(f"Successfully loaded CSV '{file_name_for_log}' with chardet encoding '{final_encoding_used}'. Shape: {df.shape}.")
                except Exception as e_enc:
                    logger.error(f"Error reading CSV '{file_name_for_log}' with chardet encoding '{detected_encoding_chardet}': {e_enc}", exc_info=True)
                    df = None # Ensure df is None if this attempt fails
            else:
                logger.warning(f"Chardet could not confidently detect encoding for '{file_name_for_log}' (confidence: {confidence:.2f}). Trying latin1 as last resort.")
                df = None # Proceed to latin1 if chardet is not confident
        else:
            logger.error(f"Could not read sample data for chardet encoding detection from '{file_name_for_log}'.")
            df = None

    # Attempt 4: Try latin1 as a final fallback if chardet also failed or was not confident
    if df is None:
        try:
            logger.info(f"Attempting to read CSV '{file_name_for_log}' with latin1 as a fallback.")
            file_like_obj.seek(0)
            df = pd.read_csv(file_like_obj, encoding='latin1', on_bad_lines=on_bad_lines_config, skipinitialspace=True)
            final_encoding_used = 'latin1'
            logger.info(f"Successfully loaded CSV '{file_name_for_log}' with latin1. Shape: {df.shape}.")
        except Exception as e_latin1:
            logger.error(f"All encoding attempts failed for '{file_name_for_log}', including latin1: {e_latin1}", exc_info=True)
            st.error(f"Error reading CSV '{file_name_for_log}': Could not determine file encoding. Please ensure it's a valid CSV (e.g., UTF-8, latin1).")
            return None
            
    # Delimiter sniffing if a ParserError occurred and df is still None (or if initial read was problematic)
    # This part is tricky to integrate perfectly if an encoding error also happened.
    # Assuming if we reached here with df=None, it was an encoding issue primarily.
    # If a ParserError specifically was the reason for df being None, one might try delimiter sniffing.
    # For now, the primary focus of the fix is encoding. Delimiter sniffing can be added if ParserError is frequent.

    if df is None: # Should not happen if latin1 fallback is successful, but as a safeguard
        logger.error(f"DataFrame is None after all parsing attempts for '{file_name_for_log}'.")
        st.error(f"Could not load the CSV file '{file_name_for_log}' after multiple parsing strategies.")
        return None
    
    logger.info(f"Final encoding used for '{file_name_for_log}': {final_encoding_used}")
    
    if on_bad_lines_config == 'warn' and not df.empty:
        logger.info(f"CSV '{file_name_for_log}' loaded with on_bad_lines='{on_bad_lines_config}'. Check console/logs for pandas warnings if any bad lines were encountered.")

    original_csv_headers = df.columns.tolist()
    conceptual_to_original_csv_map = {
        conceptual_key: csv_header
        for conceptual_key, csv_header in (user_column_mapping or {}).items()
    }

    if user_column_mapping:
        rename_map = {csv_col: conceptual_key
                      for conceptual_key, csv_col in user_column_mapping.items()
                      if csv_col in df.columns}
        df.rename(columns=rename_map, inplace=True)
        logger.info(f"Applied user column mapping for '{file_name_for_log}'. Renamed columns based on: {rename_map}")
        logger.info(f"DataFrame columns after user mapping for '{file_name_for_log}': {df.columns.tolist()}")
    else:
        logger.warning(f"No user column mapping provided for '{file_name_for_log}'. Applying default header cleaning.")
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in original_csv_headers]
        logger.info(f"Cleaned DataFrame headers (default) for '{file_name_for_log}': {df.columns.tolist()}")

    missing_critical_mapped_cols = [
        key for key in CRITICAL_CONCEPTUAL_COLUMNS if key not in df.columns
    ]
    if missing_critical_mapped_cols:
        error_message_parts = [
            f"Critical data fields are missing after mapping for '{file_name_for_log}': {', '.join(missing_critical_mapped_cols)}."
        ]
        # ... (rest of error message construction remains the same) ...
        final_error_msg = " ".join(error_message_parts)
        logger.error(final_error_msg)
        st.error(final_error_msg)
        return None

    # --- Type Conversion and Feature Engineering ---
    # (The rest of the function from "Combine Date and Entry Time" onwards remains the same)
    # --- MODIFICATION START: Combine Date and Entry Time ---
    date_conceptual_key = 'date' 
    time_conceptual_key = 'entry_time_str'

    if date_conceptual_key in df.columns and time_conceptual_key in df.columns:
        logger.info(f"Found both '{date_conceptual_key}' and '{time_conceptual_key}' columns in '{file_name_for_log}'. Attempting to combine them.")
        date_series_str = df[date_conceptual_key].astype(str)
        time_series_str = df[time_conceptual_key].astype(str)
        combined_datetime_str_series = date_series_str.str.strip() + " " + time_series_str.str.strip()
        df[date_conceptual_key] = combined_datetime_str_series
        logger.info(f"Combined '{date_conceptual_key}' and '{time_conceptual_key}' into the '{date_conceptual_key}' column for '{file_name_for_log}' for datetime parsing.")
    elif date_conceptual_key in df.columns:
        logger.info(f"Only '{date_conceptual_key}' column found in '{file_name_for_log}'. Proceeding with it for datetime parsing.")
    # --- MODIFICATION END ---

    try:
        for conceptual_key, _ in CONCEPTUAL_COLUMNS.items():
            actual_col_name_in_df = conceptual_key 
            original_csv_col_name = conceptual_to_original_csv_map.get(conceptual_key, "N/A (Not Mapped or Direct)")

            if actual_col_name_in_df not in df.columns:
                if conceptual_key == 'risk_pct': 
                    df['risk_numeric_internal'] = 0.0 
                elif conceptual_key == 'duration_minutes':
                    df['duration_minutes_numeric'] = pd.NA 
                continue

            logger.debug(f"Processing mapped conceptual column '{conceptual_key}' (from CSV '{original_csv_col_name}') for '{file_name_for_log}'")
            series = df[actual_col_name_in_df]
            original_series_sample = series.dropna().head().tolist()

            try:
                expected_type = CONCEPTUAL_COLUMN_TYPES.get(conceptual_key)
                
                if expected_type == 'datetime': 
                    df[actual_col_name_in_df] = pd.to_datetime(series, errors='coerce')
                    if df[actual_col_name_in_df].isnull().sum() > 0:
                        logger.warning(f"{df[actual_col_name_in_df].isnull().sum()} invalid date formats in mapped '{conceptual_key}' (CSV: '{original_csv_col_name}') for '{file_name_for_log}'. Rows with invalid dates will be dropped.")
                        df.dropna(subset=[actual_col_name_in_df], inplace=True)
                    if df.empty: logger.error(f"DataFrame for '{file_name_for_log}' empty after dropping invalid dates."); return None
                
                elif expected_type == 'numeric':
                    converted_series = pd.to_numeric(series, errors='coerce')
                    num_failed_conversions = series[converted_series.isnull() & series.notnull()].count()
                    if num_failed_conversions > 0:
                        logger.warning(
                            f"Could not convert {num_failed_conversions} values in CSV column '{original_csv_col_name}' "
                            f"(mapped to '{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}') for '{file_name_for_log}' to numbers. "
                            f"Sample original values that failed: {series[converted_series.isnull() & series.notnull()].dropna().head().tolist()}. "
                            "These will be treated as NaN (missing)."
                        )
                    df[actual_col_name_in_df] = converted_series
                    if conceptual_key == 'pnl' and df[actual_col_name_in_df].isnull().all() and not series.dropna().empty:
                         error_msg = f"The mapped PnL column ('{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}' from CSV '{original_csv_col_name}') for '{file_name_for_log}' contains no valid numeric data after conversion."
                         logger.error(error_msg + f" Original series sample: {original_series_sample}")
                         st.error(error_msg)
                         return None
                
                elif expected_type == 'text':
                    df[actual_col_name_in_df] = clean_text_column(series).fillna('N/A')
                
                else: 
                    df[actual_col_name_in_df] = series.astype(str).fillna('N/A')

            except ValueError as ve: 
                error_msg = f"Error converting data for field '{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}' (CSV: '{original_csv_col_name}') for '{file_name_for_log}': {ve}. Original sample: {original_series_sample}."
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                return None
            except Exception as e_inner: 
                error_msg = f"Unexpected error processing field '{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}' (CSV: '{original_csv_col_name}') for '{file_name_for_log}': {e_inner}. Original sample: {original_series_sample}."
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                return None
        
        risk_pct_conceptual_key = 'risk_pct' 
        if risk_pct_conceptual_key in df.columns and pd.api.types.is_numeric_dtype(df[risk_pct_conceptual_key]):
            df['risk_numeric_internal'] = df[risk_pct_conceptual_key].fillna(0.0) / 100.0 
        else:
            df['risk_numeric_internal'] = 0.0 
            if risk_pct_conceptual_key not in df.columns:
                logger.warning(f"Conceptual column '{risk_pct_conceptual_key}' not mapped for '{file_name_for_log}'. Defaulting 'risk_numeric_internal' to 0.0.")
            else:
                logger.warning(f"Mapped '{risk_pct_conceptual_key}' (CSV: '{conceptual_to_original_csv_map.get(risk_pct_conceptual_key, 'N/A')}') for '{file_name_for_log}' not numeric. Defaulting 'risk_numeric_internal' to 0.0.")

        duration_conceptual_key = 'duration_minutes'
        if duration_conceptual_key in df.columns and pd.api.types.is_numeric_dtype(df[duration_conceptual_key]):
            df['duration_minutes_numeric'] = df[duration_conceptual_key].copy().fillna(pd.NA)
        else:
            df['duration_minutes_numeric'] = pd.NA
            if duration_conceptual_key not in df.columns:
                logger.warning(f"Conceptual column '{duration_conceptual_key}' not mapped for '{file_name_for_log}'. Defaulting 'duration_minutes_numeric' to NA.")
            else:
                 logger.warning(f"Mapped '{duration_conceptual_key}' (CSV: '{conceptual_to_original_csv_map.get(duration_conceptual_key, 'N/A')}') for '{file_name_for_log}' not numeric. Defaulting 'duration_minutes_numeric' to NA.")

    except Exception as e: 
        logger.error(f"Error during main type conversion/cleaning loop for '{file_name_for_log}': {e}", exc_info=True)
        st.error(f"An unexpected error occurred during data type processing for '{file_name_for_log}': {e}")
        return None

    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
        st.error(f"Critical 'date' column is missing or not in datetime format after processing for '{file_name_for_log}'. Cannot proceed.")
        logger.error(f"Feature engineering skipped for '{file_name_for_log}': 'date' column missing or not datetime.")
        return None 
        
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    try:
        if 'pnl' not in df.columns or not pd.api.types.is_numeric_dtype(df['pnl']):
            st.error(f"PnL column is missing or not numeric for '{file_name_for_log}'. Cannot perform PnL-based feature engineering.")
            logger.error(f"Feature engineering skipped for '{file_name_for_log}': PnL column missing or not numeric.")
            return df 

        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['win'] = df['pnl'] > 0

        if 'trade_outcome_csv_str' in df.columns:
            df['trade_result_processed'] = df['trade_outcome_csv_str'].astype(str).str.upper()
            valid_outcomes = ['WIN', 'LOSS', 'BREAKEVEN', 'BE'] 
            df.loc[~df['trade_result_processed'].isin(valid_outcomes), 'trade_result_processed'] = 'UNKNOWN'
            df.loc[df['trade_result_processed'] == 'BE', 'trade_result_processed'] = 'BREAKEVEN'
        else:
            df['trade_result_processed'] = np.select([df['pnl'] > 0, df['pnl'] < 0], ['WIN', 'LOSS'], default='BREAKEVEN')
        
        df['trade_hour'] = df['date'].dt.hour
        df['trade_day_of_week'] = df['date'].dt.day_name()
        df['trade_month_num'] = df['date'].dt.month
        df['trade_month_name'] = df['date'].dt.strftime('%B') 
        df['trade_year'] = df['date'].dt.year
        df['trade_date_only'] = df['date'].dt.date 

        if 'cumulative_pnl' in df.columns and not df['cumulative_pnl'].empty:
            df['drawdown_abs'], df['drawdown_pct'] = _calculate_drawdown_series_for_df(df['cumulative_pnl'])
        else:
            df['drawdown_abs'] = pd.Series(dtype=float)
            df['drawdown_pct'] = pd.Series(dtype=float)
            logger.warning(f"Could not calculate drawdown series for '{file_name_for_log}' as 'cumulative_pnl' was missing or empty.")

        if 'pnl' in df.columns and 'risk_numeric_internal' in df.columns and pd.api.types.is_numeric_dtype(df['risk_numeric_internal']):
            df['reward_risk_ratio_calculated'] = df.apply(
                lambda row: row['pnl'] / abs(row['risk_numeric_internal'])
                            if pd.notna(row['pnl']) and pd.notna(row['risk_numeric_internal']) and abs(row['risk_numeric_internal']) > 1e-9 
                            else pd.NA, axis=1
            )
        else:
            df['reward_risk_ratio_calculated'] = pd.NA
            logger.warning(f"Could not calculate 'reward_risk_ratio_calculated' for '{file_name_for_log}' due to missing PnL or valid 'risk_numeric_internal'.")

        df['trade_number'] = range(1, len(df) + 1)
        logger.info(f"Feature engineering complete for '{file_name_for_log}' using mapped column names.")
    except Exception as e:
        logger.error(f"Error in feature engineering after mapping for '{file_name_for_log}': {e}", exc_info=True)
        st.error(f"Feature engineering error after mapping for '{file_name_for_log}': {e}")
        return df 

    if df.empty:
        st.warning(f"No valid trade data found after processing and mapping for '{file_name_for_log}'."); return None

    logger.info(f"Data processing complete for '{file_name_for_log}'. Final DataFrame shape: {df.shape}. Final columns: {df.columns.tolist()}")
    return df
