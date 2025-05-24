# components/column_mapper_ui.py
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any, OrderedDict as TypingOrderedDict
from collections import OrderedDict
from io import BytesIO
from thefuzz import fuzz
import re
import chardet
import os # Added for os.SEEK_END

try:
    from config import (
        APP_TITLE, CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_TYPES,
        CONCEPTUAL_COLUMN_SYNONYMS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_CATEGORIES
    )
except ImportError:
    APP_TITLE = "TradingDashboard_Default"
    CONCEPTUAL_COLUMNS = {"date": "Date", "pnl": "PnL"}
    CONCEPTUAL_COLUMN_TYPES = {"date": "datetime", "pnl": "numeric"}
    CONCEPTUAL_COLUMN_SYNONYMS = {"pnl": ["profit"], "date": ["timestamp"]}
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([("Core", ["date", "pnl"])])

import logging
logger = logging.getLogger(APP_TITLE)

class ColumnMapperUI:
    def __init__(
        self,
        uploaded_file_name: str,
        uploaded_file_bytes: Optional[BytesIO],
        csv_headers: List[str],
        conceptual_columns_map: Dict[str, str],
        conceptual_column_types: Dict[str, str],
        conceptual_column_synonyms: Dict[str, List[str]],
        critical_conceptual_cols: List[str],
        conceptual_column_categories: TypingOrderedDict[str, List[str]],
        initial_mapping_override: Optional[Dict[str, Optional[str]]] = None,
        detected_encoding: Optional[str] = None # NEW: Accept detected encoding
    ):
        self.uploaded_file_name = uploaded_file_name
        self.uploaded_file_bytes_for_preview = uploaded_file_bytes
        self.csv_headers_for_select = [""] + csv_headers
        self.raw_csv_headers_for_preview_matching = csv_headers
        self.conceptual_columns_map = conceptual_columns_map
        self.conceptual_column_types = conceptual_column_types
        self.conceptual_column_synonyms = conceptual_column_synonyms
        self.critical_conceptual_cols = critical_conceptual_cols if critical_conceptual_cols else []
        self.conceptual_column_categories = conceptual_column_categories
        self.initial_mapping_override = initial_mapping_override
        self.passed_detected_encoding = detected_encoding # Store passed encoding
        self.mapping: Dict[str, Optional[str]] = {}
        self.preview_df: Optional[pd.DataFrame] = None

        if self.uploaded_file_bytes_for_preview:
            try:
                with st.spinner("Generating data preview for mapping... (max 50KB sample)"):
                    self.uploaded_file_bytes_for_preview.seek(0)
                    sample_limit_bytes = 50 * 1024  # 50KB
                    
                    actual_buffer_size = 0
                    if hasattr(self.uploaded_file_bytes_for_preview, 'getbuffer'):
                        actual_buffer_size = self.uploaded_file_bytes_for_preview.getbuffer().nbytes
                    elif hasattr(self.uploaded_file_bytes_for_preview, 'seek') and hasattr(self.uploaded_file_bytes_for_preview, 'tell'):
                        current_pos = self.uploaded_file_bytes_for_preview.tell()
                        self.uploaded_file_bytes_for_preview.seek(0, os.SEEK_END)
                        actual_buffer_size = self.uploaded_file_bytes_for_preview.tell()
                        self.uploaded_file_bytes_for_preview.seek(current_pos) 
                    
                    bytes_to_read_for_preview = min(actual_buffer_size, sample_limit_bytes) if actual_buffer_size > 0 else sample_limit_bytes
                    preview_sample_bytes = self.uploaded_file_bytes_for_preview.read(bytes_to_read_for_preview)
                    self.uploaded_file_bytes_for_preview.seek(0) 

                    preview_io = BytesIO(preview_sample_bytes)
                    
                    primary_encoding_to_try = self.passed_detected_encoding if self.passed_detected_encoding else 'utf-8'
                    fallback_encodings = ['latin1', 'cp1252']
                    
                    read_successful = False
                    # Try with passed/default encoding first
                    try:
                        logger.info(f"ColumnMapperUI preview: Attempting to read with primary encoding: {primary_encoding_to_try}")
                        self.preview_df = pd.read_csv(preview_io, nrows=5, engine='python', skipinitialspace=True, encoding=primary_encoding_to_try)
                        logger.info(f"ColumnMapperUI preview: Successfully read with {primary_encoding_to_try}.")
                        read_successful = True
                    except (UnicodeDecodeError, pd.errors.ParserError) as e_primary:
                        logger.warning(f"ColumnMapperUI preview: Primary encoding '{primary_encoding_to_try}' failed ({e_primary}).")
                        preview_io.seek(0) # Reset for next attempt

                    # Try fallback encodings if primary failed
                    if not read_successful:
                        for enc in fallback_encodings:
                            try:
                                logger.info(f"ColumnMapperUI preview: Trying fallback encoding: {enc}")
                                self.preview_df = pd.read_csv(preview_io, nrows=5, engine='python', skipinitialspace=True, encoding=enc)
                                logger.info(f"ColumnMapperUI preview: Successfully read with fallback encoding {enc}.")
                                read_successful = True
                                break 
                            except (UnicodeDecodeError, pd.errors.ParserError):
                                logger.warning(f"ColumnMapperUI preview: Fallback encoding '{enc}' failed.")
                                preview_io.seek(0)
                    
                    # If all attempts fail, then try chardet as a last resort
                    if not read_successful and preview_sample_bytes:
                        logger.warning("ColumnMapperUI preview: All direct encoding attempts failed. Trying chardet as a last resort.")
                        preview_io.seek(0)
                        try:
                            detected_chardet_info = chardet.detect(preview_sample_bytes)
                            chardet_encoding = detected_chardet_info.get('encoding')
                            chardet_confidence = detected_chardet_info.get('confidence', 0.0)
                            logger.info(f"ColumnMapperUI preview (chardet fallback): Detected encoding {chardet_encoding} with confidence {chardet_confidence:.2f}.")
                            if chardet_encoding and chardet_confidence > 0.3: # Lower confidence threshold for fallback
                                preview_io.seek(0)
                                self.preview_df = pd.read_csv(preview_io, nrows=5, encoding=chardet_encoding, engine='python', skipinitialspace=True)
                                logger.info(f"ColumnMapperUI preview: Successfully read with chardet encoding {chardet_encoding}.")
                                read_successful = True
                            else:
                                logger.warning("ColumnMapperUI preview (chardet fallback): Low confidence or no encoding detected by chardet.")
                        except Exception as e_chardet_read:
                            logger.error(f"ColumnMapperUI preview (chardet fallback): Failed to read with chardet encoding: {e_chardet_read}", exc_info=True)
                    
                    if not read_successful:
                         logger.error(f"ColumnMapperUI preview: All encoding attempts failed for '{self.uploaded_file_name}'. Preview will be empty or None.")
                         self.preview_df = pd.DataFrame(columns=self.raw_csv_headers_for_preview_matching) if self.raw_csv_headers_for_preview_matching else None


            except pd.errors.EmptyDataError:
                logger.warning(f"ColumnMapperUI preview: CSV for '{self.uploaded_file_name}' appears empty or has no data in the first 5 rows.")
                self.preview_df = pd.DataFrame(columns=self.raw_csv_headers_for_preview_matching)
            except Exception as e_outer_preview:
                 logger.error(f"ColumnMapperUI: Outer error during preview generation ('{self.uploaded_file_name}'): {e_outer_preview}", exc_info=True)
                 self.preview_df = None
        
        logger.debug(f"ColumnMapperUI initialized for '{self.uploaded_file_name}'. Preview DF shape: {self.preview_df.shape if self.preview_df is not None else 'None'}")

    def _normalize_header(self, header: str) -> str:
        if not isinstance(header, str): header = str(header)
        normalized = header.strip().lower().replace(':', '_').replace('%', 'pct')
        normalized = re.sub(r'[\s\-\./\(\)]+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized).strip('_')
        return normalized

    def _attempt_automatic_mapping(self) -> Dict[str, Optional[str]]:
        auto_mapping: Dict[str, Optional[str]] = {}
        normalized_csv_headers_map = {self._normalize_header(h): h for h in self.raw_csv_headers_for_preview_matching}
        used_csv_headers = set()
        specific_csv_header_targets = {
            "trade_model": "strategy", "r_r": "r_r_csv_num", "pnl": "pnl", "date": "date",
            "symbol_1": "symbol", "lesson_learned": "notes", "duration_mins": "duration_minutes",
            "risk_pct": "risk_pct", "entry": "entry_price", "exit": "exit_price", "size": "trade_size_num"
        }
        for norm_specific_csv, target_conceptual_key in specific_csv_header_targets.items():
            if norm_specific_csv in normalized_csv_headers_map:
                original_csv_header = normalized_csv_headers_map[norm_specific_csv]
                if original_csv_header not in used_csv_headers and target_conceptual_key not in auto_mapping:
                    auto_mapping[target_conceptual_key] = original_csv_header
                    used_csv_headers.add(original_csv_header)
        for conceptual_key in self.conceptual_columns_map.keys():
            if conceptual_key in auto_mapping: continue
            mapped_csv_header = None; norm_conceptual_key = self._normalize_header(conceptual_key)
            if norm_conceptual_key in normalized_csv_headers_map and normalized_csv_headers_map[norm_conceptual_key] not in used_csv_headers:
                mapped_csv_header = normalized_csv_headers_map[norm_conceptual_key]
            if not mapped_csv_header and conceptual_key in self.conceptual_column_synonyms:
                for synonym in self.conceptual_column_synonyms[conceptual_key]:
                    norm_synonym = self._normalize_header(synonym)
                    if norm_synonym in normalized_csv_headers_map and normalized_csv_headers_map[norm_synonym] not in used_csv_headers:
                        mapped_csv_header = normalized_csv_headers_map[norm_synonym]; break
            FUZZY_MATCH_THRESHOLD = 85
            if not mapped_csv_header:
                best_match_score = 0; potential_header = None
                for norm_csv_h, original_csv_h in normalized_csv_headers_map.items():
                    if original_csv_h in used_csv_headers: continue
                    score = fuzz.ratio(norm_conceptual_key, norm_csv_h)
                    if score > best_match_score and score >= FUZZY_MATCH_THRESHOLD:
                        best_match_score = score; potential_header = original_csv_h
                if potential_header: mapped_csv_header = potential_header
            if mapped_csv_header:
                auto_mapping[conceptual_key] = mapped_csv_header
                used_csv_headers.add(mapped_csv_header)
        return auto_mapping

    def _infer_column_data_type(self, csv_column_name: str) -> str:
        if self.preview_df is None or csv_column_name not in self.preview_df.columns: return "unknown"
        column_sample = self.preview_df[csv_column_name].dropna().convert_dtypes()
        if column_sample.empty: return "empty"
        try:
            numeric_sample = pd.to_numeric(column_sample)
            if (numeric_sample.dropna() % 1 == 0).all(): return "integer"
            return "float"
        except (ValueError, TypeError): pass
        try: pd.to_datetime(column_sample, errors='raise', infer_datetime_format=True); return "datetime"
        except (ValueError, TypeError, pd.errors.ParserError): pass
        return "text"

    def _render_mapping_selectboxes(self, conceptual_keys_to_render: List[str], initial_mapping_for_render: Dict[str, Optional[str]]):
        cols_ui = st.columns(2); col_idx = 0
        for conceptual_key in conceptual_keys_to_render:
            if conceptual_key not in self.conceptual_columns_map: continue
            conceptual_desc = self.conceptual_columns_map[conceptual_key]; target_container = cols_ui[col_idx % 2]; col_idx += 1
            with target_container:
                is_critical = conceptual_key in self.critical_conceptual_cols
                label_html = f"{conceptual_desc}" + (" <span class='critical-marker'>*</span>" if is_critical else "")
                default_csv_header = initial_mapping_for_render.get(conceptual_key)
                default_index = 0
                if default_csv_header and default_csv_header in self.csv_headers_for_select:
                    try: default_index = self.csv_headers_for_select.index(default_csv_header)
                    except ValueError: logger.error(f"Error finding index for '{default_csv_header}' for '{conceptual_key}'.")
                elif default_csv_header: logger.warning(f"Default CSV header '{default_csv_header}' for '{conceptual_key}' not in available CSV headers for selectbox.")
                selectbox_key = f"map_{self.uploaded_file_name.replace('.', '_').replace(' ', '_')}_{conceptual_key}_v4_mapper_hang"
                st.markdown(f"<label class='selectbox-label' for='{selectbox_key}'>{label_html}</label>", unsafe_allow_html=True)
                selected_csv_col = st.selectbox("", options=self.csv_headers_for_select, index=default_index, key=selectbox_key, label_visibility="collapsed", help=f"Select CSV column for '{conceptual_desc}'. Expected type: '{self.conceptual_column_types.get(conceptual_key, 'any')}'. {'Critical.' if is_critical else 'Optional.'}")
                self.mapping[conceptual_key] = selected_csv_col if selected_csv_col else None
                if selected_csv_col:
                    inferred_type = self._infer_column_data_type(selected_csv_col); expected_type = self.conceptual_column_types.get(conceptual_key, "any"); type_mismatch = False
                    if expected_type == "numeric" and inferred_type not in ["integer", "float", "empty", "unknown"]: type_mismatch = True
                    elif expected_type == "datetime" and inferred_type not in ["datetime", "empty", "unknown"]: type_mismatch = True
                    if type_mismatch: st.markdown(f"<small class='type-mismatch-warning'>⚠️ Expected '{expected_type}', but '{selected_csv_col}' looks like '{inferred_type}'.</small>", unsafe_allow_html=True)

    def render(self) -> Optional[Dict[str, Optional[str]]]:
        st.markdown("<div class='column-mapper-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='component-subheader'>Map Columns for '{self.uploaded_file_name}'</h3>", unsafe_allow_html=True)
        effective_initial_mapping = self.initial_mapping_override if self.initial_mapping_override is not None else self._attempt_automatic_mapping()
        if self.initial_mapping_override is not None: logger.info(f"ColumnMapperUI: Using provided initial_mapping_override for '{self.uploaded_file_name}'.")
        else: logger.info(f"ColumnMapperUI: Using auto-detected mapping for '{self.uploaded_file_name}'.")

        st.markdown("<div class='data-preview-container'>", unsafe_allow_html=True)
        if self.preview_df is not None and not self.preview_df.empty:
            df_to_display_preview = self.preview_df.copy()
            try:
                ordered_conceptual_keys = list(self.conceptual_columns_map.keys())
                mapped_csv_cols_ordered = []
                seen_mapped_csv_cols = set()
                for conceptual_key in ordered_conceptual_keys:
                    mapped_csv_header = effective_initial_mapping.get(conceptual_key)
                    if mapped_csv_header and mapped_csv_header in df_to_display_preview.columns and mapped_csv_header not in seen_mapped_csv_cols:
                        mapped_csv_cols_ordered.append(mapped_csv_header); seen_mapped_csv_cols.add(mapped_csv_header)
                remaining_original_cols = [col for col in df_to_display_preview.columns if col not in seen_mapped_csv_cols]
                final_display_order = mapped_csv_cols_ordered + remaining_original_cols
                if set(final_display_order) == set(df_to_display_preview.columns) and len(final_display_order) == len(df_to_display_preview.columns):
                    df_to_display_preview = df_to_display_preview[final_display_order]
            except Exception as e_reorder: logger.error(f"Error reordering preview for '{self.uploaded_file_name}': {e_reorder}.")
            st.markdown("<p class='data-preview-title'>Data Preview (First 5 Rows of Sample):</p>", unsafe_allow_html=True)
            st.dataframe(df_to_display_preview, hide_index=True, use_container_width=True)
        elif self.uploaded_file_bytes_for_preview is not None:
            st.markdown("<div class='data-preview-placeholder' style='color: var(--error-color);'>Could not generate data preview. The file might be corrupted, not a standard CSV, or the initial sample (first 50KB) was unparsable. Please check the file and try re-uploading.</div>", unsafe_allow_html=True)
        else:
             st.markdown("<div class='data-preview-placeholder'>Data preview is not available (no file content).</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='mapper-instructions'><span class='instruction-icon'>ℹ️</span> Map your CSV columns to the application's expected fields. Fields marked with a red asterisk (<strong>*</strong>) are critical. A warning icon (⚠️) next to a selection indicates a potential data type mismatch.</div>", unsafe_allow_html=True)
        form_key = f"column_mapping_form_{self.uploaded_file_name.replace('.', '_').replace(' ', '_')}_v4_mapper_hang"
        with st.form(key=form_key):
            cols_top_button = st.columns([0.75, 0.25]);
            with cols_top_button[1]: submit_button_top = st.form_submit_button("Confirm Mapping", use_container_width=True, type="primary")
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            if not self.conceptual_column_categories:
                st.warning("Column categories not defined. Displaying all columns together."); self._render_mapping_selectboxes(list(self.conceptual_columns_map.keys()), effective_initial_mapping)
            else:
                for category_name, conceptual_keys_in_category in self.conceptual_column_categories.items():
                    valid_keys_in_category = [key for key in conceptual_keys_in_category if key in self.conceptual_columns_map]
                    if not valid_keys_in_category: continue
                    has_critical_in_cat = any(key in self.critical_conceptual_cols for key in valid_keys_in_category)
                    expander_label_html = f"<strong>{category_name}</strong>" + (" <span class='critical-marker'>*</span>" if has_critical_in_cat else "")
                    is_core_category = category_name.lower() in ["critical trade info", "core"]
                    open_by_default = is_core_category or any(key in self.critical_conceptual_cols and not effective_initial_mapping.get(key) for key in valid_keys_in_category)
                    st.markdown(f"<h6>{expander_label_html}</h6>", unsafe_allow_html=True)
                    with st.expander("View/Edit Mappings", expanded=open_by_default): self._render_mapping_selectboxes(valid_keys_in_category, effective_initial_mapping)
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            _, col_btn_mid, _ = st.columns([0.3, 0.4, 0.3]);
            with col_btn_mid: submit_button_bottom = st.form_submit_button("Confirm Mapping", use_container_width=True, type="primary")

        if submit_button_top or submit_button_bottom:
            missing_critical = [self.conceptual_columns_map.get(k, k) for k in self.critical_conceptual_cols if not self.mapping.get(k)]
            if missing_critical: st.error(f"Critical fields not mapped: {', '.join(missing_critical)}."); st.markdown("</div>", unsafe_allow_html=True); return None
            csv_to_critical_map: Dict[str, List[str]] = {}
            for conc_key, csv_header in self.mapping.items():
                if csv_header and conc_key in self.critical_conceptual_cols: csv_to_critical_map.setdefault(csv_header, []).append(self.conceptual_columns_map.get(conc_key, conc_key))
            has_critical_duplicates = False
            for csv_h, mapped_fields in csv_to_critical_map.items():
                if len(mapped_fields) > 1: st.error(f"CSV column '{csv_h}' mapped to multiple critical fields: {', '.join(mapped_fields)}."); has_critical_duplicates = True
            if has_critical_duplicates: st.markdown("</div>", unsafe_allow_html=True); return None
            logger.info(f"Column mapping confirmed for '{self.uploaded_file_name}': {self.mapping}")
            st.success(f"Column mapping confirmed for '{self.uploaded_file_name}'.")
            st.markdown("</div>", unsafe_allow_html=True)
            return {k: v for k, v in self.mapping.items() if v}
        st.markdown("</div>", unsafe_allow_html=True)
        return None
