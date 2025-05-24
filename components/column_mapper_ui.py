# components/column_mapper_ui.py
"""
Component for allowing users to map their uploaded CSV columns
to the application's expected conceptual columns, with data preview,
enhanced auto-mapping, data type validation, categorized display,
and confirm buttons at top and bottom.
Now accepts an initial_mapping_override.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any, OrderedDict as TypingOrderedDict # Use typing.OrderedDict for type hint
from collections import OrderedDict # Use collections.OrderedDict for runtime
from io import BytesIO
from thefuzz import fuzz
import re

try:
    from config import (
        APP_TITLE,
        CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES,
        CONCEPTUAL_COLUMN_SYNONYMS,
        CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_CATEGORIES
    )
except ImportError:
    APP_TITLE = "TradingDashboard_Default"
    # ... (fallback definitions as before) ...
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
        conceptual_column_categories: TypingOrderedDict[str, List[str]], # Use typing.OrderedDict for hint
        initial_mapping_override: Optional[Dict[str, Optional[str]]] = None # New parameter
    ):
        self.uploaded_file_name = uploaded_file_name
        self.uploaded_file_bytes = uploaded_file_bytes
        self.csv_headers = [""] + csv_headers  # Add empty option for unmapping
        self.raw_csv_headers = csv_headers # Keep original for preview matching
        self.conceptual_columns_map = conceptual_columns_map 
        self.conceptual_column_types = conceptual_column_types
        self.conceptual_column_synonyms = conceptual_column_synonyms
        self.critical_conceptual_cols = critical_conceptual_cols if critical_conceptual_cols else []
        self.conceptual_column_categories = conceptual_column_categories # This is collections.OrderedDict
        
        self.initial_mapping_override = initial_mapping_override # Store the override
        self.mapping: Dict[str, Optional[str]] = {} # This will be populated by selectboxes
        self.preview_df: Optional[pd.DataFrame] = None

        if self.uploaded_file_bytes:
            try:
                self.uploaded_file_bytes.seek(0)
                # Attempt to read with encoding detection for preview robustness
                try:
                    self.preview_df = pd.read_csv(self.uploaded_file_bytes, nrows=5)
                except UnicodeDecodeError:
                    self.uploaded_file_bytes.seek(0)
                    import chardet # Local import for this specific case
                    raw_sample = self.uploaded_file_bytes.read(20000)
                    detected_encoding = chardet.detect(raw_sample)['encoding']
                    self.uploaded_file_bytes.seek(0)
                    if detected_encoding:
                        self.preview_df = pd.read_csv(self.uploaded_file_bytes, nrows=5, encoding=detected_encoding)
                    else: # Fallback if chardet fails
                        self.uploaded_file_bytes.seek(0)
                        self.preview_df = pd.read_csv(self.uploaded_file_bytes, nrows=5, encoding='latin1') # Try latin1 as common fallback
                self.uploaded_file_bytes.seek(0) 
            except Exception as e:
                logger.error(f"ColumnMapperUI: Error reading CSV for preview ('{self.uploaded_file_name}'): {e}", exc_info=True)
                self.preview_df = None # Ensure preview_df is None on error
        
        logger.debug(f"ColumnMapperUI initialized for file: {self.uploaded_file_name}. Initial override: {self.initial_mapping_override is not None}")

    def _normalize_header(self, header: str) -> str:
        # ... (implementation as before) ...
        if not isinstance(header, str): header = str(header)
        normalized = header.strip().lower().replace(':', '_').replace('%', 'pct')
        normalized = re.sub(r'[\s\-\./\(\)]+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized).strip('_')
        return normalized

    def _attempt_automatic_mapping(self) -> Dict[str, Optional[str]]:
        # ... (implementation as before) ...
        # This method provides the default auto-mapping if no override is given.
        auto_mapping: Dict[str, Optional[str]] = {}
        normalized_csv_headers_map = {self._normalize_header(h): h for h in self.raw_csv_headers}
        used_csv_headers = set() 

        specific_csv_header_targets = {
            "trade_model": "strategy", "r_r": "r_r_csv_num", "pnl": "pnl", "date": "date",
            "symbol_1": "symbol", "lesson_learned": "notes", "duration_mins": "duration_minutes",
            "risk_pct": "risk_pct", "entry": "entry_price", "exit": "exit_price", "size": "quantity"
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
                    if score > best_match_score and score >= FUZZY_MATCH_THRESHOLD: best_match_score = score; potential_header = original_csv_h
                if potential_header: mapped_csv_header = potential_header
            if mapped_csv_header: auto_mapping[conceptual_key] = mapped_csv_header; used_csv_headers.add(mapped_csv_header)
        return auto_mapping


    def _infer_column_data_type(self, csv_column_name: str) -> str:
        # ... (implementation as before) ...
        if self.preview_df is None or csv_column_name not in self.preview_df.columns: return "unknown"
        column_sample = self.preview_df[csv_column_name].dropna().convert_dtypes()
        if column_sample.empty: return "empty"
        try: numeric_sample = pd.to_numeric(column_sample); return "integer" if (numeric_sample % 1 == 0).all() else "float"
        except (ValueError, TypeError): pass
        try: pd.to_datetime(column_sample, errors='raise', infer_datetime_format=True); return "datetime"
        except (ValueError, TypeError, pd.errors.ParserError): pass
        return "text"

    def render(self) -> Optional[Dict[str, Optional[str]]]:
        st.markdown("<div class='column-mapper-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='component-subheader'>Map Columns for '{self.uploaded_file_name}'</h3>", unsafe_allow_html=True)
        
        # Use override if provided, otherwise attempt automatic mapping
        effective_initial_mapping = self.initial_mapping_override if self.initial_mapping_override is not None else self._attempt_automatic_mapping()
        if self.initial_mapping_override is not None:
            logger.info(f"ColumnMapperUI: Using provided initial_mapping_override for '{self.uploaded_file_name}'.")
        else:
            logger.info(f"ColumnMapperUI: Using auto-detected mapping for '{self.uploaded_file_name}'.")


        st.markdown("<div class='data-preview-container'>", unsafe_allow_html=True)
        if self.preview_df is not None and not self.preview_df.empty:
            # ... (data preview logic as before, using effective_initial_mapping for reordering hint) ...
            df_to_display = self.preview_df
            try:
                ordered_conceptual_keys = list(self.conceptual_columns_map.keys())
                mapped_csv_cols_ordered = []
                seen_mapped_csv_cols = set()
                for conceptual_key in ordered_conceptual_keys:
                    mapped_csv_header = effective_initial_mapping.get(conceptual_key) # Use effective map
                    if mapped_csv_header and mapped_csv_header in self.preview_df.columns and mapped_csv_header not in seen_mapped_csv_cols:
                        mapped_csv_cols_ordered.append(mapped_csv_header); seen_mapped_csv_cols.add(mapped_csv_header)
                remaining_original_cols = [col for col in self.preview_df.columns if col not in seen_mapped_csv_cols]
                final_display_order = mapped_csv_cols_ordered + remaining_original_cols
                if set(final_display_order) == set(self.preview_df.columns) and len(final_display_order) == len(self.preview_df.columns):
                    df_to_display = self.preview_df[final_display_order]
            except Exception as e: logger.error(f"Error reordering preview for '{self.uploaded_file_name}': {e}.")
            st.markdown("<p class='data-preview-title'>Data Preview (First 5 Rows):</p>", unsafe_allow_html=True)
            st.dataframe(df_to_display, hide_index=True, use_container_width=True)
        else:
            st.markdown("<div class='data-preview-placeholder'>Data preview is not available or the file is empty.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) 
        
        # ... (instructions markdown as before) ...
        st.markdown(
            "<div class='mapper-instructions'>"
            "<span class='instruction-icon'>ℹ️</span> " 
            "Map your CSV columns to the application's expected fields. "
            "Fields marked with a red asterisk (<strong>*</strong>) are critical. " 
            "A warning icon (⚠️) next to a selection indicates a potential data type mismatch."
            "</div>", unsafe_allow_html=True)

        form_key = f"column_mapping_form_{self.uploaded_file_name.replace('.', '_').replace(' ', '_')}_v2" # Incremented key
        with st.form(key=form_key):
            # ... (submit button top, hr, category rendering logic as before, passing effective_initial_mapping) ...
            cols_top_button = st.columns([0.75, 0.25]);
            with cols_top_button[1]: submit_button_top = st.form_submit_button("Confirm Mapping", use_container_width=True, type="primary")
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            if not self.conceptual_column_categories:
                st.warning("Column categories not defined. Displaying all columns together."); self._render_mapping_selectboxes(list(self.conceptual_columns_map.keys()), effective_initial_mapping)
            else:
                for category_name, conceptual_keys_in_category in self.conceptual_column_categories.items():
                    valid_keys_in_category = [key for key in conceptual_keys_in_category if key in self.conceptual_columns_map]
                    if not valid_keys_in_category: continue
                    has_critical = any(key in self.critical_conceptual_cols for key in valid_keys_in_category)
                    expander_label_html = f"<strong>{category_name}</strong>" + (" <span class='critical-marker'>*</span>" if has_critical else "")
                    open_by_default = any(key in self.critical_conceptual_cols and not effective_initial_mapping.get(key) for key in valid_keys_in_category)
                    st.markdown(f"<h6>{expander_label_html}</h6>", unsafe_allow_html=True)
                    with st.expander("View/Edit Mappings", expanded=open_by_default): self._render_mapping_selectboxes(valid_keys_in_category, effective_initial_mapping)
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            _, col_btn_mid, _ = st.columns([0.3, 0.4, 0.3]);
            with col_btn_mid: submit_button_bottom = st.form_submit_button("Confirm Column Mapping", use_container_width=True, type="primary")

        if submit_button_top or submit_button_bottom:
            # ... (validation logic as before) ...
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
            return {k: v for k, v in self.mapping.items() if v} # Return only mapped items
        
        st.markdown("</div>", unsafe_allow_html=True)
        return None

    def _render_mapping_selectboxes(self, conceptual_keys_to_render: List[str], initial_mapping_for_render: Dict[str, Optional[str]]):
        # ... (implementation as before, using initial_mapping_for_render for default_csv_header) ...
        cols_ui = st.columns(2); col_idx = 0
        for conceptual_key in conceptual_keys_to_render:
            if conceptual_key not in self.conceptual_columns_map: continue
            conceptual_desc = self.conceptual_columns_map[conceptual_key]; target_container = cols_ui[col_idx % 2]; col_idx += 1
            with target_container:
                is_critical = conceptual_key in self.critical_conceptual_cols
                label_html = f"{conceptual_desc}" + (" <span class='critical-marker'>*</span>" if is_critical else "")
                default_csv_header = initial_mapping_for_render.get(conceptual_key) # Use the passed initial map
                default_index = 0
                if default_csv_header and default_csv_header in self.csv_headers:
                    try: default_index = self.csv_headers.index(default_csv_header)
                    except ValueError: logger.error(f"Error finding index for '{default_csv_header}' for '{conceptual_key}'. Defaulting to 0.")
                elif default_csv_header: logger.warning(f"Default CSV header '{default_csv_header}' for '{conceptual_key}' not in available CSV headers.")
                selectbox_key = f"map_{self.uploaded_file_name.replace('.', '_').replace(' ', '_')}_{conceptual_key}_v2" # Incremented key
                st.markdown(f"<label class='selectbox-label' for='{selectbox_key}'>{label_html}</label>", unsafe_allow_html=True)
                selected_csv_col = st.selectbox("", options=self.csv_headers, index=default_index, key=selectbox_key, label_visibility="collapsed", help=f"Select CSV column for '{conceptual_desc}'. Expected type: '{self.conceptual_column_types.get(conceptual_key, 'any')}'. {'Critical.' if is_critical else 'Optional.'}")
                self.mapping[conceptual_key] = selected_csv_col if selected_csv_col else None
                if selected_csv_col:
                    inferred_type = self._infer_column_data_type(selected_csv_col); expected_type = self.conceptual_column_types.get(conceptual_key, "any"); type_mismatch = False
                    if expected_type == "numeric" and inferred_type not in ["integer", "float", "empty", "unknown"]: type_mismatch = True
                    elif expected_type == "datetime" and inferred_type not in ["datetime", "empty", "unknown"]: type_mismatch = True
                    if type_mismatch: st.markdown(f"<small class='type-mismatch-warning'>⚠️ Expected '{expected_type}', but '{selected_csv_col}' looks like '{inferred_type}'.</small>", unsafe_allow_html=True)

# ... (if __name__ == "__main__": block as before, no changes needed for this specific update) ...
if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="Column Mapper Test")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_test = logging.getLogger(APP_TITLE)
    logger_test.info("Starting ColumnMapperUI test run...")
    try:
        with open("style.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: st.warning("style.css not found.")
    st.title("Test Categorized Column Mapper UI (with Initial Override)")
    _MOCK_CONCEPTUAL_COLUMNS = OrderedDict([("date", "Date"), ("pnl", "PnL"), ("symbol", "Symbol"), ("quantity", "Qty")])
    _MOCK_CONCEPTUAL_COLUMN_TYPES = {"date": "datetime", "pnl": "numeric", "symbol": "text", "quantity": "numeric"}
    _MOCK_CONCEPTUAL_COLUMN_SYNONYMS = {"pnl": ["profit"], "date": ["timestamp"], "quantity": ["size", "volume"]}
    _MOCK_CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol"]
    _MOCK_CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([("Core", ["date", "symbol", "pnl", "quantity"])])
    
    sample_csv_content_1 = "Trade Date,Profit/Loss,Ticker,Volume\n2023-01-01,100,AAPL,10"
    mock_file_bytes_1 = BytesIO(sample_csv_content_1.encode('utf-8'))
    mock_csv_headers_1 = sample_csv_content_1.splitlines()[0].split(',')

    st.write("### Scenario 1: No Initial Override (Auto-detection)")
    mapper1 = ColumnMapperUI("test1.csv", mock_file_bytes_1, mock_csv_headers_1, _MOCK_CONCEPTUAL_COLUMNS, _MOCK_CONCEPTUAL_COLUMN_TYPES, _MOCK_CONCEPTUAL_COLUMN_SYNONYMS, _MOCK_CRITICAL_CONCEPTUAL_COLUMNS, _MOCK_CONCEPTUAL_COLUMN_CATEGORIES)
    res1 = mapper1.render()
    if res1: st.success("Mapping 1 Confirmed:"); st.json(res1)

    st.write("### Scenario 2: With Initial Override")
    # Simulate a previously saved mapping
    saved_mapping_override = {"date": "Trade Date", "pnl": "Profit/Loss", "symbol": "Ticker", "quantity": "Volume"}
    mock_file_bytes_1.seek(0) # Reset for second mapper instance
    mapper2 = ColumnMapperUI("test2_override.csv", mock_file_bytes_1, mock_csv_headers_1, _MOCK_CONCEPTUAL_COLUMNS, _MOCK_CONCEPTUAL_COLUMN_TYPES, _MOCK_CONCEPTUAL_COLUMN_SYNONYMS, _MOCK_CRITICAL_CONCEPTUAL_COLUMNS, _MOCK_CONCEPTUAL_COLUMN_CATEGORIES, initial_mapping_override=saved_mapping_override)
    res2 = mapper2.render()
    if res2: st.success("Mapping 2 Confirmed (with override):"); st.json(res2)
    logger_test.info("ColumnMapperUI test run finished.")
