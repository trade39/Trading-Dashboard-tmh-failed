# pages/10_📝_Trade_Notes.py

import streamlit as st
import pandas as pd
import logging
import datetime as dt

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, CONCEPTUAL_COLUMNS
    # Assuming NotesViewerComponent is not being used for now, as per previous simplifications.
    # If it were, it would be imported here.
    # from components.notes_viewer import NotesViewerComponent
    from utils.common_utils import display_custom_message, format_currency
    from services.data_service import DataService
except ImportError as e:
    st.error(f"Trade Notes Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error_TNP"
    logger = logging.getLogger(APP_TITLE)
    logger.critical(f"CRITICAL IMPORT ERROR in Trade Notes Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "notes": "notes", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy", "trade_id": "trade_id"}
    CONCEPTUAL_COLUMNS = {k: k.title() for k in EXPECTED_COLUMNS.keys()}
    def display_custom_message(message, type="error"): st.error(message)
    class DataService: # Mock service
        def get_trade_notes(self, *args, **kwargs): return []
        def add_trade_note(self, *args, **kwargs): return None
        def delete_trade_note(self, *args, **kwargs): return False
    st.stop()

logger = logging.getLogger(APP_TITLE)
data_service = DataService()

def show_trade_notes_page():
    st.title("📝 Trade Notes & Journal (Database)")
    logger.info("Rendering Trade Notes Page (Database Version).")

    # --- User Authentication Check ---
    if 'authenticated_user' not in st.session_state or not st.session_state.authenticated_user:
        display_custom_message("You must be logged in to manage trade notes.", "warning")
        st.stop()
    
    current_user_id = st.session_state.authenticated_user.get('user_id')
    if not current_user_id:
        display_custom_message("User ID not found. Please log in again.", "error")
        logger.error("Trade Notes Page: current_user_id is missing from session_state.authenticated_user.")
        st.stop()

    # --- Section: Add New Note ---
    st.subheader("➕ Add New Trade Note")
    
    trade_id_conceptual_name = CONCEPTUAL_COLUMNS.get('trade_id', 'Trade ID')
    trade_id_col_df = EXPECTED_COLUMNS.get('trade_id')

    with st.form("add_note_form_user_scoped", clear_on_submit=True): # Added _user_scoped to key
        trade_identifier_input = ""
        available_trade_ids = []
        if 'filtered_data' in st.session_state and \
           st.session_state.filtered_data is not None and \
           not st.session_state.filtered_data.empty and \
           trade_id_col_df and trade_id_col_df in st.session_state.filtered_data.columns:
            
            available_trade_ids = st.session_state.filtered_data[trade_id_col_df].astype(str).dropna().unique().tolist()

        if available_trade_ids:
            trade_identifier_input = st.selectbox(
                f"Select or Enter {trade_id_conceptual_name} to Link Note:",
                options=[""] + sorted(available_trade_ids),
                index=0,
                help=f"Select the {trade_id_conceptual_name} from your currently active trade data that this note refers to. If not listed, you can type it if known."
            )
            if not trade_identifier_input: # If user selected the empty string, allow manual input
                trade_identifier_manual_input = st.text_input(f"Or Manually Enter {trade_id_conceptual_name} (if not in list):", key="manual_trade_id_for_note_user_scoped")
                if trade_identifier_manual_input:
                    trade_identifier_input = trade_identifier_manual_input
        else:
            trade_identifier_input = st.text_input(f"Enter {trade_id_conceptual_name} to Link Note:", help="The unique identifier of the trade this note is for (e.g., from your CSV).")

        note_content_input = st.text_area("Note Content:", height=150, placeholder="Enter your detailed trade observations, lessons learned, etc.")
        tags_input = st.text_input("Tags (comma-separated):", placeholder="e.g., FOMO, good_setup, news_event")
        
        submitted_add_note = st.form_submit_button("💾 Save Note")

        if submitted_add_note:
            if not note_content_input:
                st.warning("Note content cannot be empty.")
            elif not trade_identifier_input: # trade_identifier is now optional at DB level, but good to have for linking.
                st.warning(f"{trade_id_conceptual_name} is recommended to link the note to a specific trade.")
                # Allow saving notes not linked to a specific trade_id if desired by making trade_identifier optional
                # For now, let's keep it as recommended. If it should be truly optional, remove this elif.
            
            # Proceed even if trade_identifier_input is empty string (will be saved as None or empty by DataService)
            added_note = data_service.add_trade_note(
                user_id=current_user_id, # Pass user_id
                trade_identifier=str(trade_identifier_input).strip() if trade_identifier_input else None,
                note_content=note_content_input,
                tags=tags_input if tags_input and tags_input.strip() else None
            )
            if added_note:
                st.success(f"Note for {trade_id_conceptual_name} '{added_note.trade_identifier if added_note.trade_identifier else 'General Note'}' saved successfully!")
                st.rerun()
            else:
                st.error("Failed to save the note. Please check logs.")
    
    st.markdown("---")

    # --- Section: View Existing Notes ---
    st.subheader("📖 View Existing Notes")

    col_filter1, col_filter2 = st.columns([2,1])
    filter_trade_id = col_filter1.text_input("Filter by Trade ID:", key="filter_notes_trade_id_user_scoped")
    # Optional: Date range filter for notes (can be added later)
    # filter_start_date = col_filter2.date_input("Start Date (Note Timestamp)", value=None, key="filter_notes_start_date_user_scoped")
    # filter_end_date = col_filter2.date_input("End Date (Note Timestamp)", value=None, key="filter_notes_end_date_user_scoped")

    try:
        db_notes = data_service.get_trade_notes(
            user_id=current_user_id, # Pass user_id
            trade_identifier=filter_trade_id if filter_trade_id else None
            # start_date=filter_start_date,
            # end_date=filter_end_date
        )

        if not db_notes:
            if filter_trade_id:
                display_custom_message(f"No notes found for Trade ID '{filter_trade_id}' for your account.", "info")
            else:
                display_custom_message("You haven't added any trade notes yet. Use the form above to add your first note!", "info")
        else:
            notes_display_data = []
            for note_db in db_notes:
                note_data = {
                    "note_id_db": note_db.id,
                    "trade_identifier": note_db.trade_identifier,
                    "note_date": note_db.note_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if note_db.note_timestamp else "N/A",
                    "notes": note_db.note_content,
                    "tags": note_db.tags,
                    "pnl": None, "symbol": None, "strategy": None, "trade_date": None
                }

                # Enrich with trade data from current session's filtered_data if trade_id matches
                if note_db.trade_identifier and \
                   'filtered_data' in st.session_state and \
                   st.session_state.filtered_data is not None and \
                   trade_id_col_df and trade_id_col_df in st.session_state.filtered_data.columns:
                    
                    trade_df = st.session_state.filtered_data
                    # Ensure comparison is robust (e.g., string to string)
                    matching_trade = trade_df[trade_df[trade_id_col_df].astype(str) == str(note_db.trade_identifier)]
                    if not matching_trade.empty:
                        trade_info = matching_trade.iloc[0]
                        note_data['pnl'] = trade_info.get(EXPECTED_COLUMNS.get('pnl'))
                        note_data['symbol'] = trade_info.get(EXPECTED_COLUMNS.get('symbol'))
                        note_data['strategy'] = trade_info.get(EXPECTED_COLUMNS.get('strategy'))
                        trade_date_val = trade_info.get(EXPECTED_COLUMNS.get('date'))
                        if pd.notnull(trade_date_val):
                             note_data['trade_date'] = pd.to_datetime(trade_date_val).strftime('%Y-%m-%d') if isinstance(trade_date_val, (str, pd.Timestamp, dt.date, dt.datetime)) else str(trade_date_val)
                notes_display_data.append(note_data)
            
            notes_df_for_display = pd.DataFrame(notes_display_data)

            if not notes_df_for_display.empty:
                st.caption(f"Displaying {len(notes_df_for_display)} notes for your account.")
                for index, row_note in notes_df_for_display.iterrows():
                    card_md = f"""
                    <div class="custom-card" style="margin-bottom: 1rem; border-left: 5px solid var(--primary-color);">
                        <p style="font-size: 0.8rem; color: var(--text-muted-color);">
                            <strong>Note ID (DB):</strong> {row_note['note_id_db']} | 
                            <strong>{trade_id_conceptual_name}:</strong> {row_note['trade_identifier'] if row_note['trade_identifier'] else "N/A"} | 
                            <strong>Noted:</strong> {row_note['note_date']}
                        </p>
                        """
                    if row_note['trade_date']: # Only show trade details if enriched
                        card_md += f"""
                        <p style='font-size: 0.9rem;'>
                            <strong>Trade Date:</strong> {row_note['trade_date']} | 
                            <strong>Symbol:</strong> {row_note.get('symbol', 'N/A')} | 
                            <strong>Strategy:</strong> {row_note.get('strategy', 'N/A')} | 
                            <strong>PnL:</strong> {format_currency(row_note['pnl']) if pd.notna(row_note['pnl']) else 'N/A'}
                        </p>"""
                    card_md += f"""
                        <hr style="margin: 0.5rem 0;">
                        <p style="white-space: pre-wrap; margin-bottom: 0.5rem;">{st.markdown(row_note['notes'], unsafe_allow_html=True) if "##" in row_note['notes'] or "**" in row_note['notes'] else row_note['notes']}</p>
                        {f"<p style='font-size: 0.85rem;'><strong>Tags:</strong> <em>{row_note['tags']}</em></p>" if row_note['tags'] else ""}
                    </div>
                    """
                    if st.button(f"🗑️ Delete Note ID {row_note['note_id_db']}", key=f"delete_note_{row_note['note_id_db']}_user_scoped", help="Permanently delete this note."):
                        if data_service.delete_trade_note(user_id=current_user_id, note_id=row_note['note_id_db']): # Pass user_id
                            st.success(f"Note ID {row_note['note_id_db']} deleted.")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete note ID {row_note['note_id_db']}.")
                    st.markdown(card_md, unsafe_allow_html=True)
                    st.markdown("---")
            else:
                display_custom_message("No notes found matching your criteria for your account.", "info")

    except Exception as e_render_notes:
        logger.error(f"Error rendering trade notes view: {e_render_notes}", exc_info=True)
        display_custom_message(f"An error occurred while displaying trade notes: {e_render_notes}", "error")

if __name__ == "__main__":
    if not logging.getLogger(APP_TITLE).hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if 'app_initialized' not in st.session_state:
        st.session_state.filtered_data = pd.DataFrame({
            EXPECTED_COLUMNS.get('trade_id', 'trade_id'): ['TRADE001', 'TRADE002', 'TRADE003'],
            EXPECTED_COLUMNS.get('date', 'date'): [pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-02'), pd.to_datetime('2023-01-03')],
            EXPECTED_COLUMNS.get('symbol', 'symbol'): ['AAPL', 'MSFT', 'GOOG'],
            EXPECTED_COLUMNS.get('strategy', 'strategy'): ['Alpha1', 'Beta2', 'Alpha1'],
            EXPECTED_COLUMNS.get('pnl', 'pnl'): [100, -50, 200]
        })
        st.session_state.app_initialized = True
        # Mock authenticated user for standalone testing
        st.session_state.authenticated_user = {'user_id': 1, 'username': 'testuser'} 
        logger.info("Minimal session state initialized for Trade Notes standalone test with user_id.")
        st.sidebar.info("Trade Notes Page: Running in standalone test mode with mock data and user_id.")

    show_trade_notes_page()
