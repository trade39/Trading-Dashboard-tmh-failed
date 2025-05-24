# pages/10_📝_Trade_Notes.py

import streamlit as st
import pandas as pd
import logging
import datetime as dt # Ensure datetime is imported

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, CONCEPTUAL_COLUMNS
    from components.notes_viewer import NotesViewerComponent # Assuming this component can be adapted or data prepped for it
    from utils.common_utils import display_custom_message
    from services.data_service import DataService # Import DataService
except ImportError as e:
    st.error(f"Trade Notes Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error_TNP"
    logger = logging.getLogger(APP_TITLE)
    logger.critical(f"CRITICAL IMPORT ERROR in Trade Notes Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "notes": "notes", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy", "trade_id": "trade_id"}
    CONCEPTUAL_COLUMNS = {k: k.title() for k in EXPECTED_COLUMNS.keys()}
    class NotesViewerComponent: # Mock component
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("NotesViewerComponent could not be loaded due to prior import error.")
    def display_custom_message(message, type="error"): st.error(message)
    class DataService: # Mock service
        def get_trade_notes(self, *args, **kwargs): return []
        def add_trade_note(self, *args, **kwargs): return None
        def delete_trade_note(self, *args, **kwargs): return False
    st.stop()

logger = logging.getLogger(APP_TITLE)
data_service = DataService() # Instantiate DataService

def show_trade_notes_page():
    st.title("📝 Trade Notes & Journal (Database)")
    logger.info("Rendering Trade Notes Page (Database Version).")

    # --- Section: Add New Note ---
    st.subheader("➕ Add New Trade Note")
    
    # Get the conceptual name for trade_id for user display
    trade_id_conceptual_name = CONCEPTUAL_COLUMNS.get('trade_id', 'Trade ID')
    # Get the actual column name used in the DataFrame for trade_id (after mapping)
    trade_id_col_df = EXPECTED_COLUMNS.get('trade_id')

    with st.form("add_note_form", clear_on_submit=True):
        # Input for trade_identifier (which corresponds to the 'trade_id' from the CSV)
        # Users need to know which trade they are adding a note for.
        # Providing a dropdown of available trade_ids from the session_state.filtered_data would be ideal.
        trade_identifier_input = ""
        if 'filtered_data' in st.session_state and st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty and trade_id_col_df and trade_id_col_df in st.session_state.filtered_data.columns:
            available_trade_ids = st.session_state.filtered_data[trade_id_col_df].dropna().astype(str).unique().tolist()
            if available_trade_ids:
                trade_identifier_input = st.selectbox(
                    f"Select or Enter {trade_id_conceptual_name} to Link Note:",
                    options=[""] + sorted(available_trade_ids), # Add an empty option for manual entry
                    index=0,
                    help=f"Select the {trade_id_conceptual_name} from your trade data that this note refers to. If not listed, you can type it if known."
                )
                # Allow manual override/entry if a free-text field is preferred for non-listed IDs
                if not trade_identifier_input: # If user selected the empty string, allow manual input
                     trade_identifier_manual_input = st.text_input(f"Or Manually Enter {trade_id_conceptual_name} (if not in list):", key="manual_trade_id_for_note")
                     if trade_identifier_manual_input:
                         trade_identifier_input = trade_identifier_manual_input
            else:
                trade_identifier_input = st.text_input(f"Enter {trade_id_conceptual_name} to Link Note:", help="The unique identifier of the trade this note is for.")
        else:
            trade_identifier_input = st.text_input(f"Enter {trade_id_conceptual_name} to Link Note:", help="The unique identifier of the trade this note is for.")

        note_content_input = st.text_area("Note Content:", height=150, placeholder="Enter your detailed trade observations, lessons learned, etc.")
        tags_input = st.text_input("Tags (comma-separated):", placeholder="e.g., FOMO, good_setup, news_event")
        
        submitted_add_note = st.form_submit_button("💾 Save Note")

        if submitted_add_note:
            if not note_content_input:
                st.warning("Note content cannot be empty.")
            elif not trade_identifier_input:
                st.warning(f"{trade_id_conceptual_name} is required to link the note to a trade.")
            else:
                added_note = data_service.add_trade_note(
                    trade_identifier=str(trade_identifier_input).strip(), # Ensure it's a string
                    note_content=note_content_input,
                    tags=tags_input if tags_input.strip() else None
                )
                if added_note:
                    st.success(f"Note for {trade_id_conceptual_name} '{added_note.trade_identifier}' saved successfully!")
                    st.rerun() # Rerun to refresh the notes list
                else:
                    st.error("Failed to save the note. Please check logs.")
    
    st.markdown("---")

    # --- Section: View Existing Notes ---
    st.subheader("📖 View Existing Notes")

    # Filtering options
    col_filter1, col_filter2 = st.columns([2,1])
    filter_trade_id = col_filter1.text_input("Filter by Trade ID:", key="filter_notes_trade_id")
    # Date range filter (optional, can be added later for more complexity)
    # filter_start_date = col_filter2.date_input("Start Date", value=None, key="filter_notes_start_date")
    # filter_end_date = col_filter2.date_input("End Date", value=None, key="filter_notes_end_date")

    try:
        # Fetch notes from the database
        # For now, fetch all or filter by trade_id if provided
        db_notes = data_service.get_trade_notes(
            trade_identifier=filter_trade_id if filter_trade_id else None
            # start_date=filter_start_date, # Implement if date filters are added
            # end_date=filter_end_date
        )

        if not db_notes:
            if filter_trade_id:
                display_custom_message(f"No notes found for Trade ID '{filter_trade_id}'. You can add one above.", "info")
            else:
                display_custom_message("No trade notes found in the database. Add your first note above!", "info")
        else:
            # Prepare data for NotesViewerComponent or direct display
            # The NotesViewerComponent expects a DataFrame. Let's construct one.
            # We need to decide what trade details to show alongside the note.
            # For now, we'll primarily show note data and the linked trade_identifier.
            
            notes_display_data = []
            for note_db in db_notes:
                note_data = {
                    "note_id_db": note_db.id, # Keep DB ID for potential delete/edit
                    "trade_identifier": note_db.trade_identifier, # This is the link to the CSV's trade_id
                    # The 'date' for the note itself is note_db.note_timestamp
                    # The 'date' for the trade would come from joining with filtered_df
                    "note_date": note_db.note_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if note_db.note_timestamp else "N/A",
                    "notes": note_db.note_content, # This aligns with one of the component's expected conceptual columns
                    "tags": note_db.tags,
                    # --- Placeholder for trade-specific details ---
                    # These would ideally be fetched by looking up note_db.trade_identifier
                    # in the main st.session_state.filtered_data
                    "pnl": None, # Placeholder
                    "symbol": None, # Placeholder
                    "strategy": None, # Placeholder
                    "trade_date": None # Placeholder for actual trade date
                }

                # Attempt to enrich with data from filtered_df if trade_id_col_df is valid
                if 'filtered_data' in st.session_state and st.session_state.filtered_data is not None and trade_id_col_df and trade_id_col_df in st.session_state.filtered_data.columns:
                    trade_df = st.session_state.filtered_data
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
            
            notes_df_for_component = pd.DataFrame(notes_display_data)

            if not notes_df_for_component.empty:
                # The NotesViewerComponent needs 'date', 'notes', 'pnl', 'symbol', 'strategy'
                # We've named our columns to match these conceptual ideas where possible.
                # The 'date' column for the component should ideally be the *trade date*.
                # We are using 'note_date' for the note's own timestamp for now if trade_date isn't found.
                
                # Ensure essential columns for the component exist, even if None/NaN
                component_date_col = 'trade_date' if 'trade_date' in notes_df_for_component else 'note_date'
                
                # Rename for component compatibility if needed, or adapt component
                # For now, we assume NotesViewerComponent can handle these column names.
                # If it strictly needs 'date', 'pnl', etc., we would rename:
                # notes_df_for_component.rename(columns={'note_date': 'date', 'note_content': 'notes'}, inplace=True)

                st.caption(f"Displaying {len(notes_df_for_component)} notes.")
                
                # --- Direct Display (if NotesViewerComponent is problematic or for simplicity) ---
                for index, row_note in notes_df_for_component.iterrows():
                    card_md = f"""
                    <div class="custom-card" style="margin-bottom: 1rem; border-left: 5px solid var(--primary-color);">
                        <p style="font-size: 0.8rem; color: var(--text-muted-color);">
                            <strong>Note ID (DB):</strong> {row_note['note_id_db']} | 
                            <strong>{trade_id_conceptual_name}:</strong> {row_note['trade_identifier']} | 
                            <strong>Noted:</strong> {row_note['note_date']}
                        </p>
                        {f"<p style='font-size: 0.9rem;'><strong>Trade Date:</strong> {row_note['trade_date']} | <strong>Symbol:</strong> {row_note['symbol']} | <strong>Strategy:</strong> {row_note['strategy']} | <strong>PnL:</strong> {row_note['pnl'] if row_note['pnl'] is not None else 'N/A'}</p>" if row_note['trade_date'] else ""}
                        <hr style="margin: 0.5rem 0;">
                        <p style="white-space: pre-wrap; margin-bottom: 0.5rem;">{row_note['notes']}</p>
                        {f"<p style='font-size: 0.85rem;'><strong>Tags:</strong> <em>{row_note['tags']}</em></p>" if row_note['tags'] else ""}
                    </div>
                    """
                    # Add a delete button for each note
                    # The key for the button must be unique, e.g., using note_id_db
                    if st.button(f"🗑️ Delete Note ID {row_note['note_id_db']}", key=f"delete_note_{row_note['note_id_db']}", help="Permanently delete this note."):
                        if data_service.delete_trade_note(row_note['note_id_db']):
                            st.success(f"Note ID {row_note['note_id_db']} deleted.")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete note ID {row_note['note_id_db']}.")
                    st.markdown(card_md, unsafe_allow_html=True)
                    st.markdown("---")


                # --- Original NotesViewerComponent call (commented out for direct display first) ---
                # notes_component = NotesViewerComponent(
                #     notes_dataframe=notes_df_for_component,
                #     date_col=component_date_col, # Use the determined date column
                #     notes_col='notes', # This must be the column name in notes_df_for_component
                #     pnl_col='pnl' if 'pnl' in notes_df_for_component else None,
                #     symbol_col='symbol' if 'symbol' in notes_df_for_component else None,
                #     strategy_col='strategy' if 'strategy' in notes_df_for_component else None,
                #     # Add other necessary params for NotesViewerComponent
                #     default_sort_by_display="Date", # Or "Note Date"
                #     default_sort_ascending=False,
                #     items_per_page=5,
                #     # Pass the database note ID if the component can handle it for actions
                #     db_id_col='note_id_db' if 'note_id_db' in notes_df_for_component else None
                # )
                # notes_component.render()

            else:
                display_custom_message("Formatted notes data is empty. Cannot display.", "info")

    except Exception as e_render_notes:
        logger.error(f"Error rendering trade notes view: {e_render_notes}", exc_info=True)
        display_custom_message(f"An error occurred while displaying trade notes: {e_render_notes}", "error")

if __name__ == "__main__":
    # Ensure logger is configured for standalone run if needed
    if not logging.getLogger(APP_TITLE).hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if 'app_initialized' not in st.session_state:
        # Minimal session state for standalone testing
        st.session_state.filtered_data = pd.DataFrame({
            EXPECTED_COLUMNS.get('trade_id', 'trade_id'): ['TRADE001', 'TRADE002', 'TRADE003'],
            EXPECTED_COLUMNS.get('date', 'date'): [pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-02'), pd.to_datetime('2023-01-03')],
            EXPECTED_COLUMNS.get('symbol', 'symbol'): ['AAPL', 'MSFT', 'GOOG'],
            EXPECTED_COLUMNS.get('strategy', 'strategy'): ['Alpha1', 'Beta2', 'Alpha1'],
            EXPECTED_COLUMNS.get('pnl', 'pnl'): [100, -50, 200]
        })
        st.session_state.app_initialized = True
        logger.info("Minimal session state initialized for Trade Notes standalone test.")
        st.sidebar.info("Trade Notes Page: Running in standalone test mode with mock data.")

    show_trade_notes_page()
