# services/data_service.py

import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
import yfinance as yf
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, scoped_session
from sqlalchemy.exc import SQLAlchemyError
import datetime as dt # Use dt alias to avoid conflict

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DATABASE_URL # Import DATABASE_URL
    from data_processing import load_and_process_data
except ImportError as e:
    print(f"Warning (data_service.py): Could not import from root config or data_processing: {e}. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_Service"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy", "trade_id": "trade_id", "notes": "notes"}
    DATABASE_URL = "sqlite:///placeholder_trading_dashboard.db" # Fallback DATABASE_URL
    def load_and_process_data(uploaded_file_obj: Any, user_column_mapping: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
        if uploaded_file_obj:
            try:
                df = pd.read_csv(uploaded_file_obj)
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if 'pnl' in df.columns: df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
                return df
            except Exception as e_placeholder:
                print(f"Placeholder load_and_process_data error: {e_placeholder}")
        return None

logger = logging.getLogger(APP_TITLE)

# --- Database Setup ---
try:
    engine = create_engine(DATABASE_URL, echo=False) # Set echo=True for SQL logging during development
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info(f"Database engine created for URL: {DATABASE_URL}")
except Exception as e_db_setup:
    logger.critical(f"Failed to create database engine or sessionmaker for URL '{DATABASE_URL}': {e_db_setup}", exc_info=True)
    # Define SessionLocal and Base as None or mock them if engine creation fails
    # This allows the rest of the app to load, though DB functionality will be broken.
    SessionLocal = None
    Base = None
    engine = None # Ensure engine is None if setup fails

# --- ORM Models ---
# We'll start with a TradeNote model.
# A Trade model could also be defined if you want to store structured trade data from the CSV.
class TradeNoteDB(Base): # Renamed to avoid conflict if you have a TradeNote component
    __tablename__ = "trade_notes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # Assuming trade_id from your CSV can uniquely identify a trade.
    # If trade_id is not globally unique (e.g., resets per day/symbol),
    # you might need a composite key or a more robust way to link notes.
    # For now, let's assume 'trade_id_conceptual' (mapped from user's CSV) is the link.
    trade_identifier = Column(String, index=True, nullable=True) # This would store the value from the 'trade_id' conceptual column
    note_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    note_content = Column(Text, nullable=False)
    tags = Column(String, nullable=True) # Comma-separated tags, for example
    # Optional: Link to other details if you store them
    # symbol = Column(String, nullable=True)
    # strategy = Column(String, nullable=True)
    # pnl = Column(Float, nullable=True)

    # Add a unique constraint if trade_identifier + note_timestamp should be unique,
    # or if a note for a specific trade_identifier should be unique (then just trade_identifier if only one note per trade)
    # For simplicity, we'll rely on the auto-incrementing ID for now.

    def __repr__(self):
        return f"<TradeNoteDB(id={self.id}, trade_identifier='{self.trade_identifier}', timestamp='{self.note_timestamp.strftime('%Y-%m-%d %H:%M')}')>"

# Function to create all tables
def create_db_tables():
    if Base and engine: # Check if Base and engine were initialized
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables checked/created successfully (if they didn't exist).")
        except Exception as e_create_tables:
            logger.error(f"Error creating database tables: {e_create_tables}", exc_info=True)
            st.error(f"Database Error: Could not create tables. Check logs. Error: {e_create_tables}")
    else:
        logger.error("Database engine or Base not initialized. Cannot create tables.")
        # Optionally, display an error in Streamlit if in that context
        if 'st' in globals() and hasattr(st, 'error'):
            st.error("Database connection failed. Features requiring database will not work.")


@st.cache_resource # Cache the database session setup
def get_db_session_cached():
    """Provides a cached database session."""
    if not SessionLocal:
        logger.error("SessionLocal is not initialized. Cannot create DB session.")
        return None
    try:
        db = scoped_session(SessionLocal) # scoped_session for thread safety if needed, or just SessionLocal()
        logger.debug("Database session requested from get_db_session_cached.")
        return db
    except Exception as e_get_session:
        logger.error(f"Error creating database session in get_db_session_cached: {e_get_session}", exc_info=True)
        return None

# --- Benchmark Data Fetching (kept from your original data_service.py) ---
@st.cache_data(ttl=3600)
def get_benchmark_data_static(
    ticker: str,
    start_date_str: str,
    end_date_str: str
) -> Optional[pd.Series]:
    logger_static_func = logging.getLogger(f"{APP_TITLE}.get_benchmark_data_static")
    logger_static_func.info(f"Executing get_benchmark_data_static (caching active) for {ticker} from {start_date_str} to {end_date_str}")
    if not ticker:
        logger_static_func.info("No benchmark ticker provided. Skipping data fetch.")
        return None
    try:
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)
        if start_dt > end_dt:
            logger_static_func.warning(f"Benchmark start date {start_date_str} is after end date {end_date_str}.")
            return None
        fetch_end_dt = end_dt + pd.Timedelta(days=1)
        if start_dt == end_dt: fetch_end_dt = end_dt + pd.Timedelta(days=2)

        data = yf.download(ticker, start=start_dt, end=fetch_end_dt, progress=False, auto_adjust=True, actions=False)
        if data.empty or 'Close' not in data.columns:
            logger_static_func.warning(f"No data or 'Close' not found for benchmark {ticker}.")
            return None
        daily_adj_close = data['Close'].dropna()
        if len(daily_adj_close) < 2:
            logger_static_func.warning(f"Not enough benchmark data points for {ticker} to calculate returns.")
            return None
        daily_returns = daily_adj_close.pct_change().dropna()
        if daily_returns.empty:
            logger_static_func.warning(f"Calculated daily returns for benchmark {ticker} are empty.")
            return None
        daily_returns.name = f"{ticker}_returns"
        logger_static_func.info(f"Successfully fetched benchmark returns for {ticker}. Shape: {daily_returns.shape}")
        return daily_returns
    except Exception as e_fetch_benchmark:
        logger_static_func.error(f"Error fetching benchmark data for {ticker}: {e_fetch_benchmark}", exc_info=True)
        return None

class DataService:
    def __init__(self):
        logger.info("DataService initialized.")
        # Attempt to create tables when DataService is instantiated.
        # This should ideally be called once at app startup.
        create_db_tables()

    def get_db(self) -> Optional[scoped_session]: # Return type updated
        """Gets a database session. Handles session closing."""
        db = get_db_session_cached()
        if db is None:
            logger.error("Failed to get DB session from cached provider.")
        return db # The caller will be responsible for db.close() or db.remove() for scoped_session

    def get_processed_trading_data(
        self,
        uploaded_file_obj: Any,
        user_column_mapping: Optional[Dict[str, str]] = None,
        original_file_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if uploaded_file_obj is None:
            logger.debug("DataService: No file object provided.")
            return None
        file_name_for_log = original_file_name or getattr(uploaded_file_obj, 'name', "In-memory CSV data")
        try:
            logger.info(f"DataService: Processing file: '{file_name_for_log}' with user mapping.")
            processed_df = load_and_process_data(uploaded_file_obj, user_column_mapping=user_column_mapping)
            if processed_df is not None:
                logger.info(f"DataService: File '{file_name_for_log}' processed. Shape: {processed_df.shape}")
                # --- Potential: Save trades to a 'trades' table here if desired ---
                # self.save_trades_from_df(processed_df) # Example call
            else:
                logger.warning(f"DataService: Processing of file '{file_name_for_log}' returned None.")
            return processed_df
        except Exception as e_process:
            logger.error(f"DataService: Error processing '{file_name_for_log}': {e_process}", exc_info=True)
            return None

    def filter_data(
        self,
        df: pd.DataFrame,
        filters: Dict[str, Any],
        column_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        if df is None or df.empty:
            logger.debug("DataService: DataFrame for filtering is None or empty.")
            return pd.DataFrame()
        # ... (rest of your filter_data logic - remains unchanged for now) ...
        effective_column_map = column_map if column_map is not None else EXPECTED_COLUMNS
        filtered_df = df.copy()
        logger.info(f"DataService: Applying filters. Initial shape: {filtered_df.shape}. Filters: {filters}")
        date_col_conceptual = effective_column_map.get('date', 'date')
        date_range = filters.get('selected_date_range')
        if date_col_conceptual in filtered_df.columns and date_range and len(date_range) == 2:
            try:
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df[date_col_conceptual]).dt.date >= start_date.date()) &
                    (pd.to_datetime(filtered_df[date_col_conceptual]).dt.date <= end_date.date())
                ]
            except Exception as e_date_filter:
                logger.error(f"Error applying date filter on '{date_col_conceptual}': {e_date_filter}", exc_info=True)

        symbol_col_conceptual = effective_column_map.get('symbol', 'symbol')
        selected_symbol = filters.get('selected_symbol')
        if symbol_col_conceptual in filtered_df.columns and selected_symbol and selected_symbol != "All":
            try:
                filtered_df = filtered_df[filtered_df[symbol_col_conceptual].astype(str) == str(selected_symbol)]
            except Exception as e_symbol_filter:
                logger.error(f"Error applying symbol filter on '{symbol_col_conceptual}': {e_symbol_filter}", exc_info=True)

        strategy_col_conceptual = effective_column_map.get('strategy', 'strategy')
        selected_strategy = filters.get('selected_strategy')
        if strategy_col_conceptual in filtered_df.columns and selected_strategy and selected_strategy != "All":
            try:
                filtered_df = filtered_df[filtered_df[strategy_col_conceptual].astype(str) == str(selected_strategy)]
            except Exception as e_strat_filter:
                logger.error(f"Error applying strategy filter on '{strategy_col_conceptual}': {e_strat_filter}", exc_info=True)
        logger.info(f"DataService: Filtering complete. Final shape: {filtered_df.shape}")
        return filtered_df

    # --- New Database Methods for Trade Notes ---
    def add_trade_note(self, trade_identifier: Optional[str], note_content: str, tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        """Adds a new trade note to the database."""
        db = self.get_db()
        if not db: return None
        
        new_note = TradeNoteDB(
            trade_identifier=trade_identifier,
            note_content=note_content,
            tags=tags,
            note_timestamp=dt.datetime.utcnow() # Ensure timestamp is set
        )
        try:
            db.add(new_note)
            db.commit()
            db.refresh(new_note)
            logger.info(f"Added new trade note ID: {new_note.id} for trade_identifier: '{trade_identifier}'")
            return new_note
        except SQLAlchemyError as e_add_note:
            db.rollback()
            logger.error(f"Error adding trade note for '{trade_identifier}': {e_add_note}", exc_info=True)
            return None
        finally:
            db.close() # or db.remove() if using scoped_session and managing it per request

    def get_trade_notes(
        self,
        trade_identifier: Optional[str] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        limit: Optional[int] = None
    ) -> List[TradeNoteDB]:
        """Retrieves trade notes, optionally filtered by trade_identifier or date range."""
        db = self.get_db()
        if not db: return []
        
        try:
            query = db.query(TradeNoteDB)
            if trade_identifier:
                query = query.filter(TradeNoteDB.trade_identifier == trade_identifier)
            if start_date:
                query = query.filter(TradeNoteDB.note_timestamp >= start_date)
            if end_date:
                # Adjust end_date to include the whole day
                end_date_inclusive = end_date + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
                query = query.filter(TradeNoteDB.note_timestamp <= end_date_inclusive)
            
            query = query.order_by(TradeNoteDB.note_timestamp.desc())
            
            if limit:
                query = query.limit(limit)
            
            notes = query.all()
            logger.info(f"Retrieved {len(notes)} trade notes with filters: trade_id='{trade_identifier}', start='{start_date}', end='{end_date}', limit={limit}")
            return notes
        except SQLAlchemyError as e_get_notes:
            logger.error(f"Error retrieving trade notes: {e_get_notes}", exc_info=True)
            return []
        finally:
            db.close()

    def update_trade_note(self, note_id: int, new_content: Optional[str] = None, new_tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        """Updates the content or tags of an existing trade note."""
        db = self.get_db()
        if not db: return None

        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id).first()
            if not note:
                logger.warning(f"Trade note with ID {note_id} not found for update.")
                return None
            
            updated = False
            if new_content is not None:
                note.note_content = new_content
                updated = True
            if new_tags is not None: # Allows clearing tags by passing ""
                note.tags = new_tags
                updated = True
            
            if updated:
                note.note_timestamp = dt.datetime.utcnow() # Update timestamp on modification
                db.commit()
                db.refresh(note)
                logger.info(f"Trade note ID {note_id} updated.")
            else:
                logger.info(f"No changes applied to trade note ID {note_id}.")
            return note
        except SQLAlchemyError as e_update_note:
            db.rollback()
            logger.error(f"Error updating trade note ID {note_id}: {e_update_note}", exc_info=True)
            return None
        finally:
            db.close()

    def delete_trade_note(self, note_id: int) -> bool:
        """Deletes a trade note by its ID."""
        db = self.get_db()
        if not db: return False
        
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id).first()
            if not note:
                logger.warning(f"Trade note with ID {note_id} not found for deletion.")
                return False
            
            db.delete(note)
            db.commit()
            logger.info(f"Trade note ID {note_id} deleted successfully.")
            return True
        except SQLAlchemyError as e_delete_note:
            db.rollback()
            logger.error(f"Error deleting trade note ID {note_id}: {e_delete_note}", exc_info=True)
            return False
        finally:
            db.close()

if __name__ == "__main__":
    logger.info("--- Testing DataService with DB Integration (SQLite) ---")
    # This ensures tables are created if the script is run directly (for testing)
    # In the Streamlit app, DataService instantiation (e.g., in app.py or on first use)
    # will trigger create_db_tables().
    if SessionLocal: # Check if SessionLocal was initialized
        create_db_tables() # Explicitly call if running standalone for testing
        
        # Example Usage (for standalone testing):
        # test_data_service = DataService()
        # if test_data_service.get_db(): # Check if db connection is possible
        #     # Add a test note
        #     added_note = test_data_service.add_trade_note(
        #         trade_identifier="TEST_TRADE_001",
        #         note_content="This is a test note for standalone execution.",
        #         tags="test, example"
        #     )
        #     if added_note:
        #         print(f"Added note: {added_note}")

        #         # Get notes
        #         notes = test_data_service.get_trade_notes(trade_identifier="TEST_TRADE_001")
        #         print(f"Retrieved notes: {notes}")

        #         # Update note
        #         if notes:
        #             updated_note = test_data_service.update_trade_note(notes[0].id, new_content="Updated test note content.")
        #             print(f"Updated note: {updated_note}")

        #         # Delete note
        #         # if notes:
        #         #     deleted = test_data_service.delete_trade_note(notes[0].id)
        #         #     print(f"Deletion status for note ID {notes[0].id}: {deleted}")
        # else:
        #     print("Could not establish database connection for standalone test.")
    else:
        print("SessionLocal not initialized. Database tests cannot run.")
    pass
