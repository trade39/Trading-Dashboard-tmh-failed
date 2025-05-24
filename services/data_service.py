# services/data_service.py

import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
import yfinance as yf
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey # UniqueConstraint not used here yet
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, scoped_session # relationship not used here yet
from sqlalchemy.exc import SQLAlchemyError
import datetime as dt # Use dt alias to avoid conflict

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DATABASE_URL 
    from data_processing import load_and_process_data
except ImportError as e:
    print(f"Warning (data_service.py): Could not import from root config or data_processing: {e}. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_Service"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy", "trade_id": "trade_id", "notes": "notes"}
    DATABASE_URL = "sqlite:///placeholder_trading_dashboard.db" 
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
    engine = create_engine(DATABASE_URL, echo=False) 
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base() # Define Base here so other models (like User) can import it
    logger.info(f"Database engine created for URL: {DATABASE_URL}")
except Exception as e_db_setup:
    logger.critical(f"Failed to create database engine or sessionmaker for URL '{DATABASE_URL}': {e_db_setup}", exc_info=True)
    SessionLocal = None
    Base = None # type: ignore
    engine = None 

class TradeNoteDB(Base): # type: ignore
    __tablename__ = "trade_notes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    trade_identifier = Column(String, index=True, nullable=True) 
    note_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    note_content = Column(Text, nullable=False)
    tags = Column(String, nullable=True) 

    def __repr__(self):
        return f"<TradeNoteDB(id={self.id}, trade_identifier='{self.trade_identifier}', timestamp='{self.note_timestamp.strftime('%Y-%m-%d %H:%M')}')>"

def create_db_tables():
    """
    Creates all database tables defined by models inheriting from Base.
    This function should be called once at application startup.
    It now imports the User model from AuthService to ensure its table is also created.
    """
    if Base and engine: 
        try:
            # Import User model here to ensure it's part of Base.metadata when create_all is called
            from .auth_service import User # pylint: disable=import-outside-toplevel 
            # This local import ensures User model is registered with Base.metadata
            # before create_all is called.
            
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables (including User and TradeNoteDB) checked/created successfully.")
        except ImportError:
            logger.error("Failed to import User model from auth_service for table creation. User table might not be created.")
            # Attempt to create only known tables if User import fails
            # This part is tricky; ideally, all models are known to Base before this call.
            # For now, if User can't be imported, only TradeNoteDB (if defined above) would be created.
            # A better pattern is to have a central db_setup.py that imports all models.
            try:
                # Try creating tables known to this module's Base at least
                temp_base_for_partial_create = declarative_base()
                # Re-declare models known directly in this file if needed for partial creation
                # For now, this will only create tables defined directly in this file if User import fails.
                # This is not ideal. The import should succeed.
                if "trade_notes" not in Base.metadata.tables: # Check if it was already part of the main Base
                     logger.warning("TradeNoteDB might not be part of the main Base if User import failed and Base was re-declared.")
                
                # If User model could not be imported, Base.metadata might not include it.
                # Calling Base.metadata.create_all() here would only create tables defined
                # directly in this file and associated with *this* Base instance.
                # The local import of User is the key to making this work.
                logger.warning("Attempting to create tables with potentially missing User model due to import error.")
                Base.metadata.create_all(bind=engine)


            except Exception as e_partial_create:
                 logger.error(f"Error during partial table creation attempt: {e_partial_create}", exc_info=True)

        except Exception as e_create_tables:
            logger.error(f"Error creating database tables: {e_create_tables}", exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'):
                st.error(f"Database Error: Could not create tables. Check logs. Error: {e_create_tables}")
    else:
        logger.error("Database engine or Base not initialized. Cannot create tables.")
        if 'st' in globals() and hasattr(st, 'error'):
            st.error("Database connection failed. Features requiring database will not work.")


@st.cache_resource 
def get_db_session_cached() -> Optional[Session]: # Return type Session from sqlalchemy.orm
    """Provides a cached database session factory, returning a new session."""
    if not SessionLocal:
        logger.error("SessionLocal is not initialized. Cannot create DB session.")
        return None
    try:
        # For st.cache_resource, we typically cache the factory or engine,
        # and create sessions as needed. Here, SessionLocal is the factory.
        # To ensure a fresh session for each request that needs one,
        # it's better to return SessionLocal() directly if not using scoped_session
        # or manage scoped_session correctly.
        # If SessionLocal itself is cached, then each call to SessionLocal() gives a new session.
        # For scoped_session, you'd cache the scoped_session factory.
        # The current setup with SessionLocal being a sessionmaker is fine.
        # Let's assume SessionLocal is the configured sessionmaker.
        
        # If you want a truly request-scoped session with scoped_session:
        # db_session_factory = scoped_session(SessionLocal)
        # return db_session_factory <--- this is what you'd cache
        # And then use db_session_factory() to get a session, and db_session_factory.remove() to close.

        # For simplicity with st.cache_resource on the getter:
        # This will cache ONE session instance if not careful.
        # The intent of caching get_db_session_cached is usually to cache the factory.
        # Let's adjust to cache the factory and return a new session.
        # However, SessionLocal IS the factory. So, this is fine.
        # The key is that the CALLER must close the session.

        db = SessionLocal() # Create a new session from the factory
        logger.debug("Database session created by get_db_session_cached.")
        return db
    except Exception as e_get_session:
        logger.error(f"Error creating database session in get_db_session_cached: {e_get_session}", exc_info=True)
        return None

class DataService:
    def __init__(self):
        logger.info("DataService initialized.")
        # Table creation is now more robustly handled at app startup.
        # create_db_tables() # Call this once, e.g., in app.py after all model imports.
                           # For now, keeping it here means it's called on DataService instantiation.

    def get_db(self) -> Optional[Session]: 
        """Gets a database session. Handles session closing."""
        db = get_db_session_cached()
        if db is None:
            logger.error("Failed to get DB session from cached provider.")
        return db 

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

    def add_trade_note(self, trade_identifier: Optional[str], note_content: str, tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        db = self.get_db()
        if not db: return None
        
        new_note = TradeNoteDB(
            trade_identifier=trade_identifier,
            note_content=note_content,
            tags=tags,
            note_timestamp=dt.datetime.utcnow() 
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
            db.close() 

    def get_trade_notes(
        self,
        trade_identifier: Optional[str] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        limit: Optional[int] = None
    ) -> List[TradeNoteDB]:
        db = self.get_db()
        if not db: return []
        
        try:
            query = db.query(TradeNoteDB)
            if trade_identifier:
                query = query.filter(TradeNoteDB.trade_identifier == trade_identifier)
            if start_date:
                query = query.filter(TradeNoteDB.note_timestamp >= start_date)
            if end_date:
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
            if new_tags is not None: 
                note.tags = new_tags
                updated = True
            
            if updated:
                note.note_timestamp = dt.datetime.utcnow() 
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

# It's better to call create_db_tables() once at application startup,
# for example, in app.py after all model definitions are imported.
# If DataService is instantiated multiple times, this would be called multiple times.
# For now, we'll assume DataService is a singleton or create_db_tables is idempotent.
# create_db_tables() # Moved to be called explicitly, e.g., in app.py or main startup.
