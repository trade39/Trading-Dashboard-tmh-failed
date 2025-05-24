# services/data_service.py

import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
import yfinance as yf
import logging
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import datetime as dt
import os # For file path operations
import uuid # For generating unique filenames
import hashlib # For file hashing
from io import BytesIO # For handling file content

try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from data_processing import load_and_process_data
    from .database_setup import Base, get_db_session
    from .database_setup import UserFile # Import UserFile model
except ImportError as e:
    print(f"Warning (data_service.py): Could not import modules: {e}. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_Service"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy", "trade_id": "trade_id", "notes": "notes"}
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy import create_engine
    Base = declarative_base() # type: ignore
    engine_fallback_data = create_engine("sqlite:///./temp_data_service_test.db")
    SessionLocal_fallback_data = sessionmaker(autocommit=False, autoflush=False, bind=engine_fallback_data)
    def get_db_session(): return SessionLocal_fallback_data()
    def load_and_process_data(uploaded_file_obj: Any, user_column_mapping: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
        if uploaded_file_obj:
            try: return pd.read_csv(uploaded_file_obj)
            except: return None
        return None
    class UserFile: pass # Placeholder

logger = logging.getLogger(APP_TITLE)

# Define the root directory for storing user files locally
# IMPORTANT: For Streamlit Cloud, this path is ephemeral.
# For persistent storage on Streamlit Cloud, use cloud storage (S3, GCS, Azure Blob).
USER_FILES_ROOT_DIR = "user_data_storage" 
if not os.path.exists(USER_FILES_ROOT_DIR):
    try:
        os.makedirs(USER_FILES_ROOT_DIR, exist_ok=True)
        logger.info(f"Created root directory for user file storage: {USER_FILES_ROOT_DIR}")
    except OSError as e:
        logger.error(f"Could not create root directory {USER_FILES_ROOT_DIR}: {e}. Local file storage will fail.")
        # In a real app, you might want to raise an exception or handle this more gracefully.

class TradeNoteDB(Base): # type: ignore
    __tablename__ = "trade_notes"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    trade_identifier = Column(String, index=True, nullable=True)
    note_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    note_content = Column(Text, nullable=False)
    tags = Column(String, nullable=True)
    # user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Will be added later

    def __repr__(self):
        return f"<TradeNoteDB(id={self.id}, trade_identifier='{self.trade_identifier}')>"

@st.cache_data(ttl=3600)
def get_benchmark_data_static(
    ticker: str,
    start_date_str: str,
    end_date_str: str
) -> Optional[pd.Series]:
    # ... (implementation as before) ...
    logger_static_func = logging.getLogger(f"{APP_TITLE}.get_benchmark_data_static")
    logger_static_func.info(f"Executing get_benchmark_data_static for {ticker} from {start_date_str} to {end_date_str}")
    if not ticker: return None
    try:
        start_dt, end_dt = pd.to_datetime(start_date_str), pd.to_datetime(end_date_str)
        if start_dt > end_dt: return None
        fetch_end_dt = end_dt + pd.Timedelta(days=1)
        if start_dt == end_dt: fetch_end_dt = end_dt + pd.Timedelta(days=2)
        data = yf.download(ticker, start=start_dt, end=fetch_end_dt, progress=False, auto_adjust=True, actions=False)
        if data.empty or 'Close' not in data.columns: return None
        daily_adj_close = data['Close'].dropna()
        if len(daily_adj_close) < 2: return None
        daily_returns = daily_adj_close.pct_change().dropna()
        if daily_returns.empty: return None
        daily_returns.name = f"{ticker}_returns"; return daily_returns
    except Exception as e: logger_static_func.error(f"Error fetching benchmark {ticker}: {e}", exc_info=True); return None


class DataService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        self.logger.info("DataService initialized.")

    def _get_db(self) -> Optional[Session]:
        return get_db_session()

    def get_processed_trading_data(
        self,
        uploaded_file_obj: Any, # Can be BytesIO or Streamlit UploadedFile
        user_column_mapping: Optional[Dict[str, str]] = None,
        original_file_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if uploaded_file_obj is None: return None
        file_name_for_log = original_file_name or getattr(uploaded_file_obj, 'name', "In-memory data")
        
        # Ensure uploaded_file_obj is BytesIO for load_and_process_data if it's Streamlit's UploadedFile
        data_to_process = uploaded_file_obj
        if hasattr(uploaded_file_obj, 'getvalue') and not isinstance(uploaded_file_obj, BytesIO):
            try:
                data_to_process = BytesIO(uploaded_file_obj.getvalue())
                data_to_process.seek(0)
            except Exception as e:
                self.logger.error(f"Could not get bytes from uploaded file object for '{file_name_for_log}': {e}")
                return None
        elif isinstance(uploaded_file_obj, BytesIO):
            uploaded_file_obj.seek(0) # Ensure pointer is at the beginning

        try:
            self.logger.info(f"DataService: Processing data source: '{file_name_for_log}' with user mapping.")
            processed_df = load_and_process_data(data_to_process, user_column_mapping=user_column_mapping)
            if processed_df is not None:
                self.logger.info(f"DataService: Data source '{file_name_for_log}' processed. Shape: {processed_df.shape}")
            else:
                self.logger.warning(f"DataService: Processing of data source '{file_name_for_log}' returned None.")
            return processed_df
        except Exception as e_process:
            self.logger.error(f"DataService: Error processing '{file_name_for_log}': {e_process}", exc_info=True)
            return None

    def filter_data(
        self,
        df: pd.DataFrame,
        filters: Dict[str, Any],
        column_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        # ... (implementation as before) ...
        if df is None or df.empty: return pd.DataFrame()
        effective_column_map = column_map if column_map is not None else EXPECTED_COLUMNS
        filtered_df = df.copy()
        date_col_conceptual = effective_column_map.get('date', 'date')
        date_range = filters.get('selected_date_range')
        if date_col_conceptual in filtered_df.columns and date_range and len(date_range) == 2:
            try:
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                filtered_df = filtered_df[(pd.to_datetime(filtered_df[date_col_conceptual]).dt.date >= start_date.date()) & (pd.to_datetime(filtered_df[date_col_conceptual]).dt.date <= end_date.date())]
            except Exception as e: self.logger.error(f"Error date filter: {e}", exc_info=True)
        # ... (other filters as before) ...
        symbol_col_conceptual = effective_column_map.get('symbol', 'symbol')
        selected_symbol = filters.get('selected_symbol')
        if symbol_col_conceptual in filtered_df.columns and selected_symbol and selected_symbol != "All":
            try: filtered_df = filtered_df[filtered_df[symbol_col_conceptual].astype(str) == str(selected_symbol)]
            except Exception as e: self.logger.error(f"Error symbol filter: {e}", exc_info=True)
        strategy_col_conceptual = effective_column_map.get('strategy', 'strategy')
        selected_strategy = filters.get('selected_strategy')
        if strategy_col_conceptual in filtered_df.columns and selected_strategy and selected_strategy != "All":
            try: filtered_df = filtered_df[filtered_df[strategy_col_conceptual].astype(str) == str(selected_strategy)]
            except Exception as e: self.logger.error(f"Error strategy filter: {e}", exc_info=True)
        return filtered_df

    # --- Trade Note CRUD Methods (as before) ---
    def add_trade_note(self, trade_identifier: Optional[str], note_content: str, tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        # ... (implementation as before) ...
        db = self._get_db()
        if not db: return None
        new_note = TradeNoteDB(trade_identifier=trade_identifier, note_content=note_content, tags=tags, note_timestamp=dt.datetime.utcnow())
        try:
            db.add(new_note); db.commit(); db.refresh(new_note)
            self.logger.info(f"Added note ID: {new_note.id} for trade_id: '{trade_identifier}'")
            return new_note
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error adding note: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def get_trade_notes(self, trade_identifier: Optional[str]=None, start_date: Optional[dt.datetime]=None, end_date: Optional[dt.datetime]=None, limit: Optional[int]=None) -> List[TradeNoteDB]:
        # ... (implementation as before) ...
        db = self._get_db()
        if not db: return []
        try:
            query = db.query(TradeNoteDB)
            if trade_identifier: query = query.filter(TradeNoteDB.trade_identifier == trade_identifier)
            if start_date: query = query.filter(TradeNoteDB.note_timestamp >= start_date)
            if end_date: query = query.filter(TradeNoteDB.note_timestamp <= (end_date + dt.timedelta(days=1) - dt.timedelta(microseconds=1)))
            query = query.order_by(TradeNoteDB.note_timestamp.desc())
            if limit: query = query.limit(limit)
            return query.all()
        except SQLAlchemyError as e: self.logger.error(f"Error getting notes: {e}", exc_info=True); return []
        finally:
            if db: db.close()
            
    def update_trade_note(self, note_id: int, new_content: Optional[str]=None, new_tags: Optional[str]=None) -> Optional[TradeNoteDB]:
        # ... (implementation as before) ...
        db = self._get_db()
        if not db: return None
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id).first()
            if not note: return None
            updated = False
            if new_content is not None: note.note_content = new_content; updated = True
            if new_tags is not None: note.tags = new_tags; updated = True
            if updated: note.note_timestamp = dt.datetime.utcnow(); db.commit(); db.refresh(note)
            return note
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error updating note {note_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def delete_trade_note(self, note_id: int) -> bool:
        # ... (implementation as before) ...
        db = self._get_db()
        if not db: return False
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id).first()
            if not note: return False
            db.delete(note); db.commit(); return True
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error deleting note {note_id}: {e}", exc_info=True); return False
        finally:
            if db: db.close()

    # --- UserFile Handling Methods (New for Phase 2) ---
    def save_user_file(self, user_id: int, uploaded_file: Any) -> Optional[UserFile]:
        """
        Saves an uploaded file for a user to the local file system and records its metadata in the database.
        Args:
            user_id (int): The ID of the user uploading the file.
            uploaded_file (streamlit.UploadedFile): The file object from st.file_uploader.
        Returns:
            Optional[UserFile]: The UserFile database object if successful, else None.
        """
        if not uploaded_file:
            self.logger.warning("save_user_file: No uploaded_file object provided.")
            return None
        
        db = self._get_db()
        if not db:
            self.logger.error("save_user_file: Database session not available.")
            return None

        original_filename = uploaded_file.name
        file_content_bytes = uploaded_file.getvalue() # Read content once
        file_size = len(file_content_bytes)
        
        # Generate file hash
        try:
            file_hash = hashlib.sha256(file_content_bytes).hexdigest()
        except Exception as e_hash:
            self.logger.error(f"Error generating hash for file '{original_filename}': {e_hash}", exc_info=True)
            file_hash = None # Proceed without hash if error occurs

        # Check for existing file with same hash for the user (optional: prevent exact duplicates)
        if file_hash:
            existing_file = db.query(UserFile).filter_by(user_id=user_id, file_hash=file_hash, status="active").first()
            if existing_file:
                self.logger.info(f"User {user_id} attempted to upload duplicate file '{original_filename}' (hash: {file_hash}). Returning existing record ID {existing_file.id}.")
                db.close()
                return existing_file # Or raise an error/return specific code

        # Create user-specific directory if it doesn't exist
        user_dir = os.path.join(USER_FILES_ROOT_DIR, str(user_id))
        try:
            os.makedirs(user_dir, exist_ok=True)
        except OSError as e_mkdir:
            self.logger.error(f"Could not create directory {user_dir}: {e_mkdir}. Cannot save file.", exc_info=True)
            db.close()
            return None

        # Generate a unique filename for storage to avoid conflicts
        unique_suffix = uuid.uuid4().hex[:8]
        # Sanitize original filename for safe use in path (basic sanitization)
        safe_original_basename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in os.path.splitext(original_filename)[0])
        extension = os.path.splitext(original_filename)[1]
        storage_filename = f"{safe_original_basename}_{unique_suffix}{extension}"
        storage_identifier = os.path.join(str(user_id), storage_filename) # Relative path from USER_FILES_ROOT_DIR
        full_storage_path = os.path.join(USER_FILES_ROOT_DIR, storage_identifier)

        try:
            with open(full_storage_path, "wb") as f:
                f.write(file_content_bytes)
            self.logger.info(f"File '{original_filename}' saved locally for user {user_id} at '{full_storage_path}'.")
        except IOError as e_io:
            self.logger.error(f"IOError saving file '{original_filename}' to '{full_storage_path}': {e_io}", exc_info=True)
            db.close()
            return None

        new_user_file = UserFile(
            user_id=user_id,
            original_file_name=original_filename,
            storage_identifier=storage_identifier, # Store relative path
            file_size_bytes=file_size,
            upload_timestamp=dt.datetime.utcnow(),
            file_hash=file_hash,
            status="active"
        )
        try:
            db.add(new_user_file)
            db.commit()
            db.refresh(new_user_file)
            self.logger.info(f"UserFile record created for '{original_filename}', ID: {new_user_file.id}")
            return new_user_file
        except SQLAlchemyError as e_db_save:
            db.rollback()
            self.logger.error(f"Database error saving UserFile record for '{original_filename}': {e_db_save}", exc_info=True)
            # Attempt to clean up the saved file if DB record fails
            if os.path.exists(full_storage_path):
                try: os.remove(full_storage_path)
                except OSError as e_cleanup: self.logger.error(f"Error cleaning up file '{full_storage_path}' after DB error: {e_cleanup}")
            return None
        finally:
            if db: db.close()

    def list_user_files(self, user_id: int) -> List[UserFile]:
        """Lists active files for a given user, most recent first."""
        db = self._get_db()
        if not db: return []
        try:
            files = db.query(UserFile)\
                .filter(UserFile.user_id == user_id, UserFile.status == "active")\
                .order_by(UserFile.upload_timestamp.desc())\
                .all()
            self.logger.info(f"Retrieved {len(files)} active files for user_id {user_id}.")
            return files
        except SQLAlchemyError as e:
            self.logger.error(f"Error listing files for user_id {user_id}: {e}", exc_info=True)
            return []
        finally:
            if db: db.close()

    def get_user_file_content(self, user_file_id: int, user_id: int) -> Optional[BytesIO]:
        """
        Retrieves the content of a specific file for a user, verifying ownership.
        Returns content as BytesIO.
        """
        db = self._get_db()
        if not db: return None
        try:
            user_file_record = db.query(UserFile)\
                .filter(UserFile.id == user_file_id, UserFile.user_id == user_id, UserFile.status == "active")\
                .first()
            
            if not user_file_record:
                self.logger.warning(f"File ID {user_file_id} not found for user {user_id} or not active.")
                return None

            full_storage_path = os.path.join(USER_FILES_ROOT_DIR, user_file_record.storage_identifier)
            if not os.path.exists(full_storage_path):
                self.logger.error(f"File content not found at path: {full_storage_path} for UserFile ID {user_file_id}. DB record might be stale.")
                # Optionally, mark the DB record as 'missing' or 'error' status here
                return None
            
            with open(full_storage_path, "rb") as f:
                file_content = BytesIO(f.read())
            file_content.seek(0) # Reset pointer for a new read
            self.logger.info(f"Retrieved content for file ID {user_file_id}, name: '{user_file_record.original_file_name}'.")
            return file_content
            
        except IOError as e_io:
            self.logger.error(f"IOError reading file content for UserFile ID {user_file_id}: {e_io}", exc_info=True)
            return None
        except SQLAlchemyError as e_db:
            self.logger.error(f"Database error retrieving UserFile record {user_file_id}: {e_db}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def delete_user_file(self, user_file_id: int, user_id: int, permanent_delete_local_file: bool = False) -> bool:
        """
        Marks a user's file as 'deleted' (soft delete).
        Optionally, permanently deletes the local file.
        """
        db = self._get_db()
        if not db: return False
        try:
            user_file_record = db.query(UserFile)\
                .filter(UserFile.id == user_file_id, UserFile.user_id == user_id)\
                .first()

            if not user_file_record:
                self.logger.warning(f"File ID {user_file_id} not found for user {user_id} for deletion.")
                return False
            
            original_storage_path = user_file_record.storage_identifier # For local deletion
            user_file_record.status = "deleted"
            db.commit()
            self.logger.info(f"File ID {user_file_id} (name: '{user_file_record.original_file_name}') marked as deleted for user {user_id}.")

            if permanent_delete_local_file:
                full_storage_path = os.path.join(USER_FILES_ROOT_DIR, original_storage_path)
                if os.path.exists(full_storage_path):
                    try:
                        os.remove(full_storage_path)
                        self.logger.info(f"Permanently deleted local file: {full_storage_path}")
                    except OSError as e_remove:
                        self.logger.error(f"Error permanently deleting local file '{full_storage_path}': {e_remove}", exc_info=True)
                        # The DB record is still marked 'deleted'. Manual cleanup might be needed.
            return True
        except SQLAlchemyError as e_db:
            db.rollback()
            self.logger.error(f"Database error deleting UserFile record {user_file_id}: {e_db}", exc_info=True)
            return False
        finally:
            if db: db.close()
