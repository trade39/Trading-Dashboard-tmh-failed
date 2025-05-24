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
import os
import uuid
import hashlib
from io import BytesIO

try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    # Ensure data_processing is in the root or accessible via PYTHONPATH
    from data_processing import load_and_process_data # Assuming data_processing.py is at the root
    from .database_setup import Base, get_db_session, UserFile, UserFileMapping, TradeNoteDB
except ImportError as e:
    print(f"Warning (data_service.py): Could not import modules: {e}. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_Service"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"} # Minimal fallback
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy import create_engine
    Base = declarative_base() # type: ignore
    engine_fallback_data = create_engine("sqlite:///./temp_data_service_fallback.db") # Fallback DB
    SessionLocal_fallback_data = sessionmaker(autocommit=False, autoflush=False, bind=engine_fallback_data)
    def get_db_session(): return SessionLocal_fallback_data()
    def load_and_process_data(uploaded_file_obj: Any, user_column_mapping: Optional[Dict[str, str]] = None, original_file_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        if uploaded_file_obj:
            try: return pd.read_csv(uploaded_file_obj) # Simplistic fallback
            except: return None
        return None
    # Minimal fallback models
    class UserFile(Base): # type: ignore
        __tablename__ = "user_files_fb_ds_v2"
        id=Column(Integer, primary_key=True); user_id=Column(Integer); original_file_name=Column(String); storage_identifier=Column(String, unique=True); file_size_bytes=Column(Integer); upload_timestamp=Column(DateTime, default=dt.datetime.utcnow); file_hash=Column(String); status=Column(String, default="active")
    class UserFileMapping(Base): # type: ignore
        __tablename__ = "user_file_mappings_fb_ds_v2"
        id=Column(Integer, primary_key=True); user_id=Column(Integer); user_file_id=Column(Integer); conceptual_column=Column(String); csv_column_name=Column(String); mapped_timestamp=Column(DateTime, default=dt.datetime.utcnow)
    class TradeNoteDB(Base): # type: ignore
        __tablename__ = "trade_notes_fb_ds_v2"
        id=Column(Integer, primary_key=True); user_id=Column(Integer); trade_identifier=Column(String); note_timestamp=Column(DateTime, default=dt.datetime.utcnow); note_content=Column(Text); tags=Column(String)
    Base.metadata.create_all(bind=engine_fallback_data)


logger = logging.getLogger(APP_TITLE)

USER_FILES_ROOT_DIR = "user_data_storage"
if not os.path.exists(USER_FILES_ROOT_DIR):
    try:
        os.makedirs(USER_FILES_ROOT_DIR, exist_ok=True)
        logger.info(f"Created root directory for user file storage: {USER_FILES_ROOT_DIR}")
    except OSError as e:
        logger.error(f"Could not create root directory {USER_FILES_ROOT_DIR}: {e}. Local file storage will fail.")

@st.cache_data(ttl=3600)
def get_benchmark_data_static(
    ticker: str,
    start_date_str: str,
    end_date_str: str
) -> Optional[pd.Series]:
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
        uploaded_file_obj: Any,
        user_column_mapping: Optional[Dict[str, str]] = None,
        original_file_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if uploaded_file_obj is None: return None
        file_name_for_log = original_file_name or getattr(uploaded_file_obj, 'name', "In-memory data")
        data_to_process = uploaded_file_obj
        if hasattr(uploaded_file_obj, 'getvalue') and not isinstance(uploaded_file_obj, BytesIO):
            try: data_to_process = BytesIO(uploaded_file_obj.getvalue()); data_to_process.seek(0)
            except Exception as e: self.logger.error(f"Could not get bytes from file obj for '{file_name_for_log}': {e}"); return None
        elif isinstance(uploaded_file_obj, BytesIO): uploaded_file_obj.seek(0)
        try:
            self.logger.info(f"DataService: Processing '{file_name_for_log}' with mapping.")
            processed_df = load_and_process_data(data_to_process, user_column_mapping=user_column_mapping, original_file_name=file_name_for_log)
            if processed_df is not None: self.logger.info(f"DataService: '{file_name_for_log}' processed. Shape: {processed_df.shape}")
            else: self.logger.warning(f"DataService: Processing of '{file_name_for_log}' returned None.")
            return processed_df
        except Exception as e_process: self.logger.error(f"DataService: Error processing '{file_name_for_log}': {e_process}", exc_info=True); return None

    def filter_data(self, df: pd.DataFrame, filters: Dict[str, Any], column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
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

    def add_trade_note(self, user_id: int, trade_identifier: Optional[str], note_content: str, tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        db = self._get_db();
        if not db: return None
        new_note = TradeNoteDB(user_id=user_id, trade_identifier=trade_identifier, note_content=note_content, tags=tags, note_timestamp=dt.datetime.utcnow())
        try:
            db.add(new_note); db.commit(); db.refresh(new_note)
            self.logger.info(f"Added note ID: {new_note.id} for user_id: {user_id}, trade_id: '{trade_identifier}'")
            return new_note
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error adding note for user_id {user_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def get_trade_notes(self, user_id: int, trade_identifier: Optional[str]=None, start_date: Optional[dt.datetime]=None, end_date: Optional[dt.datetime]=None, limit: Optional[int]=None) -> List[TradeNoteDB]:
        db = self._get_db();
        if not db: return []
        try:
            query = db.query(TradeNoteDB).filter(TradeNoteDB.user_id == user_id)
            if trade_identifier: query = query.filter(TradeNoteDB.trade_identifier == trade_identifier)
            if start_date: query = query.filter(TradeNoteDB.note_timestamp >= start_date)
            if end_date: query = query.filter(TradeNoteDB.note_timestamp <= (end_date + dt.timedelta(days=1) - dt.timedelta(microseconds=1)))
            query = query.order_by(TradeNoteDB.note_timestamp.desc())
            if limit: query = query.limit(limit)
            return query.all()
        except SQLAlchemyError as e: self.logger.error(f"Error getting notes for user_id {user_id}: {e}", exc_info=True); return []
        finally:
            if db: db.close()

    def update_trade_note(self, user_id: int, note_id: int, new_content: Optional[str]=None, new_tags: Optional[str]=None) -> Optional[TradeNoteDB]:
        db = self._get_db();
        if not db: return None
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id, TradeNoteDB.user_id == user_id).first()
            if not note: self.logger.warning(f"Note ID {note_id} not found for user_id {user_id} during update."); return None
            updated = False
            if new_content is not None: note.note_content = new_content; updated = True
            if new_tags is not None: note.tags = new_tags; updated = True
            if updated: note.note_timestamp = dt.datetime.utcnow(); db.commit(); db.refresh(note)
            return note
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error updating note {note_id} for user_id {user_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def delete_trade_note(self, user_id: int, note_id: int) -> bool:
        db = self._get_db();
        if not db: return False
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id, TradeNoteDB.user_id == user_id).first()
            if not note: self.logger.warning(f"Note ID {note_id} not found for user_id {user_id} for deletion."); return False
            db.delete(note); db.commit()
            self.logger.info(f"Note ID {note_id} deleted successfully for user_id {user_id}.")
            return True
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error deleting note {note_id} for user_id {user_id}: {e}", exc_info=True); return False
        finally:
            if db: db.close()

    def save_user_file(self, user_id: int, uploaded_file: Any) -> Optional[UserFile]:
        if not uploaded_file: self.logger.warning("save_user_file: No uploaded_file object provided."); return None
        db = self._get_db();
        if not db: self.logger.error("save_user_file: Database session not available."); return None
        original_filename = uploaded_file.name; file_content_bytes = uploaded_file.getvalue()
        file_size = len(file_content_bytes); file_hash = hashlib.sha256(file_content_bytes).hexdigest()
        existing_file = db.query(UserFile).filter_by(user_id=user_id, file_hash=file_hash, status="active").first()
        if existing_file:
            self.logger.info(f"User {user_id} re-uploaded duplicate file '{original_filename}'. Returning existing ID {existing_file.id}.")
            db.close(); return existing_file
        user_dir = os.path.join(USER_FILES_ROOT_DIR, str(user_id))
        try: os.makedirs(user_dir, exist_ok=True)
        except OSError as e: self.logger.error(f"Could not create dir {user_dir}: {e}. Cannot save file.", exc_info=True); db.close(); return None
        unique_suffix = uuid.uuid4().hex[:8]
        safe_basename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in os.path.splitext(original_filename)[0])
        extension = os.path.splitext(original_filename)[1]
        storage_filename = f"{safe_basename}_{unique_suffix}{extension}"
        storage_identifier = os.path.join(str(user_id), storage_filename) # Path relative to USER_FILES_ROOT_DIR
        full_storage_path = os.path.join(USER_FILES_ROOT_DIR, storage_identifier)
        try:
            with open(full_storage_path, "wb") as f: f.write(file_content_bytes)
            self.logger.info(f"File '{original_filename}' saved to '{full_storage_path}' for user {user_id}.")
        except IOError as e: self.logger.error(f"IOError saving '{original_filename}' to '{full_storage_path}': {e}", exc_info=True); db.close(); return None
        new_user_file = UserFile(user_id=user_id, original_file_name=original_filename, storage_identifier=storage_identifier, file_size_bytes=file_size, file_hash=file_hash, status="active")
        try:
            db.add(new_user_file); db.commit(); db.refresh(new_user_file)
            self.logger.info(f"UserFile record created for '{original_filename}', ID: {new_user_file.id}")
            return new_user_file
        except SQLAlchemyError as e:
            db.rollback(); self.logger.error(f"DB error saving UserFile for '{original_filename}': {e}", exc_info=True)
            if os.path.exists(full_storage_path):
                try: os.remove(full_storage_path)
                except OSError as e_cleanup: self.logger.error(f"Error cleaning up '{full_storage_path}': {e_cleanup}")
            return None
        finally:
            if db: db.close()

    def list_user_files(self, user_id: int) -> List[UserFile]:
        db = self._get_db();
        if not db: return []
        try:
            files = db.query(UserFile).filter(UserFile.user_id == user_id, UserFile.status == "active").order_by(UserFile.upload_timestamp.desc()).all()
            self.logger.info(f"Retrieved {len(files)} active files for user_id {user_id}.")
            return files
        except SQLAlchemyError as e: self.logger.error(f"Error listing files for user_id {user_id}: {e}", exc_info=True); return []
        finally:
            if db: db.close()

    def get_user_file_record_by_id(self, user_file_id: int, user_id: int) -> Optional[UserFile]:
        """Retrieves a single UserFile record by its ID, ensuring it belongs to the user."""
        db = self._get_db()
        if not db:
            self.logger.error("get_user_file_record_by_id: Database session not available.")
            return None
        try:
            user_file_rec = db.query(UserFile).filter(
                UserFile.id == user_file_id,
                UserFile.user_id == user_id,
                UserFile.status == "active"
            ).first()
            if not user_file_rec:
                self.logger.warning(f"UserFile record ID {user_file_id} not found or not active for user_id {user_id}.")
            return user_file_rec
        except SQLAlchemyError as e:
            self.logger.error(f"Database error retrieving UserFile record ID {user_file_id} for user_id {user_id}: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def get_user_file_content(self, user_file_id: int, user_id: int) -> Optional[BytesIO]:
        db = self._get_db();
        if not db: return None
        try:
            user_file_rec = db.query(UserFile).filter(UserFile.id == user_file_id, UserFile.user_id == user_id, UserFile.status == "active").first()
            if not user_file_rec: self.logger.warning(f"File ID {user_file_id} not found or not active for user {user_id}."); return None
            full_path = os.path.join(USER_FILES_ROOT_DIR, user_file_rec.storage_identifier)
            if not os.path.exists(full_path): self.logger.error(f"File content not found at {full_path} for UserFile ID {user_file_id}."); return None
            with open(full_path, "rb") as f: content = BytesIO(f.read())
            content.seek(0); self.logger.info(f"Retrieved content for file ID {user_file_id}."); return content
        except IOError as e: self.logger.error(f"IOError reading file content for UserFile ID {user_file_id}: {e}", exc_info=True); return None
        except SQLAlchemyError as e: self.logger.error(f"DB error retrieving UserFile {user_file_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def delete_user_file(self, user_file_id: int, user_id: int, permanent_delete_local_file: bool = False) -> bool:
        db = self._get_db();
        if not db: return False
        try:
            user_file_rec = db.query(UserFile).filter(UserFile.id == user_file_id, UserFile.user_id == user_id).first()
            if not user_file_rec: self.logger.warning(f"File ID {user_file_id} not found for user {user_id} for deletion."); return False
            original_path = user_file_rec.storage_identifier
            user_file_rec.status = "deleted"; db.commit()
            self.logger.info(f"File ID {user_file_rec.id} marked as deleted for user {user_id}.")
            if permanent_delete_local_file:
                full_path = os.path.join(USER_FILES_ROOT_DIR, original_path)
                if os.path.exists(full_path):
                    try: os.remove(full_path); self.logger.info(f"Permanently deleted local file: {full_path}")
                    except OSError as e: self.logger.error(f"Error permanently deleting local file '{full_path}': {e}", exc_info=True)
            return True
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"DB error deleting UserFile {user_file_id}: {e}", exc_info=True); return False
        finally:
            if db: db.close()

    def save_user_column_mapping(self, user_id: int, user_file_id: int, mapping: Dict[str, str]) -> bool:
        db = self._get_db()
        if not db: return False
        try:
            db.query(UserFileMapping).filter_by(user_id=user_id, user_file_id=user_file_id).delete(synchronize_session=False)
            new_mappings_to_add = []
            for conceptual_col, csv_col_name in mapping.items():
                if csv_col_name: # Only save if a CSV column is actually selected
                    new_mappings_to_add.append(UserFileMapping(user_id=user_id, user_file_id=user_file_id, conceptual_column=conceptual_col, csv_column_name=csv_col_name))
            if new_mappings_to_add: db.add_all(new_mappings_to_add)
            db.commit()
            self.logger.info(f"Saved/Updated {len(new_mappings_to_add)} column mappings for user_id {user_id}, file_id {user_file_id}.")
            return True
        except SQLAlchemyError as e:
            db.rollback(); self.logger.error(f"Error saving column mapping for user_id {user_id}, file_id {user_file_id}: {e}", exc_info=True); return False
        finally:
            if db: db.close()

    def get_user_column_mapping(self, user_id: int, user_file_id: int) -> Optional[Dict[str, str]]:
        db = self._get_db()
        if not db: return None
        try:
            mappings_db = db.query(UserFileMapping).filter_by(user_id=user_id, user_file_id=user_file_id).all()
            if not mappings_db: self.logger.info(f"No saved mapping for user_id {user_id}, file_id {user_file_id}."); return None
            loaded_mapping = {item.conceptual_column: item.csv_column_name for item in mappings_db}
            self.logger.info(f"Retrieved mapping for user_id {user_id}, file_id {user_file_id}: {len(loaded_mapping)} items.")
            return loaded_mapping
        except SQLAlchemyError as e:
            self.logger.error(f"Error retrieving mapping for user_id {user_id}, file_id {user_file_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()
