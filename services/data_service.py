# services/data_service.py

import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict, List
import yfinance as yf
import logging
from sqlalchemy import Column, Integer, String, Float, DateTime, Text # Keep for model definition
from sqlalchemy.orm import Session # Keep for type hinting
from sqlalchemy.exc import SQLAlchemyError
import datetime as dt

try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from data_processing import load_and_process_data
    from .database_setup import Base, get_db_session # <<< IMPORT FROM database_setup
except ImportError as e:
    print(f"Warning (data_service.py): Could not import from root config, data_processing, or database_setup: {e}. Using placeholders.")
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

logger = logging.getLogger(APP_TITLE)

class TradeNoteDB(Base): # type: ignore
    __tablename__ = "trade_notes"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    trade_identifier = Column(String, index=True, nullable=True)
    note_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    note_content = Column(Text, nullable=False)
    tags = Column(String, nullable=True)

    def __repr__(self):
        return f"<TradeNoteDB(id={self.id}, trade_identifier='{self.trade_identifier}')>"

@st.cache_data(ttl=3600)
def get_benchmark_data_static(
    ticker: str,
    start_date_str: str,
    end_date_str: str
) -> Optional[pd.Series]:
    logger_static_func = logging.getLogger(f"{APP_TITLE}.get_benchmark_data_static")
    logger_static_func.info(f"Executing get_benchmark_data_static for {ticker} from {start_date_str} to {end_date_str}")
    if not ticker:
        logger_static_func.info("No benchmark ticker provided.")
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
            logger_static_func.warning(f"No data or 'Close' column not found for benchmark {ticker} in range {start_date_str}-{end_date_str}.")
            return None
        daily_adj_close = data['Close'].dropna()
        if len(daily_adj_close) < 2:
            logger_static_func.warning(f"Not enough benchmark data points ({len(daily_adj_close)}) for {ticker} to calculate returns.")
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
        file_name_for_log = original_file_name or getattr(uploaded_file_obj, 'name', "In-memory CSV data")
        try:
            self.logger.info(f"DataService: Processing file: '{file_name_for_log}' with user mapping.")
            processed_df = load_and_process_data(uploaded_file_obj, user_column_mapping=user_column_mapping)
            if processed_df is not None:
                self.logger.info(f"DataService: File '{file_name_for_log}' processed. Shape: {processed_df.shape}")
            else:
                self.logger.warning(f"DataService: Processing of file '{file_name_for_log}' returned None.")
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
        if df is None or df.empty: return pd.DataFrame()
        # ... (rest of filter_data implementation as before) ...
        effective_column_map = column_map if column_map is not None else EXPECTED_COLUMNS
        filtered_df = df.copy()
        self.logger.info(f"DataService: Applying filters. Initial shape: {filtered_df.shape}. Filters: {filters}")
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
                self.logger.error(f"Error applying date filter on '{date_col_conceptual}': {e_date_filter}", exc_info=True)

        symbol_col_conceptual = effective_column_map.get('symbol', 'symbol')
        selected_symbol = filters.get('selected_symbol')
        if symbol_col_conceptual in filtered_df.columns and selected_symbol and selected_symbol != "All":
            try:
                filtered_df = filtered_df[filtered_df[symbol_col_conceptual].astype(str) == str(selected_symbol)]
            except Exception as e_symbol_filter:
                self.logger.error(f"Error applying symbol filter on '{symbol_col_conceptual}': {e_symbol_filter}", exc_info=True)

        strategy_col_conceptual = effective_column_map.get('strategy', 'strategy')
        selected_strategy = filters.get('selected_strategy')
        if strategy_col_conceptual in filtered_df.columns and selected_strategy and selected_strategy != "All":
            try:
                filtered_df = filtered_df[filtered_df[strategy_col_conceptual].astype(str) == str(selected_strategy)]
            except Exception as e_strat_filter:
                self.logger.error(f"Error applying strategy filter on '{strategy_col_conceptual}': {e_strat_filter}", exc_info=True)
        self.logger.info(f"DataService: Filtering complete. Final shape: {filtered_df.shape}")
        return filtered_df

    def add_trade_note(self, trade_identifier: Optional[str], note_content: str, tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        db = self._get_db()
        if not db: return None
        new_note = TradeNoteDB(trade_identifier=trade_identifier, note_content=note_content, tags=tags, note_timestamp=dt.datetime.utcnow())
        try:
            db.add(new_note); db.commit(); db.refresh(new_note)
            self.logger.info(f"Added new trade note ID: {new_note.id} for trade_identifier: '{trade_identifier}'")
            return new_note
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error adding trade note: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def get_trade_notes(self, trade_identifier: Optional[str] = None, start_date: Optional[dt.datetime] = None, end_date: Optional[dt.datetime] = None, limit: Optional[int] = None) -> List[TradeNoteDB]:
        db = self._get_db()
        if not db: return []
        try:
            query = db.query(TradeNoteDB)
            if trade_identifier: query = query.filter(TradeNoteDB.trade_identifier == trade_identifier)
            if start_date: query = query.filter(TradeNoteDB.note_timestamp >= start_date)
            if end_date:
                end_date_inclusive = end_date + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
                query = query.filter(TradeNoteDB.note_timestamp <= end_date_inclusive)
            query = query.order_by(TradeNoteDB.note_timestamp.desc())
            if limit: query = query.limit(limit)
            notes = query.all()
            self.logger.info(f"Retrieved {len(notes)} trade notes.")
            return notes
        except SQLAlchemyError as e: self.logger.error(f"Error retrieving trade notes: {e}", exc_info=True); return []
        finally:
            if db: db.close()

    def update_trade_note(self, note_id: int, new_content: Optional[str] = None, new_tags: Optional[str] = None) -> Optional[TradeNoteDB]:
        db = self._get_db()
        if not db: return None
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id).first()
            if not note: return None
            updated = False
            if new_content is not None: note.note_content = new_content; updated = True
            if new_tags is not None: note.tags = new_tags; updated = True
            if updated: note.note_timestamp = dt.datetime.utcnow(); db.commit(); db.refresh(note)
            self.logger.info(f"Trade note ID {note_id} {'updated' if updated else 'not changed'}.")
            return note
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error updating note {note_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def delete_trade_note(self, note_id: int) -> bool:
        db = self._get_db()
        if not db: return False
        try:
            note = db.query(TradeNoteDB).filter(TradeNoteDB.id == note_id).first()
            if not note: return False
            db.delete(note); db.commit()
            self.logger.info(f"Trade note ID {note_id} deleted.")
            return True
        except SQLAlchemyError as e: db.rollback(); self.logger.error(f"Error deleting note {note_id}: {e}", exc_info=True); return False
        finally:
            if db: db.close()
