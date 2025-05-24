# services/database_setup.py
"""
Centralized database setup: engine, SessionLocal, Base, and table creation.
Security Hardening Update:
- Added PasswordResetToken table for secure password resets.
"""
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Index, UniqueConstraint, Float, Boolean # Added Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
import streamlit as st
from typing import Optional
import datetime as dt

try:
    from config import APP_TITLE, DATABASE_URL, DEFAULT_BENCHMARK_TICKER
except ImportError:
    print("Warning (database_setup.py): Could not import from config. Using placeholders.")
    APP_TITLE = "TradingDashboard_DB_Setup_Fallback"
    DATABASE_URL = "sqlite:///./fallback_trading_dashboard.db"
    DEFAULT_BENCHMARK_TICKER = "SPY"

logger = logging.getLogger(APP_TITLE)

engine = None
SessionLocal = None # type: ignore
Base = None # type: ignore

try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info(f"Database components (engine, SessionLocal, Base) initialized for URL: {DATABASE_URL}")
except Exception as e_db_core_setup:
    logger.critical(f"Failed to create core database components for URL '{DATABASE_URL}': {e_db_core_setup}", exc_info=True)

class UserFile(Base): # type: ignore
    __tablename__ = "user_files"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_user_files_user_id"), nullable=False, index=True)
    original_file_name = Column(String(255), nullable=False)
    storage_identifier = Column(String(1024), nullable=False, unique=True)
    file_size_bytes = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    status = Column(String(50), default="active", nullable=False, index=True)
    def __repr__(self): return f"<UserFile(id={self.id}, user_id={self.user_id}, name='{self.original_file_name}', status='{self.status}')>"

class UserFileMapping(Base): # type: ignore
    __tablename__ = "user_file_mappings"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_user_file_mappings_user_id"), nullable=False, index=True)
    user_file_id = Column(Integer, ForeignKey("user_files.id", name="fk_user_file_mappings_file_id"), nullable=False, index=True)
    conceptual_column = Column(String(255), nullable=False)
    csv_column_name = Column(String(255), nullable=False)
    mapped_timestamp = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    __table_args__ = (UniqueConstraint('user_id', 'user_file_id', 'conceptual_column', name='uq_user_file_conceptual_mapping'),
                      Index('idx_user_file_mapping_lookup', 'user_id', 'user_file_id'))
    def __repr__(self): return f"<UserFileMapping(user_file_id={self.user_file_id}, conceptual='{self.conceptual_column}' -> csv='{self.csv_column_name}')>"

class UserSettings(Base): # type: ignore
    __tablename__ = "user_settings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_user_settings_user_id"), unique=True, nullable=False, index=True)
    default_theme = Column(String(50), default="dark")
    default_risk_free_rate = Column(Float, default=0.02)
    default_benchmark_ticker = Column(String(50), default=DEFAULT_BENCHMARK_TICKER)
    last_updated = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    def __repr__(self): return f"<UserSettings(user_id={self.user_id}, theme='{self.default_theme}')>"

class PasswordResetToken(Base): # type: ignore
    """
    SQLAlchemy model to store password reset tokens.
    """
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_password_reset_tokens_user_id"), nullable=False, index=True)
    token = Column(String(128), unique=True, index=True, nullable=False) # Store hashed token if preferred, but secure random string is common
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    user = relationship("User") # Define relationship to User model

    def __repr__(self):
        return f"<PasswordResetToken(id={self.id}, user_id={self.user_id}, used={self.used}, expires_at='{self.expires_at}')>"


@st.cache_resource
def get_db_session() -> Optional[Session]:
    if not SessionLocal:
        logger.error("SessionLocal is not initialized. Cannot create DB session.")
        return None
    try:
        db = SessionLocal(); logger.debug("Database session created by get_db_session."); return db
    except Exception as e: logger.error(f"Error creating DB session: {e}", exc_info=True); return None

def create_db_tables():
    if Base and engine:
        try:
            from .auth_service import User
            from .data_service import TradeNoteDB
            # UserFile, UserFileMapping, UserSettings, PasswordResetToken are defined here

            Base.metadata.create_all(bind=engine)
            logger.info("Database tables (User, TradeNoteDB, UserFile, UserFileMapping, UserSettings, PasswordResetToken) checked/created.")
        except ImportError as e: logger.error(f"Failed to import models for table creation: {e}.", exc_info=True)
        except Exception as e:
            logger.error(f"Error creating database tables: {e}", exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'): st.error(f"DB Error: Could not create tables: {e}")
    else:
        logger.error("DB engine or Base not initialized. Cannot create tables.")
        if 'st' in globals() and hasattr(st, 'error'): st.error("DB connection components failed.")

if __name__ == "__main__":
    print("Attempting to create database tables (including PasswordResetToken)...")
    create_db_tables()
    print("Database table creation process finished. Check logs for details.")
