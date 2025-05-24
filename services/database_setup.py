# services/database_setup.py
"""
Centralized database setup: engine, SessionLocal, Base, and table creation.
User Scoping Update:
- Added user_id to TradeNoteDB (will be handled by data_service.py for model definition).
- Added UserFileMapping table for persistent column mappings per user per file.
"""
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import streamlit as st
from typing import Optional
import datetime as dt

try:
    from config import APP_TITLE, DATABASE_URL
except ImportError:
    print("Warning (database_setup.py): Could not import from config. Using placeholders.")
    APP_TITLE = "TradingDashboard_DB_Setup_Fallback"
    DATABASE_URL = "sqlite:///./fallback_trading_dashboard.db"

logger = logging.getLogger(APP_TITLE)

# --- Database Engine and Session Setup ---
engine = None
SessionLocal = None # type: ignore
Base = None # type: ignore

try:
    engine = create_engine(DATABASE_URL, echo=False) # Set echo=True for SQL logging if needed
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info(f"Database components (engine, SessionLocal, Base) initialized for URL: {DATABASE_URL}")
except Exception as e_db_core_setup:
    logger.critical(f"Failed to create core database components for URL '{DATABASE_URL}': {e_db_core_setup}", exc_info=True)

# --- ORM Model Definitions ---

class UserFile(Base): # type: ignore
    """
    SQLAlchemy UserFile model to store metadata about user-uploaded files.
    """
    __tablename__ = "user_files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_user_files_user_id"), nullable=False, index=True)
    original_file_name = Column(String(255), nullable=False)
    storage_identifier = Column(String(1024), nullable=False, unique=True) # Path relative to USER_FILES_ROOT_DIR
    file_size_bytes = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    file_hash = Column(String(64), nullable=True, index=True) # SHA256 hash
    status = Column(String(50), default="active", nullable=False, index=True) # e.g., "active", "deleted"

    def __repr__(self):
        return f"<UserFile(id={self.id}, user_id={self.user_id}, name='{self.original_file_name}', status='{self.status}')>"

class UserFileMapping(Base): # type: ignore
    """
    SQLAlchemy model to store user-defined column mappings for specific files.
    """
    __tablename__ = "user_file_mappings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_user_file_mappings_user_id"), nullable=False, index=True)
    user_file_id = Column(Integer, ForeignKey("user_files.id", name="fk_user_file_mappings_file_id"), nullable=False, index=True)
    conceptual_column = Column(String(255), nullable=False) # e.g., 'pnl', 'date'
    csv_column_name = Column(String(255), nullable=False) # Actual header from the user's CSV
    mapped_timestamp = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('user_id', 'user_file_id', 'conceptual_column', name='uq_user_file_conceptual_mapping'),
        Index('idx_user_file_mapping_lookup', 'user_id', 'user_file_id'),
    )

    def __repr__(self):
        return f"<UserFileMapping(id={self.id}, user_file_id={self.user_file_id}, conceptual='{self.conceptual_column}' -> csv='{self.csv_column_name}')>"


@st.cache_resource
def get_db_session() -> Optional[Session]:
    if not SessionLocal:
        logger.error("SessionLocal is not initialized. Cannot create DB session.")
        return None
    try:
        db = SessionLocal()
        logger.debug("Database session created by get_db_session.")
        return db
    except Exception as e_get_session:
        logger.error(f"Error creating database session in get_db_session: {e_get_session}", exc_info=True)
        return None

def create_db_tables():
    if Base and engine:
        try:
            # Import models here to ensure they are registered with Base
            from .auth_service import User  # pylint: disable=import-outside-toplevel
            from .data_service import TradeNoteDB # pylint: disable=import-outside-toplevel
            # UserFile and UserFileMapping are already defined in this file and associated with Base

            Base.metadata.create_all(bind=engine)
            logger.info("Database tables (User, TradeNoteDB, UserFile, UserFileMapping) checked/created successfully.")
        except ImportError as e_model_import:
            logger.error(f"Failed to import User or TradeNoteDB models for table creation: {e_model_import}.", exc_info=True)
        except Exception as e_create_all:
            logger.error(f"Error creating database tables: {e_create_all}", exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'): # Check if Streamlit context exists
                st.error(f"Database Error: Could not create tables. Check logs. Error: {e_create_all}")
    else:
        logger.error("Database engine or Base not initialized. Cannot create tables.")
        if 'st' in globals() and hasattr(st, 'error'): # Check if Streamlit context exists
            st.error("Database connection components failed. Database features will not work.")

if __name__ == "__main__":
    # This allows running `python -m services.database_setup` to create tables
    # Ensure config.py is accessible from the root of the project when running this.
    print("Attempting to create database tables...")
    create_db_tables()
    print("Database table creation process finished. Check logs for details.")
