# services/database_setup.py
"""
Centralized database setup: engine, SessionLocal, Base, and table creation.
"""
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import streamlit as st
from typing import Optional
import datetime as dt # Added for default timestamp

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
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info(f"Database components (engine, SessionLocal, Base) initialized for URL: {DATABASE_URL}")
except Exception as e_db_core_setup:
    logger.critical(f"Failed to create core database components for URL '{DATABASE_URL}': {e_db_core_setup}", exc_info=True)

# --- ORM Model Definitions ---
# User model will be imported in create_db_tables from auth_service
# TradeNoteDB model will be imported in create_db_tables from data_service

class UserFile(Base): # type: ignore
    """
    SQLAlchemy UserFile model to store metadata about user-uploaded files.
    """
    __tablename__ = "user_files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", name="fk_user_files_user_id"), nullable=False, index=True)
    original_file_name = Column(String(255), nullable=False)
    # storage_identifier will store the relative path for local storage,
    # or a key/path for cloud storage.
    storage_identifier = Column(String(1024), nullable=False, unique=True)
    file_size_bytes = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    file_hash = Column(String(64), nullable=True, index=True) # e.g., SHA256 hash
    status = Column(String(50), default="active", nullable=False, index=True) # e.g., "active", "archived", "deleted"

    # Define table args for potential composite indexes or constraints if needed later
    # For example, to prevent a user from uploading the exact same file (same hash) multiple times actively:
    # __table_args__ = (
    #     Index("idx_user_file_hash_active", "user_id", "file_hash", "status",
    #           unique=True,
    #           postgresql_where=(status == 'active'),  # Conditional unique index for PostgreSQL
    #           sqlite_where=(status == 'active')),     # Conditional unique index for SQLite (requires SQLite 3.8.0+)
    # )

    def __repr__(self):
        return f"<UserFile(id={self.id}, user_id={self.user_id}, name='{self.original_file_name}', status='{self.status}')>"


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
            from .auth_service import User  # pylint: disable=import-outside-toplevel
            from .data_service import TradeNoteDB # pylint: disable=import-outside-toplevel
            # UserFile is already defined in this file and associated with Base

            Base.metadata.create_all(bind=engine)
            logger.info("Database tables (User, TradeNoteDB, UserFile) checked/created successfully.")
        except ImportError as e_model_import:
            logger.error(f"Failed to import User or TradeNoteDB models for table creation: {e_model_import}.", exc_info=True)
        except Exception as e_create_all:
            logger.error(f"Error creating database tables: {e_create_all}", exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'):
                st.error(f"Database Error: Could not create tables. Check logs. Error: {e_create_all}")
    else:
        logger.error("Database engine or Base not initialized. Cannot create tables.")
        if 'st' in globals() and hasattr(st, 'error'):
            st.error("Database connection components failed. Database features will not work.")

