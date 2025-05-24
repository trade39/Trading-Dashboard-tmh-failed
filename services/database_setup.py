# services/database_setup.py
"""
Centralized database setup: engine, SessionLocal, Base, and table creation.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session # <<< THIS IMPORT IS CRUCIAL
import streamlit as st
from typing import Optional # Ensure Optional is imported

try:
    from config import APP_TITLE, DATABASE_URL
except ImportError:
    # Fallback for standalone testing or if imports fail during generation
    print("Warning (database_setup.py): Could not import from config. Using placeholders.")
    APP_TITLE = "TradingDashboard_DB_Setup_Fallback"
    DATABASE_URL = "sqlite:///./fallback_trading_dashboard.db" # Ensure this path is writable if run standalone

logger = logging.getLogger(APP_TITLE)

# --- Database Engine and Session Setup ---
engine = None
SessionLocal = None # type: ignore
Base = None # type: ignore

try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base() # Base is defined here
    logger.info(f"Database components (engine, SessionLocal, Base) initialized for URL: {DATABASE_URL}")
except Exception as e_db_core_setup:
    logger.critical(f"Failed to create core database components for URL '{DATABASE_URL}': {e_db_core_setup}", exc_info=True)
    # engine, SessionLocal, Base will remain None if this fails


@st.cache_resource
def get_db_session() -> Optional[Session]: # Session is now defined for type hint
    """
    Provides a database session.
    """
    if not SessionLocal:
        logger.error("SessionLocal is not initialized (likely due to DB engine setup failure). Cannot create DB session.")
        return None
    try:
        db = SessionLocal()
        logger.debug("Database session created by get_db_session.")
        return db
    except Exception as e_get_session:
        logger.error(f"Error creating database session in get_db_session: {e_get_session}", exc_info=True)
        return None

def create_db_tables():
    """
    Creates all database tables defined by models inheriting from the central Base.
    This function must be called after all model definitions are imported and associated with Base.
    """
    if Base and engine: # Ensure Base and engine were successfully initialized
        try:
            # These imports must succeed for tables to be registered with Base.metadata
            from .auth_service import User  # pylint: disable=import-outside-toplevel
            from .data_service import TradeNoteDB # pylint: disable=import-outside-toplevel
            # Add imports for any other ORM models here if they exist

            Base.metadata.create_all(bind=engine)
            logger.info("Database tables checked/created successfully based on all known models.")
        except ImportError as e_model_import:
            logger.error(f"Failed to import one or more ORM models for table creation: {e_model_import}. Some tables might not be created. This often indicates a circular dependency or an issue in the model files themselves.", exc_info=True)
        except Exception as e_create_all:
            logger.error(f"Error creating database tables via Base.metadata.create_all: {e_create_all}", exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'):
                st.error(f"Database Error: Could not create tables. Check logs. Error: {e_create_all}")
    else:
        logger.error("Database engine or Base not initialized. Cannot create tables.")
        if 'st' in globals() and hasattr(st, 'error'):
            st.error("Database connection components (engine/Base) failed to initialize. Database features will not work.")

