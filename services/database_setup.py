# services/database_setup.py
"""
Centralized database setup: engine, SessionLocal, Base, and table creation.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session # <<< ENSURE THIS IMPORT IS PRESENT
import streamlit as st 

try:
    from config import APP_TITLE, DATABASE_URL
except ImportError:
    print("Warning (database_setup.py): Could not import from config. Using placeholders.")
    APP_TITLE = "TradingDashboard_DB_Setup_Fallback"
    DATABASE_URL = "sqlite:///./fallback_trading_dashboard.db"

logger = logging.getLogger(APP_TITLE)

# --- Database Engine and Session Setup ---
try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info(f"Database components (engine, SessionLocal, Base) initialized for URL: {DATABASE_URL}")
except Exception as e_db_core_setup:
    logger.critical(f"Failed to create core database components (engine, SessionLocal, Base) for URL '{DATABASE_URL}': {e_db_core_setup}", exc_info=True)
    engine = None
    SessionLocal = None # type: ignore
    Base = None # type: ignore


@st.cache_resource
def get_db_session() -> Optional[Session]: # <<< Session type hint is now valid
    """
    Provides a database session.
    """
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
    """
    Creates all database tables defined by models inheriting from the central Base.
    This function must be called after all model definitions are imported and associated with Base.
    """
    if Base and engine:
        try:
            # Import all models here to ensure they are registered with Base.metadata
            from .auth_service import User  # pylint: disable=import-outside-toplevel 
            from .data_service import TradeNoteDB # pylint: disable=import-outside-toplevel
            # Add imports for any other ORM models here

            Base.metadata.create_all(bind=engine)
            logger.info("Database tables checked/created successfully based on all known models.")
        except ImportError as e_model_import:
            logger.error(f"Failed to import one or more models for table creation: {e_model_import}. Some tables might not be created.", exc_info=True)
            try:
                logger.warning("Attempting to create tables with currently known metadata due to model import error...")
                Base.metadata.create_all(bind=engine)
            except Exception as e_partial:
                logger.error(f"Error during partial table creation: {e_partial}", exc_info=True)

        except Exception as e_create_all:
            logger.error(f"Error creating database tables via Base.metadata.create_all: {e_create_all}", exc_info=True)
            if 'st' in globals() and hasattr(st, 'error'): 
                st.error(f"Database Error: Could not create tables. Check logs. Error: {e_create_all}")
    else:
        logger.error("Database engine or Base not initialized. Cannot create tables.")
        if 'st' in globals() and hasattr(st, 'error'):
            st.error("Database connection components (engine/Base) failed to initialize. Database features will not work.")

