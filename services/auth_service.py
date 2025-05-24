# services/auth_service.py
"""
Service for handling user authentication, registration, and management.
"""
import datetime as dt
from typing import Optional

from sqlalchemy.orm import Session
from passlib.context import CryptContext

try:
    from .database_setup import Base, get_db_session # Import from new database_setup
    from config import APP_TITLE
except ImportError:
    print("AuthService: Could not import Base or get_db_session from database_setup or config. Using placeholders.")
    APP_TITLE = "TradingDashboard_AuthService_Fallback"
    from sqlalchemy import Column, Integer, String, DateTime, create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base
    Base = declarative_base() # type: ignore
    # Fallback engine for standalone testing of this file if imports fail
    engine_fallback_auth = create_engine("sqlite:///./temp_auth_service_test.db")
    SessionLocal_fallback_auth = sessionmaker(autocommit=False, autoflush=False, bind=engine_fallback_auth)
    def get_db_session(): return SessionLocal_fallback_auth()


from sqlalchemy import Column, Integer, String, DateTime # Keep these for model definition

import logging
logger = logging.getLogger(APP_TITLE)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

class User(Base): # type: ignore
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class AuthService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        self.logger.info("AuthService initialized.")

    def _get_db(self) -> Optional[Session]:
        return get_db_session() # Use the imported session getter

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            self.logger.error(f"Error verifying password: {e}", exc_info=True)
            return False

    def get_user_by_username(self, username: str) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try:
            return db.query(User).filter(User.username == username).first()
        except Exception as e:
            self.logger.error(f"Error fetching user by username '{username}': {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            self.logger.error(f"Error fetching user by ID {user_id}: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Optional[User]:
        db = self._get_db()
        if not db:
            self.logger.error("Database session not available for user registration.")
            return None

        # Check for existing username (using the service method which handles its own session)
        if self.get_user_by_username(username): # This will open and close its own session
            self.logger.warning(f"Registration attempt failed: Username '{username}' already exists.")
            # No need to close db here as get_user_by_username does it.
            return None 

        # Check for existing email within the current session 'db'
        if email:
            try:
                existing_email_user = db.query(User).filter(User.email == email).first()
                if existing_email_user:
                    self.logger.warning(f"Registration attempt failed: Email '{email}' already registered.")
                    return None # Email already exists
            except Exception as e_email_check:
                self.logger.error(f"Error checking existing email '{email}': {e_email_check}", exc_info=True)
                return None # Fail registration on DB error during check
            # No finally db.close() here, as it's part of the larger transaction

        hashed_password = self.hash_password(password)
        new_user = User(
            username=username,
            hashed_password=hashed_password,
            email=email,
            created_at=dt.datetime.utcnow()
        )
        try:
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            self.logger.info(f"User '{username}' registered successfully with ID {new_user.id}.")
            return new_user
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error registering user '{username}': {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = self.get_user_by_username(username) # Handles its own session
        if not user:
            self.logger.warning(f"Authentication failed: User '{username}' not found.")
            return None

        if not self.verify_password(password, user.hashed_password):
            self.logger.warning(f"Authentication failed: Invalid password for user '{username}'.")
            return None

        db = self._get_db()
        if db:
            try:
                user_in_session = db.query(User).filter(User.id == user.id).first() # Re-attach/fetch in current session
                if user_in_session:
                    user_in_session.last_login_at = dt.datetime.utcnow()
                    db.commit()
                    self.logger.info(f"User '{username}' authenticated successfully. Last login updated.")
                    # Return the user object that might have been updated (though last_login_at is often not immediately needed by caller)
                    db.refresh(user_in_session) # Ensure the returned object has the latest state
                    return user_in_session
                else:
                    self.logger.error(f"Could not re-fetch user '{username}' in new session to update last_login_at. Returning original user object.")
                    return user # Return the original user object if re-fetch fails
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error updating last_login_at for user '{username}': {e}", exc_info=True)
                return user # Return original user object on error
            finally:
                if db: db.close()
        else:
            self.logger.error("Database session not available to update last_login_at. Returning original user object.")
            return user # Return original user object if DB session fails
