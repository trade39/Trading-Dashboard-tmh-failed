# services/auth_service.py
"""
Service for handling user authentication, registration, and management.
Now includes methods for managing user-specific settings.
"""
import datetime as dt
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session, relationship # Added relationship
from passlib.context import CryptContext

try:
    # UserSettings model is now defined in database_setup
    from .database_setup import Base, get_db_session, UserSettings
    from config import APP_TITLE, RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER # For default settings
except ImportError:
    print("AuthService: Could not import from database_setup or config. Using placeholders.")
    APP_TITLE = "TradingDashboard_AuthService_Fallback"
    RISK_FREE_RATE = 0.02
    DEFAULT_BENCHMARK_TICKER = "SPY"
    from sqlalchemy import Column, Integer, String, DateTime, Float, create_engine, ForeignKey
    from sqlalchemy.orm import sessionmaker, declarative_base
    Base = declarative_base() # type: ignore
    engine_fallback_auth = create_engine("sqlite:///./temp_auth_service_test.db")
    SessionLocal_fallback_auth = sessionmaker(autocommit=False, autoflush=False, bind=engine_fallback_auth)
    def get_db_session(): return SessionLocal_fallback_auth()
    class UserSettings(Base): # type: ignore
        __tablename__ = "user_settings_fallback_auth" # Different name to avoid conflict if main setup runs
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, unique=True)
        default_theme = Column(String(50), default="dark")
        default_risk_free_rate = Column(Float, default=0.02)
        default_benchmark_ticker = Column(String(50), default="SPY")


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

    # Define the one-to-one relationship to UserSettings
    settings = relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class AuthService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        self.logger.info("AuthService initialized.")

    def _get_db(self) -> Optional[Session]:
        return get_db_session()

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try: return pwd_context.verify(plain_password, hashed_password)
        except Exception as e: self.logger.error(f"Error verifying password: {e}", exc_info=True); return False

    def get_user_by_username(self, username: str) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try: return db.query(User).filter(User.username == username).first()
        except Exception as e: self.logger.error(f"Error fetching user '{username}': {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try: return db.query(User).filter(User.id == user_id).first()
        except Exception as e: self.logger.error(f"Error fetching user ID {user_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Optional[User]:
        db = self._get_db()
        if not db: self.logger.error("DB session not available for registration."); return None
        if self.get_user_by_username(username): self.logger.warning(f"Username '{username}' already exists."); return None
        if email:
            try:
                if db.query(User).filter(User.email == email).first():
                    self.logger.warning(f"Email '{email}' already registered."); return None
            except Exception as e: self.logger.error(f"Error checking email '{email}': {e}", exc_info=True); return None
        
        hashed_password = self.hash_password(password)
        new_user = User(username=username, hashed_password=hashed_password, email=email, created_at=dt.datetime.utcnow())
        
        # Create default settings for the new user
        default_settings = UserSettings(
            user_id=new_user.id, # This will be set after new_user is flushed if user_id is FK
            default_theme="dark", # Default theme from your preference
            default_risk_free_rate=RISK_FREE_RATE, # Global default
            default_benchmark_ticker=DEFAULT_BENCHMARK_TICKER # Global default
        )
        new_user.settings = default_settings # Associate settings with user

        try:
            db.add(new_user) # This will also add default_settings due to cascade
            db.commit()
            db.refresh(new_user)
            if new_user.settings: # Verify settings were created
                 db.refresh(new_user.settings)
            self.logger.info(f"User '{username}' registered (ID {new_user.id}) with default settings.")
            return new_user
        except Exception as e:
            db.rollback(); self.logger.error(f"Error registering user '{username}': {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        # ... (authentication logic as before, including last_login_at update) ...
        user = self.get_user_by_username(username)
        if not user or not self.verify_password(password, user.hashed_password):
            self.logger.warning(f"Authentication failed for user '{username}'.")
            return None
        db = self._get_db()
        if db:
            try:
                user_in_db = db.query(User).filter(User.id == user.id).first() # Re-fetch in session
                if user_in_db:
                    user_in_db.last_login_at = dt.datetime.utcnow()
                    db.commit()
                    db.refresh(user_in_db) # Refresh to get updated fields
                    self.logger.info(f"User '{username}' authenticated. Last login updated.")
                    return user_in_db
                return user # Should not happen if user was fetched initially
            except Exception as e:
                db.rollback(); self.logger.error(f"Error updating last_login for '{username}': {e}", exc_info=True)
                return user # Return original user object if DB update fails
            finally:
                if db: db.close()
        return user


    # --- User Settings Methods ---
    def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        """Retrieves settings for a given user_id."""
        db = self._get_db()
        if not db: return None
        try:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if settings:
                self.logger.info(f"Retrieved settings for user_id {user_id}.")
            else:
                self.logger.info(f"No settings found for user_id {user_id}, will use app defaults.")
            return settings
        except Exception as e:
            self.logger.error(f"Error retrieving settings for user_id {user_id}: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def update_user_settings(self, user_id: int, new_settings_data: Dict[str, Any]) -> Optional[UserSettings]:
        """Updates or creates settings for a given user_id."""
        db = self._get_db()
        if not db: return None
        try:
            user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not user_settings:
                # If no settings exist, create them (should ideally be created on user registration)
                self.logger.info(f"No existing settings for user_id {user_id}. Creating new settings entry.")
                user_settings = UserSettings(
                    user_id=user_id,
                    default_theme=new_settings_data.get('default_theme', "dark"),
                    default_risk_free_rate=new_settings_data.get('default_risk_free_rate', RISK_FREE_RATE),
                    default_benchmark_ticker=new_settings_data.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER)
                )
                db.add(user_settings)
            else:
                # Update existing settings
                if 'default_theme' in new_settings_data:
                    user_settings.default_theme = new_settings_data['default_theme']
                if 'default_risk_free_rate' in new_settings_data:
                    user_settings.default_risk_free_rate = new_settings_data['default_risk_free_rate']
                if 'default_benchmark_ticker' in new_settings_data:
                    user_settings.default_benchmark_ticker = new_settings_data['default_benchmark_ticker']
                # Add other updatable settings here
            
            user_settings.last_updated = dt.datetime.utcnow()
            db.commit()
            db.refresh(user_settings)
            self.logger.info(f"Updated settings for user_id {user_id}: {new_settings_data}")
            return user_settings
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error updating settings for user_id {user_id}: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

