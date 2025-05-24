# services/auth_service.py
"""
Service for handling user authentication, registration, and management.
Now includes enhanced password complexity rules during registration.
"""
import datetime as dt
from typing import Optional, Dict, Any
import re # For password complexity regex

from sqlalchemy.orm import Session, relationship
from passlib.context import CryptContext

try:
    from .database_setup import Base, get_db_session, UserSettings
    from config import APP_TITLE, RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER
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
        __tablename__ = "user_settings_fallback_auth_pw"
        id = Column(Integer, primary_key=True); user_id = Column(Integer, unique=True)
        default_theme = Column(String(50), default="dark")
        default_risk_free_rate = Column(Float, default=0.02)
        default_benchmark_ticker = Column(String(50), default="SPY")


import logging
logger = logging.getLogger(APP_TITLE)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Password complexity rules
MIN_PASSWORD_LENGTH = 8
PASSWORD_REGEX_UPPER = re.compile(r"[A-Z]")
PASSWORD_REGEX_LOWER = re.compile(r"[a-z]")
PASSWORD_REGEX_DIGIT = re.compile(r"\d")
PASSWORD_REGEX_SPECIAL = re.compile(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]") # Adjust as needed

class User(Base): # type: ignore
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    settings = relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")
    def __repr__(self): return f"<User(id={self.id}, username='{self.username}')>"

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

    def validate_password_complexity(self, password: str) -> Optional[str]:
        """
        Validates password against complexity rules.
        Returns None if valid, or an error message string if invalid.
        """
        if len(password) < MIN_PASSWORD_LENGTH:
            return f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        if not PASSWORD_REGEX_UPPER.search(password):
            return "Password must contain at least one uppercase letter."
        if not PASSWORD_REGEX_LOWER.search(password):
            return "Password must contain at least one lowercase letter."
        if not PASSWORD_REGEX_DIGIT.search(password):
            return "Password must contain at least one digit."
        if not PASSWORD_REGEX_SPECIAL.search(password):
            return "Password must contain at least one special character (e.g., !@#$%)."
        return None


    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
        """
        Registers a new user.
        Returns a dictionary with 'user': User object on success, 
        or 'error': error message string on failure.
        """
        db = self._get_db()
        if not db:
            self.logger.error("DB session not available for registration.")
            return {"error": "Database connection error."}

        if self.get_user_by_username(username):
            self.logger.warning(f"Username '{username}' already exists.")
            return {"error": f"Username '{username}' already exists."}
        
        if email:
            try:
                if db.query(User).filter(User.email == email).first():
                    self.logger.warning(f"Email '{email}' already registered.")
                    return {"error": f"Email '{email}' already registered."}
            except Exception as e:
                self.logger.error(f"Error checking email '{email}': {e}", exc_info=True)
                return {"error": "Error validating email."}

        password_validation_error = self.validate_password_complexity(password)
        if password_validation_error:
            self.logger.warning(f"Password validation failed for user '{username}': {password_validation_error}")
            return {"error": password_validation_error}
        
        hashed_password = self.hash_password(password)
        new_user = User(username=username, hashed_password=hashed_password, email=email, created_at=dt.datetime.utcnow())
        
        default_settings = UserSettings(
            default_theme="dark",
            default_risk_free_rate=RISK_FREE_RATE,
            default_benchmark_ticker=DEFAULT_BENCHMARK_TICKER
        )
        new_user.settings = default_settings

        try:
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            if new_user.settings: db.refresh(new_user.settings)
            self.logger.info(f"User '{username}' registered (ID {new_user.id}) with default settings.")
            return {"user": new_user}
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error registering user '{username}': {e}", exc_info=True)
            return {"error": "Registration failed due to a server error."}
        finally:
            if db: db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        # ... (implementation as before) ...
        user = self.get_user_by_username(username)
        if not user or not self.verify_password(password, user.hashed_password):
            self.logger.warning(f"Authentication failed for user '{username}'.")
            return None
        db = self._get_db()
        if db:
            try:
                user_in_db = db.query(User).filter(User.id == user.id).first()
                if user_in_db:
                    user_in_db.last_login_at = dt.datetime.utcnow(); db.commit(); db.refresh(user_in_db)
                    self.logger.info(f"User '{username}' authenticated. Last login updated.")
                    return user_in_db
                return user 
            except Exception as e:
                db.rollback(); self.logger.error(f"Error updating last_login for '{username}': {e}", exc_info=True)
                return user 
            finally:
                if db: db.close()
        return user

    def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        # ... (implementation as before) ...
        db = self._get_db();
        if not db: return None
        try:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if settings: self.logger.info(f"Retrieved settings for user_id {user_id}.")
            else: self.logger.info(f"No settings found for user_id {user_id}, will use app defaults.")
            return settings
        except Exception as e: self.logger.error(f"Error retrieving settings for user_id {user_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def update_user_settings(self, user_id: int, new_settings_data: Dict[str, Any]) -> Optional[UserSettings]:
        # ... (implementation as before) ...
        db = self._get_db();
        if not db: return None
        try:
            user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not user_settings:
                self.logger.info(f"No existing settings for user_id {user_id}. Creating new settings entry.")
                user_settings = UserSettings(user_id=user_id, default_theme=new_settings_data.get('default_theme', "dark"), default_risk_free_rate=new_settings_data.get('default_risk_free_rate', RISK_FREE_RATE), default_benchmark_ticker=new_settings_data.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER))
                db.add(user_settings)
            else:
                if 'default_theme' in new_settings_data: user_settings.default_theme = new_settings_data['default_theme']
                if 'default_risk_free_rate' in new_settings_data: user_settings.default_risk_free_rate = new_settings_data['default_risk_free_rate']
                if 'default_benchmark_ticker' in new_settings_data: user_settings.default_benchmark_ticker = new_settings_data['default_benchmark_ticker']
            user_settings.last_updated = dt.datetime.utcnow(); db.commit(); db.refresh(user_settings)
            self.logger.info(f"Updated settings for user_id {user_id}: {new_settings_data}")
            return user_settings
        except Exception as e:
            db.rollback(); self.logger.error(f"Error updating settings for user_id {user_id}: {e}", exc_info=True); return None
        finally:
            if db: db.close()
