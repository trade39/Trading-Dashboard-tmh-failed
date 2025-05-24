# services/auth_service.py
"""
Service for handling user authentication, registration, management,
and password reset functionality.
"""
import datetime as dt
from typing import Optional, Dict, Any
import re
import secrets # For generating secure tokens

# --- SQLAlchemy Core Imports ---
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.orm import Session, relationship
# --- End SQLAlchemy Core Imports ---

from passlib.context import CryptContext

try:
    # Use relative import for database_setup within the same package
    from .database_setup import Base, get_db_session, UserSettings, PasswordResetToken
    # Import from root config.py
    from config import APP_TITLE, RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER, APP_BASE_URL, PASSWORD_RESET_LINK_EXPIRY_HOURS
    # Import from root utils.email_utils
    from utils.email_utils import send_password_reset_email
except ImportError as e:
    # This fallback is for extreme cases where the primary imports fail.
    # It's less likely to be hit if the project structure is correct and app.py manages PYTHONPATH.
    print(f"AuthService: Critical import failed: {e}. Using placeholders. This may indicate a serious setup issue.")
    APP_TITLE = "TradingDashboard_AuthService_Fallback"
    RISK_FREE_RATE = 0.02; DEFAULT_BENCHMARK_TICKER = "SPY"; APP_BASE_URL = "http://localhost:8501"; PASSWORD_RESET_LINK_EXPIRY_HOURS = 1

    # Fallback SQLAlchemy setup (minimal, for basic structure if main DB setup fails)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, declarative_base
    Base = declarative_base() # type: ignore
    # Fallback UserSettings and PasswordResetToken if they couldn't be imported
    if 'UserSettings' not in globals():
        class UserSettings(Base): # type: ignore
            __tablename__ = "user_settings_fallback_auth_service"
            id=Column(Integer, primary_key=True); user_id=Column(Integer, unique=True); default_theme=Column(String(50),default="dark")
            default_risk_free_rate=Column(Float,default=0.02); default_benchmark_ticker=Column(String(50),default="SPY")
            last_updated = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)


    if 'PasswordResetToken' not in globals():
        class PasswordResetToken(Base): # type: ignore
            __tablename__ = "password_reset_tokens_fallback_auth_service"
            id=Column(Integer, primary_key=True); user_id=Column(Integer); token=Column(String, unique=True); expires_at=Column(DateTime); used=Column(Boolean,default=False)
            created_at = Column(DateTime, default=dt.datetime.utcnow)


    # Fallback email sender
    if 'send_password_reset_email' not in globals():
        def send_password_reset_email(r,u,l): print(f"Mock email (fallback) to {r} with link {l}"); return True
    
    # Fallback DB session (in-memory, not persistent)
    if 'get_db_session' not in globals():
        engine_fallback_auth = create_engine("sqlite:///:memory:")
        SessionLocal_fallback_auth = sessionmaker(autocommit=False, autoflush=False, bind=engine_fallback_auth)
        def get_db_session(): return SessionLocal_fallback_auth()
        Base.metadata.create_all(bind=engine_fallback_auth) # Create tables for fallback session


import logging
logger = logging.getLogger(APP_TITLE)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

MIN_PASSWORD_LENGTH = 8
PASSWORD_REGEX_UPPER = re.compile(r"[A-Z]")
PASSWORD_REGEX_LOWER = re.compile(r"[a-z]")
PASSWORD_REGEX_DIGIT = re.compile(r"\d")
PASSWORD_REGEX_SPECIAL = re.compile(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]")

# User model defined here, using imported SQLAlchemy components
class User(Base): # type: ignore
    __tablename__ = "users" # This table name should match what's used in database_setup.py if User is defined there too.
                            # It's generally better to define models once in database_setup.py and import them.
                            # If User is already defined in database_setup.py and imported, this redefinition might cause issues
                            # or be ignored if the one from database_setup.py is already in Base.metadata.
                            # For now, assuming this is the primary definition or it's compatible.

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, index=True, nullable=False) # Added length
    email = Column(String(255), unique=True, index=True, nullable=True)    # Added length
    hashed_password = Column(String(255), nullable=False)                   # Added length
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships should ideally be defined where UserSettings and PasswordResetToken are definitively imported from
    # If UserSettings and PasswordResetToken are correctly imported from .database_setup, these relationships are fine.
    settings = relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")
    password_reset_tokens = relationship("PasswordResetToken", backref="user_tokens") # Added backref

    def __repr__(self): return f"<User(id={self.id}, username='{self.username}')>"

class AuthService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        # Ensure User model is known to Base metadata if it wasn't created by database_setup.py
        # This is tricky. Best practice: Define all models in database_setup.py.
        # If User is defined here AND in database_setup, ensure they are identical or one is authoritative.
        # For now, assuming create_db_tables() in app.py handles User table creation based on one definition.
        self.logger.info("AuthService initialized.")

    def _get_db(self) -> Optional[Session]:
        # This should use the get_db_session imported from .database_setup
        # which uses the main SessionLocal.
        return get_db_session()

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception: # Broad exception for safety with passlib
            return False

    def get_user_by_username(self, username: str) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try:
            return db.query(User).filter(User.username == username).first()
        finally:
            db.close()

    def get_user_by_email(self, email: str) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try:
            return db.query(User).filter(User.email == email).first()
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()

    def validate_password_complexity(self, password: str) -> Optional[str]:
        if len(password) < MIN_PASSWORD_LENGTH: return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        if not PASSWORD_REGEX_UPPER.search(password): return "Password needs an uppercase letter."
        if not PASSWORD_REGEX_LOWER.search(password): return "Password needs a lowercase letter."
        if not PASSWORD_REGEX_DIGIT.search(password): return "Password needs a digit."
        if not PASSWORD_REGEX_SPECIAL.search(password): return "Password needs a special character (e.g., !@#$%)."
        return None

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
        db = self._get_db()
        if not db: return {"error": "Database connection error during registration."}

        if not username.strip() or not email.strip(): # Basic validation
            return {"error": "Username and email cannot be empty."}

        # Validate email format (simple regex)
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return {"error": "Invalid email format."}


        existing_user_by_name = self.get_user_by_username(username)
        if existing_user_by_name:
            return {"error": f"Username '{username}' is already taken."}

        existing_user_by_email = self.get_user_by_email(email)
        if existing_user_by_email:
            return {"error": f"Email '{email}' is already registered."}

        password_error = self.validate_password_complexity(password)
        if password_error:
            return {"error": password_error}

        new_user = User(username=username, hashed_password=self.hash_password(password), email=email)
        # Create default settings for the new user
        # Ensure UserSettings is correctly imported and available
        default_user_settings = UserSettings(
            default_theme="dark", # Default theme
            default_risk_free_rate=RISK_FREE_RATE,
            default_benchmark_ticker=DEFAULT_BENCHMARK_TICKER
        )
        new_user.settings = default_user_settings # Associate settings with the user

        try:
            db.add(new_user)
            db.commit()
            db.refresh(new_user) # Refresh to get new_user.id and relationships
            if new_user.settings: # Also refresh settings if they were associated
                db.refresh(new_user.settings)
            self.logger.info(f"User '{username}' registered successfully (ID: {new_user.id}). Default settings created.")
            return {"user": new_user} # Return the user object or relevant parts
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error during user registration for '{username}': {e}", exc_info=True)
            return {"error": "A server error occurred during registration. Please try again later."}
        finally:
            db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = self.get_user_by_username(username)
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        # Update last_login_at
        db = self._get_db()
        if db:
            try:
                # Query the user again within this session to ensure it's managed
                user_in_db_session = db.query(User).filter(User.id == user.id).first()
                if user_in_db_session:
                    user_in_db_session.last_login_at = dt.datetime.utcnow()
                    db.commit()
                    self.logger.info(f"User '{username}' authenticated. Last login updated.")
                    return user_in_db_session # Return the session-managed user object
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error updating last_login_at for user '{username}': {e}", exc_info=True)
            finally:
                db.close()
        return user # Fallback to returning the initially fetched user if DB update fails

    def create_password_reset_token(self, email: str) -> Dict[str, Any]:
        db = self._get_db()
        if not db: return {"error": "Database connection error."}

        user = self.get_user_by_email(email)
        if not user:
            self.logger.info(f"Password reset request for non-existent or unverified email: {email}")
            return {"success": "If an account with this email exists, a password reset link has been sent."} # Generic message

        if not user.email: # Should be redundant if found by email, but good check
            return {"error": "User does not have a valid email address on file."}

        token_str = secrets.token_urlsafe(32)
        expires_delta = dt.timedelta(hours=PASSWORD_RESET_LINK_EXPIRY_HOURS)
        expires_at = dt.datetime.utcnow() + expires_delta

        new_token_record = PasswordResetToken(user_id=user.id, token=token_str, expires_at=expires_at, used=False)

        try:
            # Invalidate previous active tokens for this user
            db.query(PasswordResetToken).filter(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > dt.datetime.utcnow()
            ).update({"used": True, "expires_at": dt.datetime.utcnow() - dt.timedelta(seconds=1)})

            db.add(new_token_record)
            db.commit()
            db.refresh(new_token_record)
            self.logger.info(f"Password reset token created for user_id {user.id}, email {email}.")

            reset_link = f"{APP_BASE_URL}/?page=reset_password_form&token={token_str}"
            email_sent = send_password_reset_email(user.email, user.username, reset_link)

            if email_sent:
                return {"success": "If an account with this email exists, a password reset link has been sent."}
            else:
                # If email sending fails, consider how to handle the created token.
                # For now, the token remains, but user won't get the link.
                # A more robust system might retry sending or log for manual intervention.
                self.logger.error(f"Failed to send password reset email to {user.email}, but token was created.")
                return {"error": "Could not send password reset email at this time. Please try again later."}
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error creating password reset token for {email}: {e}", exc_info=True)
            return {"error": "Server error occurred while creating reset token."}
        finally:
            if db: db.close()

    def verify_password_reset_token(self, token_str: str) -> Optional[int]:
        db = self._get_db()
        if not db: return None
        try:
            token_record = db.query(PasswordResetToken).filter(PasswordResetToken.token == token_str).first()
            if not token_record:
                self.logger.warning(f"Verify token: Token '{token_str[:8]}...' not found.")
                return None
            if token_record.used:
                self.logger.warning(f"Verify token: Token ID {token_record.id} already used.")
                return None
            if token_record.expires_at < dt.datetime.utcnow():
                self.logger.warning(f"Verify token: Token ID {token_record.id} expired at {token_record.expires_at}.")
                return None
            self.logger.info(f"Password reset token verified for user_id {token_record.user_id}.")
            return token_record.user_id
        except Exception as e:
            self.logger.error(f"Error verifying password reset token: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()

    def reset_password_with_token(self, token_str: str, new_password: str) -> Dict[str, Any]:
        user_id_from_token = self.verify_password_reset_token(token_str)
        if not user_id_from_token:
            return {"error": "Invalid or expired password reset token. Please request a new one."}

        password_validation_error = self.validate_password_complexity(new_password)
        if password_validation_error:
            return {"error": password_validation_error}

        db = self._get_db()
        if not db: return {"error": "Database connection error."}
        try:
            user = db.query(User).filter(User.id == user_id_from_token).first()
            if not user:
                self.logger.error(f"User ID {user_id_from_token} from valid token not found in User table during password reset.")
                return {"error": "User account associated with this token could not be found."}

            user.hashed_password = self.hash_password(new_password)
            token_record = db.query(PasswordResetToken).filter(PasswordResetToken.token == token_str).first()
            if token_record:
                token_record.used = True
            db.commit()
            self.logger.info(f"Password successfully reset for user_id {user_id_from_token}.")
            return {"success": "Your password has been reset successfully."}
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error resetting password for user_id {user_id_from_token}: {e}", exc_info=True)
            return {"error": "A server error occurred while resetting your password."}
        finally:
            if db: db.close()

    def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        db = self._get_db()
        if not db: return None
        try:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            return settings
        finally:
            db.close()

    def update_user_settings(self, user_id: int, new_settings_data: Dict[str, Any]) -> Optional[UserSettings]:
        db = self._get_db()
        if not db: return None
        try:
            user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not user_settings: # Should not happen if settings are created on registration
                self.logger.warning(f"No existing settings found for user {user_id} during update. Creating new ones.")
                user_settings = UserSettings(
                    user_id=user_id,
                    default_theme=new_settings_data.get('default_theme', "dark"),
                    default_risk_free_rate=new_settings_data.get('default_risk_free_rate', RISK_FREE_RATE),
                    default_benchmark_ticker=new_settings_data.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER)
                )
                db.add(user_settings)
            else:
                for key, value in new_settings_data.items():
                    if hasattr(user_settings, key):
                        setattr(user_settings, key, value)
            user_settings.last_updated = dt.datetime.utcnow()
            db.commit()
            db.refresh(user_settings)
            self.logger.info(f"User settings updated for user_id {user_id}: {new_settings_data}")
            return user_settings
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error updating settings for user {user_id}: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()
