# services/auth_service.py
"""
Service for handling user authentication, registration, management,
and password reset functionality.
"""
import datetime as dt
from typing import Optional, Dict, Any
import re
import secrets # For generating secure tokens
from sqlalchemy.orm import Session, relationship
from passlib.context import CryptContext

try:
    from .database_setup import Base, get_db_session, UserSettings, PasswordResetToken # Import PasswordResetToken
    from config import APP_TITLE, RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER, APP_BASE_URL, PASSWORD_RESET_LINK_EXPIRY_HOURS
    from utils.email_utils import send_password_reset_email # Import email utility
except ImportError as e:
    print(f"AuthService: Critical import failed: {e}. Using placeholders.")
    APP_TITLE = "TradingDashboard_AuthService_Fallback"
    RISK_FREE_RATE = 0.02; DEFAULT_BENCHMARK_TICKER = "SPY"; APP_BASE_URL = "http://localhost:8501"; PASSWORD_RESET_LINK_EXPIRY_HOURS = 1
    from sqlalchemy import Column, Integer, String, DateTime, Float, create_engine, ForeignKey, Boolean
    from sqlalchemy.orm import sessionmaker, declarative_base
    Base = declarative_base() # type: ignore
    engine_fallback_auth = create_engine("sqlite:///./temp_auth_service_test_pw_reset.db") # Different DB for fallback
    SessionLocal_fallback_auth = sessionmaker(autocommit=False, autoflush=False, bind=engine_fallback_auth)
    def get_db_session(): return SessionLocal_fallback_auth()
    class UserSettings(Base): # type: ignore
        __tablename__ = "user_settings_fallback_auth_pw_reset"
        id=Column(Integer, primary_key=True); user_id=Column(Integer, unique=True); default_theme=Column(String(50),default="dark")
        default_risk_free_rate=Column(Float,default=0.02); default_benchmark_ticker=Column(String(50),default="SPY")
    class PasswordResetToken(Base): # type: ignore
        __tablename__ = "password_reset_tokens_fallback_auth_pw_reset"
        id=Column(Integer, primary_key=True); user_id=Column(Integer); token=Column(String, unique=True); expires_at=Column(DateTime); used=Column(Boolean,default=False)
    def send_password_reset_email(r,u,l): print(f"Mock email to {r} with link {l}"); return True


import logging
logger = logging.getLogger(APP_TITLE)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

MIN_PASSWORD_LENGTH = 8
PASSWORD_REGEX_UPPER = re.compile(r"[A-Z]")
PASSWORD_REGEX_LOWER = re.compile(r"[a-z]")
PASSWORD_REGEX_DIGIT = re.compile(r"\d")
PASSWORD_REGEX_SPECIAL = re.compile(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]")

class User(Base): # type: ignore
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True) # Email must be unique if used for reset
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    settings = relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")
    # Relationship for password reset tokens (optional, but can be useful)
    # password_reset_tokens = relationship("PasswordResetToken", back_populates="user")
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
        except Exception: return False

    def get_user_by_username(self, username: str) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try: return db.query(User).filter(User.username == username).first()
        finally: db.close()

    def get_user_by_email(self, email: str) -> Optional[User]: # New method
        db = self._get_db()
        if not db: return None
        try: return db.query(User).filter(User.email == email).first()
        finally: db.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        db = self._get_db()
        if not db: return None
        try: return db.query(User).filter(User.id == user_id).first()
        finally: db.close()

    def validate_password_complexity(self, password: str) -> Optional[str]:
        if len(password) < MIN_PASSWORD_LENGTH: return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        if not PASSWORD_REGEX_UPPER.search(password): return "Password needs an uppercase letter."
        if not PASSWORD_REGEX_LOWER.search(password): return "Password needs a lowercase letter."
        if not PASSWORD_REGEX_DIGIT.search(password): return "Password needs a digit."
        if not PASSWORD_REGEX_SPECIAL.search(password): return "Password needs a special character."
        return None

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
        db = self._get_db()
        if not db: return {"error": "Database error."}
        if self.get_user_by_username(username): return {"error": f"Username '{username}' taken."}
        if email and self.get_user_by_email(email): return {"error": f"Email '{email}' registered."}
        
        password_error = self.validate_password_complexity(password)
        if password_error: return {"error": password_error}
        
        new_user = User(username=username, hashed_password=self.hash_password(password), email=email)
        new_user.settings = UserSettings(
            default_theme="dark",
            default_risk_free_rate=RISK_FREE_RATE,
            default_benchmark_ticker=DEFAULT_BENCHMARK_TICKER
        )
        try:
            db.add(new_user); db.commit(); db.refresh(new_user)
            if new_user.settings: db.refresh(new_user.settings)
            self.logger.info(f"User '{username}' registered (ID {new_user.id}).")
            return {"user": new_user}
        except Exception as e:
            db.rollback(); self.logger.error(f"Error registering '{username}': {e}", exc_info=True)
            return {"error": "Server error during registration."}
        finally: db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = self.get_user_by_username(username)
        if not user or not self.verify_password(password, user.hashed_password): return None
        db = self._get_db()
        if db:
            try:
                user_in_db = db.query(User).filter(User.id == user.id).first()
                if user_in_db: user_in_db.last_login_at = dt.datetime.utcnow(); db.commit(); return user_in_db
            except Exception as e: db.rollback(); self.logger.error(f"Err updating last_login for {username}: {e}", exc_info=True)
            finally: db.close()
        return user

    # --- Password Reset Methods ---
    def create_password_reset_token(self, email: str) -> Dict[str, Any]:
        db = self._get_db()
        if not db: return {"error": "Database connection error."}
        
        user = self.get_user_by_email(email) # Find user by email
        if not user:
            # Important: Do not reveal if the email exists or not for security.
            # Send a generic message. The email function will also "pretend" to send.
            self.logger.info(f"Password reset request for non-existent or unverified email: {email}")
            return {"success": "If an account with this email exists, a password reset link has been sent."}

        if not user.email: # Should not happen if get_user_by_email found them, but defensive.
            return {"error": "User does not have an email address on file."}

        token_str = secrets.token_urlsafe(32)
        expires_delta = dt.timedelta(hours=PASSWORD_RESET_LINK_EXPIRY_HOURS)
        expires_at = dt.datetime.utcnow() + expires_delta
        
        new_token_record = PasswordResetToken(user_id=user.id, token=token_str, expires_at=expires_at)
        
        try:
            # Invalidate previous tokens for this user (optional, but good practice)
            db.query(PasswordResetToken).filter(PasswordResetToken.user_id == user.id, PasswordResetToken.used == False).update({"used": True, "expires_at": dt.datetime.utcnow() - dt.timedelta(seconds=1)})

            db.add(new_token_record)
            db.commit()
            db.refresh(new_token_record)
            self.logger.info(f"Password reset token created for user_id {user.id}, email {email}.")

            # Construct reset link
            # The page name 'reset_password_form' will be handled by app.py query_params logic
            reset_link = f"{APP_BASE_URL}/?page=reset_password_form&token={token_str}"
            
            email_sent = send_password_reset_email(user.email, user.username, reset_link)
            if email_sent:
                return {"success": "If an account with this email exists, a password reset link has been sent."}
            else:
                # Rollback token creation if email failed critically (though placeholder always returns True)
                # db.rollback() # Not strictly needed if placeholder always works
                return {"error": "Could not send password reset email. Please try again later."}

        except Exception as e:
            db.rollback()
            self.logger.error(f"Error creating password reset token for {email}: {e}", exc_info=True)
            return {"error": "Server error creating reset token."}
        finally:
            if db: db.close()

    def verify_password_reset_token(self, token_str: str) -> Optional[int]:
        """Verifies a token and returns user_id if valid, else None."""
        db = self._get_db()
        if not db: return None
        try:
            token_record = db.query(PasswordResetToken).filter(PasswordResetToken.token == token_str).first()
            if not token_record: self.logger.warning(f"Verify token: Token not found."); return None
            if token_record.used: self.logger.warning(f"Verify token: Token already used (ID: {token_record.id})."); return None
            if token_record.expires_at < dt.datetime.utcnow(): self.logger.warning(f"Verify token: Token expired (ID: {token_record.id})."); return None
            
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
            return {"error": "Invalid or expired password reset token."}

        password_validation_error = self.validate_password_complexity(new_password)
        if password_validation_error:
            return {"error": password_validation_error}

        db = self._get_db()
        if not db: return {"error": "Database connection error."}
        try:
            user = db.query(User).filter(User.id == user_id_from_token).first()
            if not user:
                self.logger.error(f"User ID {user_id_from_token} from valid token not found in User table.")
                return {"error": "User account not found."}

            user.hashed_password = self.hash_password(new_password)
            
            # Mark token as used
            token_record = db.query(PasswordResetToken).filter(PasswordResetToken.token == token_str).first()
            if token_record: token_record.used = True
            
            db.commit()
            self.logger.info(f"Password successfully reset for user_id {user_id_from_token}.")
            return {"success": "Password has been reset successfully."}
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error resetting password for user_id {user_id_from_token}: {e}", exc_info=True)
            return {"error": "Server error resetting password."}
        finally:
            if db: db.close()

    # --- User Settings Methods (as before) ---
    def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        db = self._get_db();
        if not db: return None
        try:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            return settings
        finally: db.close()

    def update_user_settings(self, user_id: int, new_settings_data: Dict[str, Any]) -> Optional[UserSettings]:
        db = self._get_db();
        if not db: return None
        try:
            user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not user_settings:
                user_settings = UserSettings(user_id=user_id, default_theme=new_settings_data.get('default_theme', "dark"), default_risk_free_rate=new_settings_data.get('default_risk_free_rate', RISK_FREE_RATE), default_benchmark_ticker=new_settings_data.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER))
                db.add(user_settings)
            else:
                for key, value in new_settings_data.items():
                    if hasattr(user_settings, key): setattr(user_settings, key, value)
            user_settings.last_updated = dt.datetime.utcnow(); db.commit(); db.refresh(user_settings)
            return user_settings
        except Exception as e: db.rollback(); self.logger.error(f"Error updating settings for user {user_id}: {e}", exc_info=True); return None
        finally: db.close()
