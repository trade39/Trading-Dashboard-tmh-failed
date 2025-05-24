# services/auth_service.py
import datetime as dt
from typing import Optional, Dict, Any
import re
import secrets
import logging

# SQLAlchemy Core Imports
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.orm import Session, relationship, selectinload # Added selectinload

from passlib.context import CryptContext

try:
    from .database_setup import Base, get_db_session, UserSettings, PasswordResetToken
    from config import APP_TITLE, RISK_FREE_RATE, DEFAULT_BENCHMARK_TICKER, APP_BASE_URL, PASSWORD_RESET_LINK_EXPIRY_HOURS
    from utils.email_utils import send_password_reset_email
except ImportError as e:
    # Fallback for critical import errors
    print(f"AuthService: Critical import error: {e}. Using placeholders. Functionality will be limited.")
    APP_TITLE = "TradingDashboard_Auth_Fallback"
    RISK_FREE_RATE = 0.02; DEFAULT_BENCHMARK_TICKER = "SPY"; APP_BASE_URL = "http://localhost:8501"; PASSWORD_RESET_LINK_EXPIRY_HOURS = 1
    # Minimal Base for User model definition if main Base fails
    from sqlalchemy.orm import declarative_base
    Base = declarative_base() # type: ignore
    # Define dummy UserSettings and PasswordResetToken if they couldn't be imported
    if 'UserSettings' not in globals():
        class UserSettings(Base): # type: ignore
            __tablename__ = "user_settings_auth_fb"
            id = Column(Integer, primary_key=True); user_id = Column(Integer, unique=True)
            default_theme=Column(String(50),default="dark"); default_risk_free_rate=Column(Float,default=0.02)
            default_benchmark_ticker=Column(String(50),default="SPY"); last_updated = Column(DateTime, default=dt.datetime.utcnow)
    if 'PasswordResetToken' not in globals():
        class PasswordResetToken(Base): # type: ignore
            __tablename__ = "password_reset_tokens_auth_fb"
            id = Column(Integer, primary_key=True); user_id = Column(Integer); token = Column(String, unique=True)
            expires_at = Column(DateTime); used = Column(Boolean,default=False); created_at = Column(DateTime, default=dt.datetime.utcnow)
    def send_password_reset_email(r,u,l): print(f"Mock email (fb) to {r} with link {l}"); return True
    def get_db_session(): return None # Fallback get_db_session

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
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    settings = relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")
    password_reset_tokens = relationship("PasswordResetToken", backref="user_tokens") # Assuming PasswordResetToken defines 'user_tokens' or similar

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
        try:
            # Basic attributes like id, username, email, hashed_password are loaded by default
            user = db.query(User).filter(User.username == username).first()
            return user
        finally: db.close()

    def get_user_by_email(self, email: str) -> Optional[User]:
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
        if not PASSWORD_REGEX_SPECIAL.search(password): return "Password needs a special character (e.g., !@#$%)."
        return None

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
        db = self._get_db()
        if not db: return {"error": "Database connection error."}
        if not username.strip() or not (email and email.strip()): return {"error": "Username and email cannot be empty."}
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email): return {"error": "Invalid email format."}

        if self.get_user_by_username(username): return {"error": f"Username '{username}' taken."}
        if self.get_user_by_email(email): return {"error": f"Email '{email}' registered."}

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
        initial_user_obj = self.get_user_by_username(username)
        if not initial_user_obj:
            self.logger.warning(f"Auth attempt for non-existent user '{username}'.")
            return None
        if not self.verify_password(password, initial_user_obj.hashed_password):
            self.logger.warning(f"Password verification failed for user '{username}'.")
            return None

        db = self._get_db()
        if not db:
            self.logger.error(f"Failed to get DB session for user '{username}' login update.")
            return None # Critical: cannot update last_login

        try:
            user_for_session = (
                db.query(User)
                .options(selectinload(User.settings)) # Eager load settings
                .filter(User.username == username)
                .first()
            )
            if user_for_session:
                user_for_session.last_login_at = dt.datetime.utcnow()
                db.commit()
                self.logger.info(f"User '{username}' (ID: {user_for_session.id}) authenticated. Last login updated.")
                # The object 'user_for_session' has its attributes (id, username) and 'settings' loaded.
                # These will be accessible even after the session (db) is closed.
                return user_for_session
            else:
                self.logger.error(f"User '{username}' found initially but not in subsequent session query during auth.")
                return None
        except Exception as e:
            db.rollback()
            self.logger.error(f"DB error during final auth steps for '{username}': {e}", exc_info=True)
            return None
        finally:
            db.close()

    def create_password_reset_token(self, email: str) -> Dict[str, Any]:
        db = self._get_db()
        if not db: return {"error": "Database connection error."}
        user = self.get_user_by_email(email)
        if not user:
            self.logger.info(f"Password reset req for non-existent/unverified email: {email}")
            return {"success": "If an account with this email exists, a reset link has been sent."}
        if not user.email: return {"error": "User does not have a valid email on file."}

        token_str = secrets.token_urlsafe(32)
        expires_at = dt.datetime.utcnow() + dt.timedelta(hours=PASSWORD_RESET_LINK_EXPIRY_HOURS)
        new_token_record = PasswordResetToken(user_id=user.id, token=token_str, expires_at=expires_at, used=False)
        try:
            db.query(PasswordResetToken).filter(
                PasswordResetToken.user_id == user.id, PasswordResetToken.used == False,
                PasswordResetToken.expires_at > dt.datetime.utcnow()
            ).update({"used": True, "expires_at": dt.datetime.utcnow() - dt.timedelta(seconds=1)})
            db.add(new_token_record); db.commit(); db.refresh(new_token_record)
            self.logger.info(f"Password reset token created for user_id {user.id}, email {email}.")
            reset_link = f"{APP_BASE_URL}/?page=reset_password_form&token={token_str}"
            email_sent = send_password_reset_email(user.email, user.username, reset_link)
            if email_sent: return {"success": "If an account with this email exists, a reset link has been sent."}
            else:
                self.logger.error(f"Failed to send reset email to {user.email}, token created.")
                return {"error": "Could not send reset email. Try again later."}
        except Exception as e:
            db.rollback(); self.logger.error(f"Error creating reset token for {email}: {e}", exc_info=True)
            return {"error": "Server error creating reset token."}
        finally:
            if db: db.close()

    def verify_password_reset_token(self, token_str: str) -> Optional[int]:
        db = self._get_db()
        if not db: return None
        try:
            token_record = db.query(PasswordResetToken).filter(PasswordResetToken.token == token_str).first()
            if not token_record: self.logger.warning(f"Verify token: Token '{token_str[:8]}...' not found."); return None
            if token_record.used: self.logger.warning(f"Verify token: Token ID {token_record.id} already used."); return None
            if token_record.expires_at < dt.datetime.utcnow(): self.logger.warning(f"Verify token: Token ID {token_record.id} expired."); return None
            self.logger.info(f"Password reset token verified for user_id {token_record.user_id}.")
            return token_record.user_id
        except Exception as e:
            self.logger.error(f"Error verifying reset token: {e}", exc_info=True); return None
        finally:
            if db: db.close()

    def reset_password_with_token(self, token_str: str, new_password: str) -> Dict[str, Any]:
        user_id = self.verify_password_reset_token(token_str)
        if not user_id: return {"error": "Invalid or expired reset token. Request a new one."}
        password_error = self.validate_password_complexity(new_password)
        if password_error: return {"error": password_error}
        db = self._get_db()
        if not db: return {"error": "Database connection error."}
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                self.logger.error(f"User ID {user_id} from token not found during password reset.");
                return {"error": "User account not found."}
            user.hashed_password = self.hash_password(new_password)
            token_record = db.query(PasswordResetToken).filter(PasswordResetToken.token == token_str).first()
            if token_record: token_record.used = True
            db.commit(); self.logger.info(f"Password reset for user_id {user_id}.");
            return {"success": "Password reset successfully."}
        except Exception as e:
            db.rollback(); self.logger.error(f"Error resetting password for user_id {user_id}: {e}", exc_info=True)
            return {"error": "Server error resetting password."}
        finally:
            if db: db.close()

    def get_user_settings(self, user_id: int) -> Optional[UserSettings]:
        db = self._get_db()
        if not db: return None
        try: return db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
        finally: db.close()

    def update_user_settings(self, user_id: int, new_settings_data: Dict[str, Any]) -> Optional[UserSettings]:
        db = self._get_db()
        if not db: return None
        try:
            settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
            if not settings:
                settings = UserSettings(user_id=user_id,
                                       default_theme=new_settings_data.get('default_theme', "dark"),
                                       default_risk_free_rate=new_settings_data.get('default_risk_free_rate', RISK_FREE_RATE),
                                       default_benchmark_ticker=new_settings_data.get('default_benchmark_ticker', DEFAULT_BENCHMARK_TICKER))
                db.add(settings)
            else:
                for key, value in new_settings_data.items():
                    if hasattr(settings, key): setattr(settings, key, value)
            settings.last_updated = dt.datetime.utcnow()
            db.commit(); db.refresh(settings)
            self.logger.info(f"User settings updated for user_id {user_id}: {new_settings_data}")
            return settings
        except Exception as e:
            db.rollback(); self.logger.error(f"Error updating settings for user {user_id}: {e}", exc_info=True)
            return None
        finally:
            if db: db.close()
