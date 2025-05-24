# services/auth_service.py
"""
Service for handling user authentication, registration, and management.
"""
import datetime as dt
from typing import Optional

from sqlalchemy.orm import Session
from passlib.context import CryptContext

# Attempt to import Base and get_db from data_service
# This creates a dependency. A more advanced setup might have a shared 'database.py'
# for Base and session management to avoid service-to-service imports for this.
try:
    from .data_service import Base, get_db_session_cached # Assuming get_db_session_cached returns a session
    from config import APP_TITLE
except ImportError:
    # Fallback for standalone testing or if imports fail during generation
    print("AuthService: Could not import Base or get_db_session_cached from data_service. Using placeholders.")
    APP_TITLE = "TradingDashboard_AuthService_Fallback"
    from sqlalchemy import create_engine, Column, Integer, String, DateTime
    from sqlalchemy.orm import sessionmaker, declarative_base
    Base = declarative_base()
    engine = create_engine("sqlite:///./temp_auth_test.db") # Temporary for fallback
    SessionLocalTest = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    def get_db_session_cached(): return SessionLocalTest()


from sqlalchemy import Column, Integer, String, DateTime

import logging
logger = logging.getLogger(APP_TITLE)

# Password hashing context
# Using pbkdf2_sha256 as a secure hashing algorithm. bcrypt is also a strong choice.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

class User(Base): # type: ignore
    """
    SQLAlchemy User model.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True) # Optional email
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class AuthService:
    """
    Service class for authentication operations.
    """
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        self.logger.info("AuthService initialized.")
        # Note: Table creation is handled by DataService's create_db_tables
        # which should be modified to include the User model.

    def _get_db(self) -> Optional[Session]:
        """Gets a database session from the shared utility."""
        # This relies on get_db_session_cached being available.
        # In a larger app, session management might be passed in or handled by a shared component.
        if get_db_session_cached is None:
            self.logger.error("Database session provider (get_db_session_cached) is not available.")
            return None
        return get_db_session_cached()

    def hash_password(self, password: str) -> str:
        """Hashes a plain password."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifies a plain password against a hashed password."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            self.logger.error(f"Error verifying password: {e}", exc_info=True)
            return False

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Retrieves a user by their username."""
        db = self._get_db()
        if not db:
            return None
        try:
            return db.query(User).filter(User.username == username).first()
        except Exception as e:
            self.logger.error(f"Error fetching user by username '{username}': {e}", exc_info=True)
            return None
        finally:
            db.close() # Or db.remove() if scoped_session

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Retrieves a user by their ID."""
        db = self._get_db()
        if not db:
            return None
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            self.logger.error(f"Error fetching user by ID {user_id}: {e}", exc_info=True)
            return None
        finally:
            db.close()

    def register_user(self, username: str, password: str, email: Optional[str] = None) -> Optional[User]:
        """
        Registers a new user.
        Checks if username or email already exists.
        Hashes the password before storing.
        """
        db = self._get_db()
        if not db:
            self.logger.error("Database session not available for user registration.")
            return None

        if self.get_user_by_username(username):
            self.logger.warning(f"Registration attempt failed: Username '{username}' already exists.")
            return None # Username already exists

        if email:
            # Add a check for existing email if email is meant to be unique and provided
            existing_email_user = db.query(User).filter(User.email == email).first()
            if existing_email_user:
                self.logger.warning(f"Registration attempt failed: Email '{email}' already registered.")
                db.close()
                return None # Email already exists

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
            db.close()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticates a user.
        If successful, updates last_login_at.
        Returns the User object if authentication is successful, otherwise None.
        """
        user = self.get_user_by_username(username) # This method handles its own session
        if not user:
            self.logger.warning(f"Authentication failed: User '{username}' not found.")
            return None

        if not self.verify_password(password, user.hashed_password):
            self.logger.warning(f"Authentication failed: Invalid password for user '{username}'.")
            return None

        # Update last_login_at
        db = self._get_db()
        if db:
            try:
                # Re-fetch user within this session to update
                user_in_session = db.query(User).filter(User.id == user.id).first()
                if user_in_session:
                    user_in_session.last_login_at = dt.datetime.utcnow()
                    db.commit()
                    self.logger.info(f"User '{username}' authenticated successfully. Last login updated.")
                else: # Should not happen if user was fetched initially
                    self.logger.error(f"Could not re-fetch user '{username}' to update last_login_at.")
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error updating last_login_at for user '{username}': {e}", exc_info=True)
            finally:
                db.close()
        else:
            self.logger.error("Database session not available to update last_login_at.")
        
        return user

if __name__ == "__main__":
    # This basic test assumes you run it where config.py and data_service.py are accessible
    # or that the fallback database setup within this file is used.
    # For a real test, you'd mock the DB session or use a test database.
    print("Testing AuthService standalone...")
    
    # Ensure tables are created (including User table if Base is shared)
    # This requires Base to be correctly imported and User model associated with it.
    # If running standalone and data_service.Base is used, data_service.create_db_tables()
    # would need to be called after User model is defined.
    # For this simple test, we'll assume tables might exist or be created by data_service.
    
    # Example: If data_service.py is in the same directory (services/)
    # and defines create_db_tables() that includes the User model.
    try:
        from data_service import create_db_tables as create_all_tables_global
        # The User model class needs to be imported by data_service.py before create_all_tables_global is called
        # or create_all_tables_global needs to be called from a place where all models are imported.
        # For now, we assume create_all_tables_global will handle it if User is on data_service.Base
        # This is tricky for standalone test. A better way is a central db init script.
        
        # Create tables if they don't exist (requires User model to be known by Base.metadata)
        # This is problematic here because data_service.create_db_tables() doesn't know about User model yet
        # unless User model is also defined in data_service or imported there before create_all_tables is called.
        # For now, this test might fail on table creation if not handled carefully in app.py.
        # A simple solution for testing:
        if Base.metadata.tables.get("users") is None and engine is not None: # Check if 'users' table exists
             User.__table__.create(bind=engine, checkfirst=True)
             print("'users' table created for test.")

    except ImportError:
        print("Could not import create_db_tables from data_service for test setup.")
    except Exception as e_test_setup:
        print(f"Error during test setup (table creation): {e_test_setup}")


    auth_service_test = AuthService()
    
    # Test registration
    test_username = "testuser123"
    test_password = "Password123!"
    test_email = "testuser@example.com"

    # Clean up existing test user if any, for idempotency
    existing_user_to_delete = auth_service_test.get_user_by_username(test_username)
    if existing_user_to_delete:
        db_del = auth_service_test._get_db()
        if db_del:
            try:
                user_to_del_sess = db_del.query(User).filter(User.username == test_username).first()
                if user_to_del_sess:
                    db_del.delete(user_to_del_sess)
                    db_del.commit()
                    print(f"Cleaned up existing user: {test_username}")
            except Exception as e_del: print(f"Cleanup error: {e_del}")
            finally: db_del.close()


    registered_user = auth_service_test.register_user(test_username, test_password, test_email)
    if registered_user:
        print(f"User registered: {registered_user.username}, ID: {registered_user.id}")
        
        # Test authentication
        authenticated_user = auth_service_test.authenticate_user(test_username, test_password)
        if authenticated_user:
            print(f"User authenticated: {authenticated_user.username}, Last Login: {authenticated_user.last_login_at}")
        else:
            print(f"Authentication failed for {test_username}.")

        # Test authentication with wrong password
        auth_fail_user = auth_service_test.authenticate_user(test_username, "WrongPassword!")
        if not auth_fail_user:
            print(f"Authentication correctly failed for {test_username} with wrong password.")
            
    else:
        print(f"Registration failed for {test_username}. User might already exist or DB error.")

    # Test non-existent user
    non_existent_user_auth = auth_service_test.authenticate_user("nosuchuser", "anypassword")
    if not non_existent_user_auth:
        print("Authentication correctly failed for non-existent user.")
