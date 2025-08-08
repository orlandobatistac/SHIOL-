"""
SHIOL+ Authentication System
============================

Database-based authentication system for administrative users
and prepared for future expansion to paid users.

TEMPORARY NOTE: Credentials are hardcoded for initial development.
TODO: Migrate to environment variables before production deployment.
"""

import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
import bcrypt
import os

@dataclass
class User:
    """Class to represent a system user."""
    id: int
    username: str
    email: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    subscription_type: str = 'free'
    subscription_expires: Optional[datetime] = None

@dataclass
class UserSession:
    """Class to represent a user session."""
    id: int
    user_id: int
    session_token: str
    expires_at: datetime
    created_at: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]

class AuthManager:
    """
    Authentication manager that handles users, sessions and permissions.
    """
    
    def __init__(self, db_path: str = "data/shiolplus.db"):
        """
        Initialize the authentication manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_auth_tables()
        self._create_default_admin()
    
    def _create_auth_tables(self):
        """Create authentication tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        email VARCHAR(100),
                        role VARCHAR(20) DEFAULT 'admin',
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        subscription_type VARCHAR(20) DEFAULT 'free',
                        subscription_expires TIMESTAMP,
                        api_key VARCHAR(100),
                        rate_limit_per_hour INTEGER DEFAULT 100
                    )
                """)
                
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_token VARCHAR(255) UNIQUE NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ip_address VARCHAR(45),
                        user_agent TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.commit()
                logger.info("Authentication tables created successfully")
                
        except Exception as e:
            logger.error(f"Error creating authentication tables: {e}")
            raise
    
    def _create_default_admin(self):
        """Create default admin user if no users exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if any users exist
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                if user_count == 0:
                    # TEMPORARY HARDCODED CREDENTIALS - TODO: Move to environment variables
                    default_username = "admin"
                    default_password = "shiol2024!"  # TODO: Change this!
                    
                    password_hash = bcrypt.hashpw(
                        default_password.encode('utf-8'), 
                        bcrypt.gensalt()
                    ).decode('utf-8')
                    
                    cursor.execute("""
                        INSERT INTO users (username, password_hash, email, role, is_active)
                        VALUES (?, ?, ?, ?, ?)
                    """, (default_username, password_hash, "admin@shiolplus.com", "admin", True))
                    
                    conn.commit()
                    logger.info(f"Default admin user created: {default_username}")
                    logger.warning("SECURITY WARNING: Default credentials are hardcoded. Change before production!")
                
        except Exception as e:
            logger.error(f"Error creating default admin user: {e}")
            raise
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, username, password_hash, email, role, is_active, 
                           created_at, last_login, subscription_type, subscription_expires
                    FROM users 
                    WHERE username = ? AND is_active = 1
                """, (username,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Authentication failed: user '{username}' not found or inactive")
                    return None
                
                # Verify password
                stored_hash = row[2].encode('utf-8')
                if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                    logger.warning(f"Authentication failed: invalid password for user '{username}'")
                    return None
                
                # Update last login
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                """, (row[0],))
                conn.commit()
                
                # Create User object
                user = User(
                    id=row[0],
                    username=row[1],
                    email=row[3],
                    role=row[4],
                    is_active=bool(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    last_login=datetime.now(),
                    subscription_type=row[8] or 'free',
                    subscription_expires=datetime.fromisoformat(row[9]) if row[9] else None
                )
                
                logger.info(f"User '{username}' authenticated successfully")
                return user
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return None
    
    def create_session(self, user: User, ip_address: Optional[str] = None, 
                      user_agent: Optional[str] = None, 
                      expires_hours: int = 24) -> Optional[str]:
        """
        Create a new session for authenticated user.
        
        Args:
            user: Authenticated user
            ip_address: Client IP address
            user_agent: Client user agent
            expires_hours: Session expiration in hours
            
        Returns:
            Session token if successful, None otherwise
        """
        try:
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO user_sessions (user_id, session_token, expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?)
                """, (user.id, session_token, expires_at.isoformat(), ip_address, user_agent))
                
                conn.commit()
                
                logger.info(f"Session created for user '{user.username}' (expires: {expires_at})")
                return session_token
                
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def verify_session(self, session_token: str) -> Optional[User]:
        """
        Verify a session token and return the associated user.
        
        Args:
            session_token: Session token to verify
            
        Returns:
            User object if session is valid, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT s.user_id, s.expires_at, u.username, u.email, u.role, 
                           u.is_active, u.created_at, u.last_login, u.subscription_type, u.subscription_expires
                    FROM user_sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.session_token = ? AND u.is_active = 1
                """, (session_token,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Check if session has expired
                expires_at = datetime.fromisoformat(row[1])
                if datetime.now() > expires_at:
                    # Clean up expired session
                    cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
                    conn.commit()
                    logger.info("Expired session cleaned up")
                    return None
                
                # Create User object
                user = User(
                    id=row[0],
                    username=row[2],
                    email=row[3],
                    role=row[4],
                    is_active=bool(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                    subscription_type=row[8] or 'free',
                    subscription_expires=datetime.fromisoformat(row[9]) if row[9] else None
                )
                
                return user
                
        except Exception as e:
            logger.error(f"Error verifying session: {e}")
            return None
    
    def logout_session(self, session_token: str) -> bool:
        """
        Logout a session by removing it from database.
        
        Args:
            session_token: Session token to logout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info("Session logged out successfully")
                    return True
                else:
                    logger.warning("Session not found for logout")
                    return False
                    
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM user_sessions WHERE expires_at < ?", (datetime.now().isoformat(),))
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired sessions")
                    
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")

# Global auth manager instance
auth_manager = AuthManager()

# FastAPI dependency functions
from fastapi import Depends, HTTPException, status, Cookie
from typing import Optional

async def get_current_user(session_token: Optional[str] = Cookie(None, alias="session_token")) -> User:
    """
    FastAPI dependency to get current authenticated user from session cookie.
    
    Args:
        session_token: Session token from cookie
        
    Returns:
        User object if authenticated
        
    Raises:
        HTTPException: If not authenticated
    """
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated - no session token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = auth_manager.verify_session(session_token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_optional_user(session_token: Optional[str] = Cookie(None, alias="session_token")) -> Optional[User]:
    """
    FastAPI dependency to get current user if authenticated, None otherwise.
    This is for optional authentication.
    
    Args:
        session_token: Session token from cookie
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not session_token:
        return None
    
    return auth_manager.verify_session(session_token)