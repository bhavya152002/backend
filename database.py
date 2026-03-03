import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv
import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Configure SQLAlchemy with optimized pool for multi-threaded access
engine = create_engine(
    DATABASE_URL,
    pool_size=20,          # Increase base pool size
    pool_recycle=1800,     # Recycle connections every 30 mins
    pool_pre_ping=True     # Check connection validity before use
)

from sqlalchemy.orm.attributes import flag_modified

# Use scoped_session for thread-safety in Flask
from sqlalchemy.orm import scoped_session
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String) # Hashed password
    role = Column(String, default="viewer")
    assigned_streams = Column(JSON, default=[]) # List of stream names/indices
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    streams = relationship("Stream", back_populates="owner")

class Stream(Base):
    __tablename__ = "streams"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("users.id"))
    stream_url = Column(String)
    stream_name = Column(String)
    motion_detection_enabled = Column(Boolean, default=True)
    detection_region = Column(JSON, default={"grid_size": 16, "matrix": [[1]*16]*16})
    recording_enabled = Column(Boolean, default=False)
    last_recording_started = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    owner = relationship("User", back_populates="streams")
    recordings = relationship("Recording", back_populates="stream")

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source = Column(String)
    message = Column(String)
    type = Column(String) # critical, warning
    animals = Column(JSON) # e.g. {"Elephant": 1}
    raw_timestamp = Column(Float) # The float timestamp for sorting
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image_url = Column(String, nullable=True)
    clip_url = Column(String, nullable=True)

class Recording(Base):
    __tablename__ = "recordings"
    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(Integer, ForeignKey("streams.id"))
    client_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    type = Column(String) # "manual", "detection"
    storage_url = Column(String) # Cloud Storage link (GCS)
    file_name = Column(String)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    stream = relationship("Stream", back_populates="recordings")

# Create tables if they don't exist
try:
    Base.metadata.create_all(bind=engine)
    print("[Database] Connected to PostgreSQL and tables ensured.")
except Exception as e:
    print(f"[Database] Connection failed: {e}")

def get_db():
    """Returns a thread-local session. Must be closed or removed after use."""
    return SessionLocal()

def get_db_session():
    """Generator for use with context managers"""
    db = SessionLocal()
    try:
        yield db
    finally:
        # For scoped sessions, remove() is better to return to pool
        SessionLocal.remove()
