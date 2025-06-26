# debug_server_conditions.py
# This script tests the exact same conditions as when the FastAPI server starts

import os
import sys
from pathlib import Path
import sqlite3

# Test from both project root and src/ directory
print("=== TESTING FROM PROJECT ROOT ===")
print(f"Current working directory: {os.getcwd()}")

# Test the config.py import from here
try:
    sys.path.insert(0, 'src')
    from config import settings
    print(f"✅ Config imported successfully")
    print(f"🔧 DATABASE_URL: {settings.DATABASE_URL}")
    print(f"🔧 BASE_DIR: {settings.BASE_DIR}")
    print(f"🔧 DATA_DIR: {settings.DATA_DIR}")
    
    # Test SQLite connection with the exact URL from config
    db_url = settings.DATABASE_URL
    db_path = db_url.replace("sqlite:///", "")
    print(f"🔧 Extracted DB path: {db_path}")
    print(f"🔧 DB path exists: {os.path.exists(db_path)}")
    
    # Try SQLAlchemy connection (like in your app)
    try:
        from sqlalchemy import create_engine
        engine = create_engine(settings.DATABASE_URL)
        connection = engine.connect()
        connection.close()
        print("✅ SQLAlchemy connection successful from project root")
    except Exception as e:
        print(f"❌ SQLAlchemy connection failed from project root: {e}")
        
except Exception as e:
    print(f"❌ Config import failed from project root: {e}")

print("\n" + "="*60)
print("=== TESTING FROM SRC/ DIRECTORY (LIKE SERVER) ===")

# Change to src directory (like when server starts)
os.chdir('src')
print(f"Changed working directory to: {os.getcwd()}")

# Remove old import to force fresh import
if 'config' in sys.modules:
    del sys.modules['config']
if 'settings' in locals():
    del settings

try:
    from config import settings
    print(f"✅ Config imported successfully from src/")
    print(f"🔧 DATABASE_URL: {settings.DATABASE_URL}")
    print(f"🔧 BASE_DIR: {settings.BASE_DIR}")
    print(f"🔧 DATA_DIR: {settings.DATA_DIR}")
    
    # Test SQLite connection with the exact URL from config
    db_url = settings.DATABASE_URL
    db_path = db_url.replace("sqlite:///", "")
    print(f"🔧 Extracted DB path: {db_path}")
    print(f"🔧 DB path exists: {os.path.exists(db_path)}")
    print(f"🔧 DB path absolute: {os.path.abspath(db_path)}")
    
    # Try direct SQLite connection
    try:
        conn = sqlite3.connect(db_path)
        conn.close()
        print("✅ Direct SQLite connection successful from src/")
    except Exception as e:
        print(f"❌ Direct SQLite connection failed from src/: {e}")
    
    # Try SQLAlchemy connection (like in your app)
    try:
        from sqlalchemy import create_engine
        engine = create_engine(settings.DATABASE_URL)
        connection = engine.connect()
        connection.close()
        print("✅ SQLAlchemy connection successful from src/")
    except Exception as e:
        print(f"❌ SQLAlchemy connection failed from src/: {e}")
        
    # Test if we can import and use the database module
    try:
        from models.database import create_tables, engine
        print("✅ Database models imported successfully")
        
        # Try to create tables (this is what fails in your server)
        from models.database import Base
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully from src/")
        
    except Exception as e:
        print(f"❌ Database table creation failed from src/: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Config import failed from src/: {e}")
    import traceback
    traceback.print_exc()

print("\n=== VERIFICATION COMPLETE ===")