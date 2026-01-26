"""Test script to reproduce the config validation error."""
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.config import Config

print("=" * 60)
print("Testing Config.load_config() and validation")
print("=" * 60)
print()

# Get path to .env file
config_dir = Path(__file__).parent
env_file = config_dir / ".." / ".env"

print(f"Loading .env file from: {env_file.resolve()}")
print(f"File exists: {env_file.resolve().exists()}")
print()

# Load .env file manually first to see what we get
load_dotenv(env_file)
base_value = os.getenv("BASE")
print(f"BASE from .env: {base_value}")
print()

# Try to create Path object
if base_value:
    base_path = Path(base_value)
    print(f"base_path object: {base_path}")
    print(f"base_path type: {type(base_path)}")
    print()
    
    # Try to check if it exists (this is where the error occurs)
    print("Attempting to check if base_dir exists...")
    print("-" * 60)
    try:
        exists = base_path.exists()
        print(f"✓ base_dir.exists() = {exists}")
    except OSError as e:
        print(f"✗ OSError occurred: {e}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error number: {e.errno}")
        print(f"  Error message: {e.strerror}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print(f"  Error type: {type(e).__name__}")
    print()

# Now try to load the full config (this will trigger the error)
print("=" * 60)
print("Attempting Config.load_config()...")
print("=" * 60)
try:
    config = Config.load_config()
    print("✓ Config loaded successfully!")
    print(f"  base_dir: {config.base_dir}")
    print(f"  rosbags_dir: {config.rosbags_dir}")
except OSError as e:
    print(f"✗ OSError occurred during config loading:")
    print(f"  Error: {e}")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Error number: {e.errno}")
    print(f"  Error message: {e.strerror}")
    print()
    print("This is the same error you're seeing in main.py")
except ValueError as e:
    print(f"✗ ValueError occurred: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
