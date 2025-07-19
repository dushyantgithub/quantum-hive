#!/usr/bin/env python3
"""
Development startup script for Quantum Hive
Run this from the project root to start the application
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Change to backend directory
os.chdir(backend_dir)

# Import and run main
from main import main

if __name__ == "__main__":
    main() 