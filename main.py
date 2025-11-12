#!/usr/bin/env python3
"""
Entry point wrapper for Philippine Property Price Prediction Neural Network

This script allows running the application from the project root directory.
It simply imports and executes the main script from the src/ directory.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main function
if __name__ == '__main__':
    # Import from src.main module to avoid circular import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "src_main", 
        os.path.join(os.path.dirname(__file__), 'src', 'main.py')
    )
    src_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(src_main)
    
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        print("Running in DEMO mode (reduced training)...")
        model = src_main.quick_demo()
    else:
        print("Running in FULL mode (complete pipeline)...")
        model = src_main.main()
