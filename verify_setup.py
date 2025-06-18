#!/usr/bin/env python3
"""
Setup Verification Script
Verifies that all files are in correct locations
Author: AdsTable
Created: 2025-06-18 07:46:34 UTC
"""

import sys
from pathlib import Path

def verify_file_structure():
    """Verify that all files are in correct locations"""
    
    required_files = [
        "src/data/provider_links.py",
        "src/core/extraction_engine.py", 
        "src/core/database_manager.py",
        "src/core/service_explorer.py",
        "database/migrations/002_add_extraction_method.sql"
    ]
    
    print("🔍 Verifying Norwegian Service Providers System file structure...")
    print(f"📅 Date: 2025-06-18 07:46:34 UTC")
    print(f"👤 User: AdsTable")
    print("=" * 60)
    
    all_good = True
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - MISSING!")
            all_good = False
    
    print("=" * 60)
    
    if all_good:
        print("🎉 All files are in correct locations!")
        print("🚀 Ready to run: python -m src.core.service_explorer")
        return True
    else:
        print("⚠️  Some files are missing. Please check the file placement.")
        return False

if __name__ == "__main__":
    success = verify_file_structure()
    sys.exit(0 if success else 1)