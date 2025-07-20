#!/usr/bin/env python3
"""
Project utility functions - Template for future utility files
"""

import argparse
import os
import sys
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def get_results_dir():
    """Get the results directory path"""
    return get_project_root() / "results"

def get_cache_dir():
    """Get the cache directory path"""
    return get_project_root() / "cache"

def list_project_files():
    """List all important project files and directories"""
    root = get_project_root()
    
    print("=== Project Structure ===")
    print(f"Project root: {root}")
    
    important_items = [
        "app.py",
        "train.py", 
        "run_demo.py",
        "automated_training.py",
        "utils/",
        "web/",
        "results/",
        "dataset/",
        "models/",
        "federated/"
    ]
    
    for item in important_items:
        item_path = root / item
        if item_path.exists():
            if item_path.is_dir():
                print(f"📁 {item}/")
            else:
                print(f"📄 {item}")
        else:
            print(f"❌ {item} (missing)")

def check_system_status():
    """Check the overall system status"""
    print("=== System Status ===")
    
    # Check cache
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob("*.pkl"))
        print(f"✅ Cache directory exists with {len(cache_files)} files")
    else:
        print("ℹ️  Cache directory does not exist")
    
    # Check results
    results_dir = get_results_dir()
    if results_dir.exists():
        result_files = list(results_dir.rglob("*.json"))
        print(f"✅ Results directory exists with {len(result_files)} JSON files")
    else:
        print("ℹ️  Results directory does not exist")
    
    # Check web interface
    web_dir = get_project_root() / "web"
    if web_dir.exists():
        html_files = list(web_dir.glob("*.html"))
        print(f"✅ Web interface exists with {len(html_files)} HTML files")
    else:
        print("❌ Web interface missing")

def main():
    parser = argparse.ArgumentParser(description="Project utility functions")
    parser.add_argument("command", choices=["list", "status", "help"],
                       help="Command to execute")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_project_files()
    
    elif args.command == "status":
        check_system_status()
    
    elif args.command == "help":
        print("=== Project Utilities ===")
        print("Available commands:")
        print("  list   - List project files and structure")
        print("  status - Check system status")
        print("  help   - Show this help message")
        print("\nUsage examples:")
        print("  python -m utils.project_utils list")
        print("  python -m utils.project_utils status")
        print("  python -m utils.cache_manager info")
        print("  python -m utils.cache_manager clear")

if __name__ == "__main__":
    main() 