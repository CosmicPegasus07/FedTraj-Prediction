#!/usr/bin/env python3
"""
Simple cache management utility for PyG data
"""

import argparse
from utils.data_utils import get_cache_info, clear_cache

def main():
    parser = argparse.ArgumentParser(description="Cache management utility")
    parser.add_argument("command", choices=["info", "clear", "status"],
                       help="Command to execute")
    
    args = parser.parse_args()
    
    if args.command == "info":
        cache_info = get_cache_info()
        print("=== Cache Information ===")
        print(f"Cached files: {cache_info['cached_files']}")
        print(f"Cache size: {cache_info['cache_size_mb']:.1f} MB")
        
        if cache_info['cached_files'] > 0:
            avg_size = cache_info['cache_size_mb'] / cache_info['cached_files']
            print(f"Average file size: {avg_size:.2f} MB")
            print(f"Estimated scenarios: {cache_info['cached_files']}")
    
    elif args.command == "clear":
        print("Clearing cache...")
        clear_cache()
        print("✅ Cache cleared successfully!")
    
    elif args.command == "status":
        cache_info = get_cache_info()
        if cache_info['cached_files'] > 0:
            print(f"✅ Cache is active with {cache_info['cached_files']} files ({cache_info['cache_size_mb']:.1f} MB)")
            print("💡 Subsequent data loading will be faster!")
        else:
            print("ℹ️  Cache is empty")
            print("💡 First data loading will build the cache")

if __name__ == "__main__":
    main() 