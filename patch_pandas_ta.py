#!/usr/bin/env python3
"""
Script to patch pandas_ta package for compatibility with newer numpy versions.
The issue is that pandas_ta tries to import `NaN` from numpy, but newer versions of numpy use `nan`.
This script creates a symlink or copies the file with a fix.
"""

import os
import sys
import importlib.util
import site
from pathlib import Path


def find_package_path(package_name):
    """Find the installation path of a package."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.origin is None:
        return None
    return Path(spec.origin).parent


def patch_pandas_ta():
    """Patch the pandas_ta package to work with newer numpy versions."""
    print("Patching pandas_ta for numpy compatibility...")
    
    # Find pandas_ta package path
    pkg_path = find_package_path("pandas_ta")
    if pkg_path is None:
        print("pandas_ta package not found.")
        return False
    
    # Path to the problematic file
    momentum_dir = pkg_path / "momentum"
    squeeze_pro_file = momentum_dir / "squeeze_pro.py"
    
    if not squeeze_pro_file.exists():
        print(f"File not found: {squeeze_pro_file}")
        return False
    
    # Create backup
    backup_file = squeeze_pro_file.with_suffix(".py.bak")
    if not backup_file.exists():
        print(f"Creating backup: {backup_file}")
        with open(squeeze_pro_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
    
    # Read the file content
    with open(squeeze_pro_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic import
    if "from numpy import NaN as npNaN" in content:
        content = content.replace(
            "from numpy import NaN as npNaN",
            "from numpy import nan as npNaN  # Fixed import for newer numpy versions"
        )
        
        # Write the patched content
        with open(squeeze_pro_file, 'w') as f:
            f.write(content)
        
        print(f"Successfully patched {squeeze_pro_file}")
        return True
    else:
        print("The file doesn't contain the expected import statement.")
        return False


if __name__ == "__main__":
    success = patch_pandas_ta()
    sys.exit(0 if success else 1)