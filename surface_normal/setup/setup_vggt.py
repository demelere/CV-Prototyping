#!/usr/bin/env python3
"""
VGGT Installation Script

This script installs VGGT (Visual Geometry Grounded Transformer) with proper PyTorch dependencies.
It follows the recommended installation process from the VGGT documentation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return False

def main():
    print("VGGT Installation Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    vggt_dir = current_dir / "vggt"
    
    if not vggt_dir.exists():
        print(f"ERROR: VGGT directory not found at {vggt_dir}")
        print("Please run this script from the directory containing the vggt folder.")
        sys.exit(1)
    
    print(f"Found VGGT directory at: {vggt_dir}")
    
    # Step 1: Install PyTorch and torchvision first to avoid CUDA version mismatches
    print("\nStep 1: Installing PyTorch and torchvision...")
    pytorch_command = "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121"
    
    if not run_command(pytorch_command, "Installing PyTorch and torchvision"):
        print("Failed to install PyTorch. Trying CPU version...")
        pytorch_cpu_command = "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu"
        if not run_command(pytorch_cpu_command, "Installing PyTorch CPU version"):
            print("ERROR: Failed to install PyTorch. Exiting.")
            sys.exit(1)
    
    # Step 2: Install other dependencies
    print("\nStep 2: Installing other dependencies...")
    deps_command = "pip install numpy==1.26.1 Pillow huggingface_hub einops safetensors"
    
    if not run_command(deps_command, "Installing additional dependencies"):
        print("ERROR: Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Step 3: Install VGGT as a package
    print("\nStep 3: Installing VGGT as a package...")
    os.chdir(vggt_dir)
    
    if not run_command("pip install -e .", "Installing VGGT package"):
        print("ERROR: Failed to install VGGT package. Exiting.")
        sys.exit(1)
    
    # Return to original directory
    os.chdir(current_dir)
    
    # Step 4: Verify installation
    print("\nStep 4: Verifying installation...")
    verify_command = 'python -c "import vggt; print(f\'VGGT version: {vggt.__version__ if hasattr(vggt, \"__version__\") else \"unknown\"}\'); print(\'VGGT imported successfully!\')"'
    
    if run_command(verify_command, "Verifying VGGT installation"):
        print("\n" + "="*50)
        print("INSTALLATION COMPLETED SUCCESSFULLY!")
        print("VGGT is now installed and ready to use.")
        print("="*50)
    else:
        print("\nWARNING: Installation completed but verification failed.")
        print("You may need to check your Python environment.")

if __name__ == "__main__":
    main()