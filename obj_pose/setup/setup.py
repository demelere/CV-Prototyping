#!/usr/bin/env python3
"""
Setup script for Depth Estimation application
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def setup_environment():
    """Set up the environment and install dependencies"""
    print("🚀 Setting up Depth Estimation environment...\n")
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: You're not in a virtual environment.")
        print("   It's recommended to create one first:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print()
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("✅ Created logs directory")
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("  python app.py")
    print("\nThe app will be available at: http://localhost:7864")
    
    return True

def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import torch
        
        print("✅ All required packages imported successfully")
        
        # Test if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available, will use CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    if setup_environment():
        test_installation()
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)
