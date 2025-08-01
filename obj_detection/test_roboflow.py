#!/usr/bin/env python3
"""
Test script to verify Roboflow API connection and model access.
Run this before using the main app to ensure everything is configured correctly.
"""

from roboflow import Roboflow
import os

def test_roboflow_connection(api_key, workspace_name, project_name, version_number):
    """Test the Roboflow connection and model access"""
    
    print("🔍 Testing Roboflow Connection...")
    print("=" * 50)
    
    try:
        # Initialize Roboflow
        print(f"1. Initializing Roboflow with API key...")
        rf = Roboflow(api_key=api_key)
        print("✅ Roboflow initialized successfully")
        
        # Test workspace access
        print(f"2. Testing workspace access: {workspace_name}")
        workspace = rf.workspace(workspace_name)
        print("✅ Workspace access successful")
        
        # Test project access
        print(f"3. Testing project access: {project_name}")
        project = workspace.project(project_name)
        print("✅ Project access successful")
        
        # Test model access
        print(f"4. Testing model access (version {version_number})")
        model = project.version(version_number).model
        print("✅ Model access successful")
        
        # Test model info
        print("5. Getting model information...")
        model_info = project.version(version_number).model
        print(f"✅ Model loaded: {type(model_info).__name__}")
        
        print("\n🎉 All tests passed! Your Roboflow configuration is working correctly.")
        print("\nYou can now use the main app with these settings:")
        print(f"   - API Key: {'*' * len(api_key)}")
        print(f"   - Workspace: {workspace_name}")
        print(f"   - Project: {project_name}")
        print(f"   - Version: {version_number}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your API key is correct")
        print("2. Verify workspace and project names match your Roboflow URL")
        print("3. Ensure the version number exists")
        print("4. Make sure your model is trained and deployed")
        return False

def main():
    """Main function to run the test"""
    print("🧪 Roboflow Configuration Test")
    print("=" * 50)
    
    # Get configuration from user
    api_key = input("Enter your Roboflow API key: ").strip()
    workspace_name = input("Enter your workspace name: ").strip()
    project_name = input("Enter your project name: ").strip()
    version_number = input("Enter your model version number (default: 1): ").strip() or "1"
    
    print("\n" + "=" * 50)
    
    # Run the test
    success = test_roboflow_connection(api_key, workspace_name, project_name, version_number)
    
    if success:
        print("\n🚀 Ready to use the main app!")
        print("Run: python app.py")
    else:
        print("\n⚠️  Please fix the issues above before using the main app.")

if __name__ == "__main__":
    main() 