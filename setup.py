#!/usr/bin/env python3
import os
import subprocess
import sys

def install_requirements():
    """Install required Python packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ['templates', 'static/css', 'static/js', 'models', 'simulation', 'data']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def generate_dummy_data():
    """Generate dummy CSV data"""
    try:
        exec(open('create_data.py').read())
        print("✅ Dummy data generated successfully!")
    except Exception as e:
        print(f"❌ Error generating dummy data: {e}")

def main():
    print("🌱 Setting up NeuroSimGreen project...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return
    
    # Generate dummy data
    generate_dummy_data()
    
    print("\n🚀 Setup completed!")
    print("To run the application:")
    print("1. python app.py")
    print("2. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
