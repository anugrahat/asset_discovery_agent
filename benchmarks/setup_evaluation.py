#!/usr/bin/env python3
"""
Setup script for drug repurposing paper evaluation.
Creates necessary directories and validates environment.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Ensure Python >= 3.8"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def setup_directories():
    """Create necessary directories"""
    dirs = ['results', 'figures', 'data']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")

def check_openai_key():
    """Verify OpenAI API key is set"""
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set - LLM components will fail")
        print("   Set with: export OPENAI_API_KEY='your-key'")
        return False
    print("✅ OpenAI API key configured")
    return True

def install_requirements():
    """Install Python dependencies"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install dependencies:")
        print(e.stderr.decode())
        return False

def validate_imports():
    """Test critical imports"""
    critical_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    
    failed = []
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package} import successful")
        except ImportError:
            failed.append(package)
            print(f"❌ {package} import failed")
    
    return len(failed) == 0

def create_sample_config():
    """Create sample configuration file"""
    config = """# Evaluation Configuration
# Modify these settings as needed

DISEASES = [
    "Alzheimer's disease",
    "pancreatic cancer", 
    "type 2 diabetes",
    "hypertension"
]

EVALUATION_SETTINGS = {
    "k_values": [5, 10, 20],
    "timeout_seconds": 30,
    "max_retries": 3,
    "cache_results": True
}

API_SETTINGS = {
    "rate_limit_delay": 0.1,  # seconds between API calls
    "batch_size": 10,
    "enable_caching": True
}
"""
    
    with open('config.py', 'w') as f:
        f.write(config)
    print("✅ Created config.py")

def main():
    """Main setup routine"""
    print("🚀 Setting up Drug Repurposing Paper Evaluation")
    print("=" * 50)
    
    # Validation steps
    check_python_version()
    setup_directories()
    openai_ok = check_openai_key()
    
    # Installation
    print("\n📦 Installing dependencies...")
    deps_ok = install_requirements()
    
    if deps_ok:
        print("\n🔍 Validating imports...")
        imports_ok = validate_imports()
    else:
        imports_ok = False
    
    # Configuration
    create_sample_config()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    status_items = [
        ("Python Environment", "✅"),
        ("Directory Structure", "✅"),
        ("OpenAI API Key", "✅" if openai_ok else "⚠️"),
        ("Dependencies", "✅" if deps_ok else "❌"),
        ("Package Imports", "✅" if imports_ok else "❌"),
        ("Configuration", "✅")
    ]
    
    all_good = all([openai_ok, deps_ok, imports_ok])
    
    for item, status in status_items:
        print(f"{status} {item}")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 Setup complete! Ready to run evaluation:")
        print("   python run_evaluation.py")
    else:
        print("❌ Setup incomplete. Please fix issues above.")
        if not openai_ok:
            print("   - Set OpenAI API key for LLM features")
        if not deps_ok:
            print("   - Install missing dependencies")
        if not imports_ok:
            print("   - Resolve import errors")
    
    print("\n📚 Next steps:")
    print("   1. Review config.py settings")
    print("   2. Run: python run_evaluation.py")
    print("   3. Generate figures: python make_figures.py")
    print("   4. Paper ready for submission! 🚀")

if __name__ == "__main__":
    main()
