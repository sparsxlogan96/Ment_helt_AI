#!/usr/bin/env python3
"""
Setup script for Mental Health AI project.
This script helps set up the development environment in PyCharm or any other IDE.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8 or higher is required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    if os.path.exists("venv") or os.path.exists(".venv"):
        print("✅ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    return run_command(
        f"{sys.executable} -m venv venv",
        "Creating Python virtual environment"
    )


def install_dependencies():
    """Install required dependencies."""
    pip_command = "venv/bin/pip" if os.path.exists("venv/bin/pip") else "pip"
    
    commands = [
        (f"{pip_command} install --upgrade pip", "Upgrading pip"),
        (f"{pip_command} install -r requirements.txt", "Installing project dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True


def verify_installation():
    """Verify that key packages are installed."""
    python_command = "venv/bin/python" if os.path.exists("venv/bin/python") else sys.executable
    
    packages = ["streamlit", "transformers", "torch", "faiss"]
    
    print("\n" + "="*60)
    print("Verifying installation")
    print("="*60)
    
    all_installed = True
    for package in packages:
        try:
            result = subprocess.run(
                f"{python_command} -c 'import {package}; print({package}.__version__ if hasattr({package}, \"__version__\") else \"installed\")'",
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
            version = result.stdout.strip()
            print(f"✅ {package}: {version}")
        except subprocess.CalledProcessError:
            print(f"❌ {package}: Not installed or import error")
            all_installed = False
    
    return all_installed


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. In PyCharm:")
    print("   - Open the project (File > Open > select this directory)")
    print("   - Configure Python interpreter:")
    print("     • Go to Settings/Preferences > Project > Python Interpreter")
    print("     • Click gear icon > Add > Existing environment")
    print("     • Select: <project-dir>/venv/bin/python")
    print("   - Run the app using the 'Streamlit App (module)' run configuration")
    print("     or click the green play button in the toolbar")
    print("\n2. From command line:")
    print("   - Activate virtual environment:")
    print("     • On Linux/Mac: source venv/bin/activate")
    print("     • On Windows: venv\\Scripts\\activate")
    print("   - Run the app: streamlit run app.py")
    print("\n3. Optional - Fine-tune GPT-2 model:")
    print("   - Run: python fine_tune_gpt2.py --data-dir mental_health_data --output-dir ./fine_tuned_model")
    print("   - Update app.py to use the fine-tuned model")
    print("\n4. Access the app:")
    print("   - Open your browser to: http://localhost:8501")
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("="*60)
    print("Mental Health AI - Development Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("\n❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages may not be properly installed")
        print("Try running: pip install -r requirements.txt")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
