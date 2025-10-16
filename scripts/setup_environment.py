#!/usr/bin/env python3
"""
SkinScan Pro - Environment Setup Script
Sets up the complete development environment and downloads required models.
"""

import os
import sys
import subprocess
import urllib.request
import json
from pathlib import Path
from typing import List, Dict, Optional

def run_command(command: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command with error handling."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result

def check_python_version():
    """Ensure Python 3.8+ is being used."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")

def check_system_requirements():
    """Check if required system tools are available."""
    required_tools = ['git', 'npm', 'node']
    missing_tools = []
    
    for tool in required_tools:
        try:
            result = run_command([tool, '--version'], check=False)
            if result.returncode == 0:
                print(f"✓ {tool} found")
            else:
                missing_tools.append(tool)
        except FileNotFoundError:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"Error: Missing required tools: {', '.join(missing_tools)}")
        print("Please install the missing tools and try again.")
        sys.exit(1)

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("\n=== Setting up Python environment ===")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"])
    else:
        print("✓ Virtual environment already exists")
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate"
        python_exe = venv_path / "Scripts" / "python"
        pip_exe = venv_path / "Scripts" / "pip"
    else:  # Unix-like
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Upgrade pip
    print("Upgrading pip...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    if Path("requirements.txt").exists():
        print("Installing Python dependencies...")
        run_command([str(pip_exe), "install", "-r", "requirements.txt"])
    else:
        print("Warning: requirements.txt not found")
    
    print("✓ Python environment setup complete")
    return str(python_exe), str(pip_exe)

def setup_frontend():
    """Set up React frontend dependencies."""
    print("\n=== Setting up Frontend ===")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("Warning: frontend directory not found")
        return
    
    os.chdir(frontend_path)
    
    # Check if package.json exists
    if not Path("package.json").exists():
        print("Initializing frontend project...")
        # Create basic package.json for React + TypeScript + Tailwind
        package_json = {
            "name": "skinscan-pro-frontend",
            "version": "1.0.0",
            "private": True,
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
                "preview": "vite preview"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "lucide-react": "^0.263.1"
            },
            "devDependencies": {
                "@types/react": "^18.2.15",
                "@types/react-dom": "^18.2.7",
                "@typescript-eslint/eslint-plugin": "^6.0.0",
                "@typescript-eslint/parser": "^6.0.0",
                "@vitejs/plugin-react": "^4.0.3",
                "autoprefixer": "^10.4.14",
                "eslint": "^8.45.0",
                "eslint-plugin-react-hooks": "^4.6.0",
                "eslint-plugin-react-refresh": "^0.4.3",
                "postcss": "^8.4.27",
                "tailwindcss": "^3.3.3",
                "typescript": "^5.0.2",
                "vite": "^4.4.5"
            }
        }
        
        with open("package.json", "w") as f:
            json.dump(package_json, f, indent=2)
    
    # Install dependencies
    print("Installing frontend dependencies...")
    run_command(["npm", "install"])
    
    os.chdir("..")
    print("✓ Frontend setup complete")

def create_config_files():
    """Create necessary configuration files."""
    print("\n=== Creating configuration files ===")
    
    # Backend .env file
    backend_env = """# Backend Configuration
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
SECRET_KEY=your-secret-key-change-in-production

# Storage
UPLOAD_MAX_SIZE=10485760  # 10MB
DATA_RETENTION_DAYS=7

# Model Configuration
MODEL_PATH=models/skin_model_ts.pt
MODEL_DEVICE=auto  # auto, cpu, cuda

# Database (optional)
DATABASE_URL=sqlite:///./skinscan.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/skinscan.log
"""
    
    backend_path = Path("backend")
    backend_path.mkdir(exist_ok=True)
    
    with open(backend_path / ".env", "w") as f:
        f.write(backend_env)
    
    # Frontend config
    frontend_config = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
"""
    
    frontend_path = Path("frontend")
    frontend_path.mkdir(exist_ok=True)
    
    with open(frontend_path / "vite.config.ts", "w") as f:
        f.write(frontend_config)
    
    # Create tailwind config
    tailwind_config = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""
    
    with open(frontend_path / "tailwind.config.js", "w") as f:
        f.write(tailwind_config)
    
    print("✓ Configuration files created")

def download_sample_models():
    """Download or create sample model files."""
    print("\n=== Setting up model files ===")
    
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    
    # Create model metadata file
    metadata = {
        "model_type": "efficientnet_b3",
        "input_size": [3, 384, 384],
        "num_conditions": 6,
        "condition_labels": [
            "acne",
            "hyperpigmentation", 
            "redness",
            "dehydration",
            "pore_size",
            "fine_lines"
        ],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "version": "1.0.0",
        "created": "2024-01-01T00:00:00Z"
    }
    
    with open(models_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Model metadata created")
    print("Note: Trained model files need to be added manually after training")

def create_sample_images():
    """Create placeholder sample images."""
    print("\n=== Creating sample data ===")
    
    # This would normally download or generate sample images
    # For now, just create the directory structure
    images_path = Path("dataset/images")
    images_path.mkdir(parents=True, exist_ok=True)
    
    masks_path = Path("dataset/masks") 
    masks_path.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files
    for i in range(1, 11):
        placeholder_path = images_path / f"SYN{i:02d}.jpg"
        if not placeholder_path.exists():
            # Create a minimal placeholder file
            with open(placeholder_path, "wb") as f:
                # This would be a real image file in practice
                f.write(b"PLACEHOLDER_IMAGE_DATA")
    
    print("✓ Sample data structure created")
    print("Note: Replace placeholder files with actual images for training")

def verify_installation():
    """Verify that the installation was successful."""
    print("\n=== Verifying installation ===")
    
    success = True
    
    # Check Python packages
    try:
        import torch
        import fastapi
        print("✓ Core Python packages installed")
    except ImportError as e:
        print(f"✗ Missing Python package: {e}")
        success = False
    
    # Check directory structure
    required_dirs = [
        "backend", "frontend", "ml", "dataset", "models", "scripts"
    ]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            success = False
    
    # Check key files
    key_files = [
        "requirements.txt",
        "dataset/annotations.jsonl",
        "backend/app.py",
        "ml/train_finetune.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            success = False
    
    if success:
        print("\n✅ Installation verification successful!")
        print("\nNext steps:")
        print("1. Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        print("2. Start backend: cd backend && python app.py")
        print("3. Start frontend: cd frontend && npm run dev")
        print("4. Visit http://localhost:3000 to use the app")
    else:
        print("\n❌ Installation verification failed. Please check the errors above.")
    
    return success

def main():
    """Main setup function."""
    print("=== SkinScan Pro Environment Setup ===\n")
    
    try:
        check_python_version()
        check_system_requirements()
        
        python_exe, pip_exe = setup_python_environment()
        setup_frontend()
        create_config_files()
        download_sample_models()
        create_sample_images()
        
        verify_installation()
        
        print("\n🎉 Setup complete! Ready to start developing.")
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()