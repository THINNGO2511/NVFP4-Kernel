#!/bin/bash

# NVFP4 Kernel Hackathon - Environment Setup Script
# This script helps you set up your development environment

echo "=========================================="
echo "NVFP4 Kernel Hackathon - Setup Script"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS"
    echo "   You may need to adjust commands for your OS"
    echo ""
fi

# Create project structure
echo "📁 Creating project structure..."
mkdir -p docs/notes
mkdir -p kernels/naive
mkdir -p kernels/optimized  
mkdir -p kernels/final
mkdir -p tests
mkdir -p benchmarks

echo "✓ Project directories created"
echo ""

# Check for Python
echo "🐍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi
echo ""

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  venv already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate and install dependencies
echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install torch numpy > /dev/null 2>&1
echo "✓ PyTorch and NumPy installed"
echo ""

# Check for Popcorn CLI
echo "🍿 Checking for Popcorn CLI..."
if command -v popcorn-cli &> /dev/null; then
    echo "✓ Popcorn CLI found!"
    popcorn-cli --version
else
    echo "❌ Popcorn CLI not found"
    echo ""
    echo "To install Popcorn CLI:"
    echo "1. Go to: https://github.com/gpu-mode/popcorn-cli/releases"
    echo "2. Download the latest release for your platform"
    echo "3. Run:"
    echo "   chmod +x popcorn-cli"
    echo "   sudo mv popcorn-cli /usr/local/bin/"
    echo ""
fi

# Check for Popcorn CLI configuration
if [ -f "$HOME/.popcorn.yaml" ]; then
    echo "✓ Popcorn CLI is configured"
else
    echo "⚠️  Popcorn CLI not configured yet"
    echo ""
    echo "To configure:"
    echo "1. Join GPU MODE Discord: https://discord.gg/gpumode"
    echo "2. In #nvidia-competition, type: /get-api-url"
    echo "3. Run: export POPCORN_API_URL=\"your-url\""
    echo "4. Run: popcorn-cli register discord"
    echo ""
fi

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "🔄 Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi
echo ""

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# macOS
.DS_Store

# Local experiments
experiments/
scratch/
*.local.*

# Jupyter
.ipynb_checkpoints/

# Logs
*.log

# Popcorn CLI (keep config private)
.popcorn.yaml
EOF
echo "✓ .gitignore created"
echo ""

# Create initial commit
if [ ! -f ".git/refs/heads/main" ] && [ ! -f ".git/refs/heads/master" ]; then
    echo "📝 Creating initial commit..."
    git add .
    git commit -m "Initial project setup - NVFP4 Kernel Hackathon" > /dev/null 2>&1
    echo "✓ Initial commit created"
fi
echo ""

# Summary
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. If Popcorn CLI isn't installed, install it now"
echo "   See: https://github.com/gpu-mode/popcorn-cli"
echo ""
echo "2. Configure Popcorn CLI"
echo "   - Join Discord: https://discord.gg/gpumode"
echo "   - Get API URL with /get-api-url command"
echo "   - Register: popcorn-cli register discord"
echo ""
echo "3. Connect to GitHub"
echo "   git remote add origin https://github.com/THINNGO2511/NVFP4-Kernel.git"
echo "   git push -u origin main"
echo ""
echo "4. Start coding!"
echo "   - Activate venv: source venv/bin/activate"
echo "   - Read: START_HERE.md"
echo "   - Follow: QUICKSTART.md"
echo ""
echo "🚀 Happy hacking!"
echo ""
