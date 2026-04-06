#!/bin/bash
# ASR Validator Launcher Script with Auto Virtual Environment Management

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           ASR Validation Tool - Launcher                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running from correct directory
if [ ! -f "asr_validator.py" ]; then
    echo "❌ Error: asr_validator.py not found in current directory"
    echo "   Script directory: $SCRIPT_DIR"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "✅ Running from: $SCRIPT_DIR"
echo ""

# Check Python availability
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Error: Python not found"
    echo "   Please install Python 3.8 or higher"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "✅ Found: $PYTHON_VERSION"
echo ""

# Virtual environment path (in same directory as script)
VENV_DIR="$SCRIPT_DIR/venv"

# Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo "✅ Found existing virtual environment at: venv/"
    echo ""
else
    echo "🔍 Virtual environment not found"
    echo ""
    echo "⚠️  SETUP REQUIRED"
    echo "   This tool needs to create a virtual environment and install"
    echo "   dependencies (~3GB download, includes PyTorch)."
    echo ""
    echo "   Location: $VENV_DIR"
    echo "   Packages: torch, torchaudio, librosa, faster-whisper, rapidfuzz"
    echo ""
    read -p "   Create virtual environment and install dependencies? (y/n): " -n 1 -r
    echo ""
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled by user"
        echo ""
        echo "Press Enter to exit..."
        read
        exit 1
    fi
    
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "   Try: $PYTHON_CMD -m pip install --upgrade pip"
        echo ""
        echo "Press Enter to exit..."
        read
        exit 1
    fi
    
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    echo "❌ Error: Cannot find activation script"
    echo "   Expected: $VENV_DIR/bin/activate or $VENV_DIR/Scripts/activate"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "✅ Virtual environment activated"
echo ""

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python -c "import torch; import torchaudio; import librosa; from faster_whisper import WhisperModel; import rapidfuzz" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not installed or incomplete"
    echo ""
    echo "📦 Installing dependencies from requirements.txt..."
    echo "   This may take 5-10 minutes (downloading ~3GB)"
    echo ""
    
    # Upgrade pip first
    python -m pip install --upgrade pip -q
    
    # Install requirements with progress
    python -m pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Installation failed"
        echo "   Check your internet connection and try again"
        echo "   Or manually run: source venv/bin/activate && pip install -r requirements.txt"
        echo ""
        echo "Press Enter to exit..."
        read
        exit 1
    fi
    
    echo ""
    echo "✅ All dependencies installed successfully"
    echo ""
else
    echo "✅ All dependencies OK"
    echo ""
fi

# Verify installation one more time
echo "🔍 Verifying installation..."
python -c "
import torch
import torchaudio
import librosa
from faster_whisper import WhisperModel
import rapidfuzz

print('✅ torch:', torch.__version__)
print('✅ torchaudio:', torchaudio.__version__)
print('✅ librosa:', librosa.__version__)
print('✅ faster-whisper: OK')
print('✅ rapidfuzz:', rapidfuzz.__version__)
"

if [ $? -ne 0 ]; then
    echo "❌ Verification failed"
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  🚀 Launching ASR Validator                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Launch the application
python asr_validator.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Application closed successfully"
else
    echo "⚠️  Application exited with code: $EXIT_CODE"
fi

echo ""
echo "Press Enter to exit..."
read

exit $EXIT_CODE
