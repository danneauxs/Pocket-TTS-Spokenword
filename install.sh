#!/bin/bash
set -e  # Exit on any error

echo "Starting installation..."

# Step 1: Install main dependencies in current directory
echo "Installing main pocket-tts dependencies..."
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv not found. Please install uv first (e.g., curl -LsSf https://astral.sh/uv/install.sh | sh)"
    exit 1
fi
uv sync

# Step 2: Set up ASR subdirectory (venv + deps, no launch)
echo "Setting up ASR dependencies..."
cd ASR
if [ ! -d "venv" ]; then
    echo "Creating ASR virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "Installing ASR dependencies..."
pip install -r requirements.txt
deactivate
cd ..

echo "Installation complete! You can now run the program via launch.sh."
