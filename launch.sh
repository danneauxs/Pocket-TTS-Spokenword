#!/bin/bash
cd "$(dirname "$0")"

# Load your environment
source ~/.bashrc

# Run the app using uv
uv run python launch_gui.py