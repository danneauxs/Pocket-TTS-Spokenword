#!/bin/bash
cd "$(dirname "$0")"

# We tell the terminal to load the bash environment FIRST, then run the command
mate-terminal -- bash -i -c "source ~/.bashrc; uv run python launch_gui.py; exec bash"
