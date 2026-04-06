#!/usr/bin/env python3
"""
Audiobook Generator GUI Launcher
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pocket_tts.gui.main_window import main

if __name__ == "__main__":
    main()
