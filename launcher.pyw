#!/usr/bin/env python3
"""
Pocket TTS Windows Launcher
First-run: installs PyTorch, transformers, and other dependencies (~2-3 GB)
Subsequent runs: launches Pocket TTS GUI immediately
Uses console output for progress feedback.
"""
import sys
import os
import subprocess
from pathlib import Path


INSTALL_DIR = Path(__file__).resolve().parent
BUNDLED_PYTHON = INSTALL_DIR / "python" / "python.exe"
BUNDLED_PYTHONW = INSTALL_DIR / "python" / "pythonw.exe"
LAUNCH_GUI = INSTALL_DIR / "launch_gui.py"
REQUIREMENTS_TXT = INSTALL_DIR / "requirements_windows.txt"
SETUP_MARKER = INSTALL_DIR / ".setup_complete"
NOCONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_cmd(cmd, label=""):
    """Run a command with streaming output."""
    if label:
        print(f"  {label}...")
        sys.stdout.flush()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        creationflags=NOCONSOLE if sys.platform == "win32" else 0
    )
    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"    {line[:120]}")
    proc.wait()
    return proc


def run_setup():
    """Run first-time setup with console output."""
    print()
    print("=" * 60)
    print("  Pocket TTS First-Time Setup")
    print("=" * 60)
    print()
    print("This will download PyTorch, language models, and other")
    print("dependencies (~2-3 GB). It may take 10-20 minutes depending")
    print("on your internet speed.")
    print()

    print("[1/3] Upgrading pip...")
    r = run_cmd(
        [str(BUNDLED_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
        "Upgrading pip"
    )
    if r.returncode != 0:
        print("  Warning: pip upgrade failed, continuing...")

    print("[2/3] Installing PyTorch (CPU-only, ~800 MB)...")
    r = run_cmd(
        [str(BUNDLED_PYTHON), "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        "Installing PyTorch"
    )
    if r.returncode != 0:
        print("  FAILED: PyTorch installation failed. Check internet connection.")
        input("\nPress Enter to exit...")
        return False

    print("[3/3] Installing remaining dependencies...")
    if REQUIREMENTS_TXT.exists():
        r = run_cmd(
            [str(BUNDLED_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_TXT)],
            "Installing requirements"
        )
        if r.returncode != 0:
            print("  Warning: some requirements had issues, continuing...")
    else:
        print("  Warning: requirements_windows.txt not found, skipping...")

    SETUP_MARKER.write_text("ok")
    print()
    print("=" * 60)
    print("  Setup complete! Launching Pocket TTS...")
    print("=" * 60)
    print()
    return True


def main():
    if not LAUNCH_GUI.exists():
        print(f"ERROR: launch_gui.py not found at: {LAUNCH_GUI}")
        input("\nPress Enter to exit...")
        return 1

    if not BUNDLED_PYTHON.exists():
        print()
        print("=" * 60)
        print("  Pocket TTS Install Error")
        print("=" * 60)
        print()
        print("python.exe not found in the install directory.")
        print("Please reinstall Pocket TTS.")
        print()
        input("Press Enter to exit...")
        return 1

    if not SETUP_MARKER.exists():
        success = run_setup()
        if not success:
            return 1

    os.chdir(str(INSTALL_DIR))
    subprocess.run(
        [str(BUNDLED_PYTHONW), str(LAUNCH_GUI)],
        creationflags=NOCONSOLE if sys.platform == "win32" else 0
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
