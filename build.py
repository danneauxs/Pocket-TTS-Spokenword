#!/usr/bin/env python3
"""
Cross-platform build script for Pocket TTS
Creates lightweight executables that download dependencies at runtime.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def get_build_command(spec_file: str, dist_dir: str = "dist") -> list:
    """Generate PyInstaller command for the given spec file."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",  # Clean cache
        "--noconfirm",  # Don't ask for confirmation
        "--distpath", dist_dir,  # Output directory
        spec_file
    ]
    return cmd

def ensure_uv_environment():
    """Ensure we're running in a uv environment with required packages."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} available")
    except ImportError:
        print("✗ PyInstaller not found. Install with: uv add pyinstaller")
        sys.exit(1)

    try:
        import qtpy
        print(f"✓ QtPy {qtpy.__version__} available")
    except ImportError:
        print("✗ QtPy not found. Install with: uv add qtpy")
        sys.exit(1)

def clean_build_artifacts():
    """Clean build artifacts with better Windows support."""
    dirs_to_clean = ["build", "dist"]
    cleaned = []

    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            try:
                # On Windows, try multiple times with delays for file locks
                import time
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(dir_name)
                        print(f"  Removed {dir_name}/")
                        cleaned.append(dir_name)
                        break
                    except OSError as e:
                        if attempt < max_attempts - 1:
                            print(f"  Retrying {dir_name} removal in 2 seconds...")
                            time.sleep(2)
                        else:
                            print(f"  Warning: Could not remove {dir_name}: {e}")
                            # Try to remove individual files instead
                            try:
                                for root, dirs, files in os.walk(dir_name, topdown=False):
                                    for name in files:
                                        try:
                                            os.remove(os.path.join(root, name))
                                        except OSError:
                                            pass  # Skip locked files
                                    for name in dirs:
                                        try:
                                            os.rmdir(os.path.join(root, name))
                                        except OSError:
                                            pass  # Skip non-empty dirs
                                print(f"  Partially cleaned {dir_name}/")
                                cleaned.append(dir_name)
                            except Exception:
                                print(f"  Could not clean {dir_name} at all")
                else:
                    print(f"  Skipped {dir_name}/ (in use)")
            except Exception as e:
                print(f"  Error cleaning {dir_name}: {e}")

    return cleaned

def build_app(spec_file: str, app_name: str):
    """Build a single application."""
    print(f"\n{'='*50}")
    print(f"Building {app_name}")
    print(f"{'='*50}")

    if not Path(spec_file).exists():
        print(f"✗ Spec file not found: {spec_file}")
        return False

    # Create dist directory if it doesn't exist
    dist_dir = "dist"
    Path(dist_dir).mkdir(exist_ok=True)

    # Run PyInstaller
    cmd = get_build_command(spec_file, dist_dir)
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Build completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-1000:])
        if e.stderr:
            print("STDERR:", e.stderr[-1000:])
        return False

def create_test_voice_asset():
    """Create a minimal test voice asset if it doesn't exist."""
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    test_voice = assets_dir / "test_voice.wav"
    if not test_voice.exists():
        print("⚠ No test voice found. Creating placeholder...")
        # Create a minimal WAV file (this is just a placeholder)
        # In a real scenario, you'd include an actual test voice file
        print("   Note: Add a real test voice file to assets/test_voice.wav")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Build Pocket TTS executables")
    parser.add_argument("--gui-only", action="store_true",
                       help="Build GUI application only")
    parser.add_argument("--cli-only", action="store_true",
                       help="Build CLI application only")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build artifacts before building")

    args = parser.parse_args()

    print("Pocket TTS Builder")
    print("==================")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print()

    # Clean if requested
    if args.clean:
        print("Cleaning build artifacts...")
        cleaned_dirs = clean_build_artifacts()
        if not cleaned_dirs:
            print("  No build artifacts to clean")

    # Ensure environment is ready
    ensure_uv_environment()

    # Create test voice asset
    create_test_voice_asset()

    # Determine what to build
    build_gui = not args.cli_only
    build_cli = not args.gui_only

    success_count = 0
    total_builds = (1 if build_gui else 0) + (1 if build_cli else 0)

    # Build GUI application
    if build_gui:
        if build_app("build_gui.spec", "GUI Application"):
            success_count += 1
        else:
            print("✗ GUI build failed")

    # Build CLI application
    if build_cli:
        if build_app("build_cli.spec", "CLI Application"):
            success_count += 1
        else:
            print("✗ CLI build failed")

    # Summary
    print(f"\n{'='*50}")
    print("Build Summary")
    print(f"{'='*50}")
    print(f"Successful builds: {success_count}/{total_builds}")

    if success_count == total_builds:
        print("✓ All builds completed successfully!")
        print("\nDistribution files:")
        dist_dir = Path("dist")
        if dist_dir.exists():
            for item in dist_dir.iterdir():
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"  {item.name} ({size_mb:.1f} MB)")
    else:
        print("✗ Some builds failed")
        sys.exit(1)

if __name__ == "__main__":
    main()