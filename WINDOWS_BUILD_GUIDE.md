# Pocket TTS Windows Installer - Build Guide

## Overview

You can now build a professional Windows installer for Pocket TTS from Linux, just like you do for BookFix. The installer bundles Python 3.12.8 embeddable and runs a setup wizard on first launch to download ML models.

## Quick Start

```bash
cd build/windows
bash build_linux.sh
```

Output: `output/Pocket-TTS-Setup-1.0.1.exe` (~400-500 MB)

## What Was Created

### 1. Build Script (`build/windows/build_linux.sh`)
- Checks for `wine64`, initializes Wine prefix
- Installs Inno Setup 6 into Wine (one-time)
- Downloads Python 3.12.8 embeddable (cached)
- Stages source files, configures Python
- Runs Inno Setup compiler to create the `.exe`

**Total time**: 5-10 minutes (faster on subsequent builds)

### 2. Launcher (`launcher.pyw`)
- Placed in project root, copied into installer
- On first run: shows setup progress, downloads PyTorch + dependencies
- Subsequent runs: immediately launches Pocket TTS GUI
- Installs deps into Python's `site-packages` (all bundled)

### 3. Requirements (`requirements_windows.txt`)
- Flat pip-installable list (no uv syntax)
- All deps except torch (which uses special index URL)
- Installed at runtime by launcher

### 4. Installer Config (`build/windows/installer.iss`)
- Inno Setup 6 script (processed by ISCC.exe)
- Installs to `%LOCALAPPDATA%\Pocket TTS` (no admin required)
- Bundles entire Python + source + assets
- Creates Start Menu + optional Desktop shortcuts

### 5. Build README (`build/windows/README.md`)
- Full documentation for the build process
- Troubleshooting guide
- Manual build instructions for Windows

## Prerequisites

### On Linux (for building)
- `wine64` — `sudo apt install wine64` on Debian/Ubuntu
- `wget`, `unzip` — Usually pre-installed
- Inno Setup 6 — Automatically installed on first run (into Wine prefix `~/.wine-pocket-tts`)

### On Windows (to run the installer)
- Windows 7 or later
- ~4 GB free disk space
- Internet connection (for first-run setup)

## How It Works

1. **Build Phase (Linux)**:
   ```
   Python 3.12.8 embeddable zip + source code + launcher.pyw 
   → Inno Setup 6 (via Wine) 
   → Pocket-TTS-Setup.exe
   ```

2. **Install Phase (Windows)**:
   ```
   User runs .exe 
   → Extracts to %LOCALAPPDATA%\Pocket TTS 
   → Creates shortcuts
   ```

3. **First Launch**:
   ```
   launcher.pyw detects setup not done
   → Shows progress window
   → Runs: pip install torch (CPU index)
   → Runs: pip install -r requirements_windows.txt
   → Writes marker file
   → Launches GUI
   ```

4. **Subsequent Launches**:
   ```
   launcher.pyw sees marker → launches GUI immediately
   ```

## Files Created/Modified

| File | Status | Purpose |
|---|---|---|
| `build/windows/build_linux.sh` | ✅ New | Linux build script |
| `build/windows/installer.iss` | ✅ New | Inno Setup config |
| `build/windows/README.md` | ✅ New | Build documentation |
| `launcher.pyw` | ✅ New | First-run setup + launcher |
| `requirements_windows.txt` | ✅ New | Pip dependencies |
| `pocket-tts-installer.iss` | ❌ Deleted | Old PyInstaller approach |
| `build_windows_installer.py` | ❌ Deleted | Old PyInstaller approach |
| `build-windows-installer.bat` | ❌ Deleted | Old PyInstaller approach |
| `README.md` | 🔄 Updated | Added Windows section |

## Building the Installer

### Step 1: Prepare Linux Environment

```bash
# Install wine64 (one-time)
sudo apt install wine64

# Clone repo (or enter existing repo)
cd /path/to/Pocket-TTS-Spokenword
```

### Step 2: Run Build Script

```bash
cd build/windows
bash build_linux.sh
```

The script will:
- Check prerequisites
- Initialize Wine prefix `~/.wine-pocket-tts`
- Download + install Inno Setup 6 (one-time, ~5 min)
- Download Python 3.12.8 embeddable (~30 MB, cached)
- Extract and configure Python (pip, certifi)
- Stage source files
- Run `ISCC.exe` to build the installer
- Output: `output/Pocket-TTS-Setup-1.0.1.exe`

### Step 3: Test on Windows

1. Copy `output/Pocket-TTS-Setup-1.0.1.exe` to a Windows machine
2. Run the installer
3. Verify:
   - Installs to `%LOCALAPPDATA%\Pocket TTS`
   - Creates shortcuts
   - First launch shows setup progress
   - Dependencies install correctly
   - GUI launches after setup completes

## Customization

### Change App Version

Edit `build/windows/installer.iss`:
```ini
#define MyAppVersion "1.0.2"
```

And update `requirements_windows.txt` references if needed.

### Change Install Location

Edit `build/windows/installer.iss`:
```ini
; Current: %LOCALAPPDATA%\Pocket TTS (no admin)
DefaultDirName={localappdata}\{#MyAppName}

; Or use Program Files (requires admin):
; DefaultDirName={autopf}\{#MyAppName}
```

### Add Application Icon

1. Create 256x256 `.ico` file
2. Place in `assets/` directory
3. Uncomment in `build/windows/installer.iss`:
   ```ini
   SetupIconFile=..\assets\app-icon.ico
   ```

### Modify Startup Dependencies

Edit `requirements_windows.txt` to add/remove packages. They will be installed on first run.

## Troubleshooting

### "wine64 not found"
```bash
# Debian/Ubuntu:
sudo apt install wine64 wine64-preloader

# Fedora/RHEL:
sudo dnf install wine-core.x86_64

# Arch:
sudo pacman -S wine
```

### Build fails with "Python not found"
- Delete cached Python zip: `rm python-3.12.8-embed-amd64.zip`
- Re-run the build script
- Check internet connection

### Inno Setup installation fails
- Check internet connection (downloads `is.exe` from jrsoftware.org)
- Delete Wine prefix: `rm -rf ~/.wine-pocket-tts`
- Re-run the build script

### Installer is very large
- Expected size: 400-500 MB (includes Python + PyQt5 + other libs)
- First-run adds ~2-3 GB for torch and ML models
- This is intentional — users need no system dependencies

## Distribution

After successful build and testing:

1. **GitHub Releases**:
   ```bash
   gh release create v1.0.1 output/Pocket-TTS-Setup-1.0.1.exe
   ```

2. **Share Link**: Distribute to users who will run it on Windows

3. **Versioning**: Update version in both:
   - `build/windows/installer.iss` (`#define MyAppVersion`)
   - This README and other docs

## Similarities to BookFix Build

The Pocket TTS Windows build uses the same approach as BookFix:

✅ Wine + Inno Setup 6 on Linux
✅ Python embeddable distribution (bundled)
✅ First-run launcher script for setup
✅ Cached downloads for faster rebuilds
✅ No admin privileges required
✅ All dependencies bundled inside installer

The main difference: Pocket TTS uses PyTorch + transformers instead of spaCy models, but the architecture is identical.

## Next Steps

1. **Build**: `bash build/windows/build_linux.sh`
2. **Test**: Run the `.exe` on Windows
3. **Release**: Publish on GitHub Releases
4. **Update**: Add to README with download link

## References

- **Inno Setup Docs**: https://jrsoftware.org/ishelp/
- **Python Embeddable Docs**: https://docs.python.org/3/using/windows.html#the-embeddable-package
- **Wine Docs**: https://www.winehq.org/
- **BookFix Build Reference**: `/media/danno/Team1/BookFixbuild/build/windows/`
