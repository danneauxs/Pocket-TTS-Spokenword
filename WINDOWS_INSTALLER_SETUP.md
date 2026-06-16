# Pocket TTS - Windows Installer Setup Guide

This guide explains how to build and create a professional Windows installer for Pocket TTS.

## Quick Start

### Option 1: Using Python (Recommended)

```bash
python build_windows_installer.py
```

This will:
1. Build the GUI executable using PyInstaller
2. Create a Windows installer (if Inno Setup is installed)

### Option 2: Using Batch Script (Windows Only)

```cmd
build-windows-installer.bat
```

## Prerequisites

### Required
- **Python 3.10+** - Must be installed and in your PATH
- **uv** - Python package manager (install via: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Optional (for Creating Installer)
- **Inno Setup 6** - Required to create the `.exe` installer
  - Download: https://jrsoftware.org/isdl.php
  - Install with default settings
  - Ensure `iscc.exe` is in your PATH

## Step-by-Step Instructions

### 1. Install Dependencies

#### Install uv (if not already installed)
On Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

#### Install project dependencies
```bash
uv sync
```

#### Setup ASR module
```bash
cd ASR
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
deactivate
cd ..
```

### 2. Build the Executable

```bash
python build_windows_installer.py
```

Or, to only build the executable without the installer:
```bash
python build_windows_installer.py --exe-only
```

The executable will be created at:
```
dist/pocket-tts-gui/pocket-tts-gui.exe
```

### 3. Create the Installer

If Inno Setup is installed, the installer is created automatically during the build process.

If you need to create the installer manually:
```bash
python build_windows_installer.py --installer-only
```

Or, if you prefer to use the Inno Setup GUI:
1. Open Inno Setup Compiler
2. File → Open → `pocket-tts-installer.iss`
3. Build → Compile

The installer will be created at:
```
dist/installer/Pocket-TTS-Setup.exe
```

## Installer Features

The Windows installer includes:

- **Application Installation**: Installs Pocket TTS to `Program Files\Pocket TTS`
- **Start Menu Shortcuts**: Easy access from Windows Start Menu
- **Desktop Shortcut**: Optional desktop shortcut for quick launch
- **Uninstall Support**: Full uninstall with registry cleanup
- **Add/Remove Programs**: Appears in Windows Control Panel

## Files Included in Installer

- `pocket-tts-gui.exe` - Main application executable
- `README.md` - Application documentation
- `LICENSE` - License file
- `assets/` - Application assets
- `Voices/` - Voice files
- `ASR/` - Speech recognition module (optional component)

## Customization

### Change Installation Directory

Edit `pocket-tts-installer.iss`:
```ini
DefaultDirName={autopf}\Pocket TTS
```

Replace with your desired path (variables like `{autopf}` = Program Files)

### Add Application Icon

1. Create a 256x256 icon file (`.ico` format)
2. Place it in the `assets/` directory
3. Edit `pocket-tts-installer.iss`:
```ini
SetupIconFile=assets\your-icon.ico
```

### Change Installation Options

In `pocket-tts-installer.iss`, modify the `[Components]` section:
- `main` - Required application component
- `shortcuts` - Create desktop shortcuts
- `asr` - Include speech recognition module

## Distribution

### Option A: Distribute the Installer
- File: `dist/installer/Pocket-TTS-Setup.exe`
- Users simply run this file to install
- **Pros**: Professional, easy for users
- **Cons**: Larger file size

### Option B: Distribute the Standalone Executable
- File: `dist/pocket-tts-gui/pocket-tts-gui.exe`
- Users can run directly without installation
- **Pros**: Portable, smaller file
- **Cons**: Less professional, manual shortcut creation

### Option C: Distribute Both
- Provide installer for standard installations
- Provide standalone exe for portable/USB installations

## Troubleshooting

### Inno Setup Not Found
If you get "Inno Setup compiler not found":
1. Download Inno Setup from: https://jrsoftware.org/isdl.php
2. Install with default settings
3. Add to PATH: `C:\Program Files (x86)\Inno Setup 6\` (adjust version if needed)
4. Or manually run from Inno Setup GUI

### Build Fails
1. Ensure Python 3.10+ is installed: `python --version`
2. Ensure uv is installed: `uv --version`
3. Ensure dependencies are installed: `uv sync`
4. Try clean rebuild: `python build.py --gui-only --clean`

### Installer is Too Large
The installer includes all dependencies. To reduce size:
1. Edit `build_gui.spec` to exclude unused dependencies
2. Comment out the ASR component in `pocket-tts-installer.iss` if not needed
3. Use UPX compression (already enabled in build_gui.spec)

## Advanced Usage

### Build with Additional Logging
```bash
python build_windows_installer.py --verbose
```

### Build for Specific Python Version
```bash
python3.11 build_windows_installer.py
```

### Create Portable ZIP Distribution
After building the executable:
```bash
powershell -Command "Compress-Archive -Path 'dist\pocket-tts-gui' -DestinationPath 'dist\Pocket-TTS-Portable.zip'"
```

## Next Steps

1. **Test the Installer**
   - Run `Pocket-TTS-Setup.exe`
   - Verify installation path
   - Check shortcuts are created
   - Test application launches

2. **Test the Application**
   - Verify all features work
   - Test with sample text
   - Check voice files are accessible

3. **Create Release**
   - Sign the installer (optional, but recommended)
   - Create release notes
   - Upload to GitHub Releases

## Version Updates

To build a new installer version:

1. Update version in `pyproject.toml`
2. Update version in `pocket-tts-installer.iss` (AppVersion)
3. Rebuild: `python build_windows_installer.py`

## Additional Resources

- **Inno Setup Documentation**: https://jrsoftware.org/ishelp/
- **PyInstaller Documentation**: https://pyinstaller.org/
- **Qt Documentation**: https://doc.qt.io/qt-5/

## Support

For issues or questions:
- GitHub Issues: https://github.com/kyutai-labs/pocket-tts/issues
- Documentation: See README.md
