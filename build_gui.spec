# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Pocket TTS GUI application.
Creates a lightweight executable that downloads dependencies at runtime.
"""

import os
import sys
from pathlib import Path

# Add the package to path for PyInstaller
sys.path.insert(0, os.path.dirname(os.path.abspath(SPEC)))

block_cipher = None

# Get the project root
project_root = Path(SPEC).parent

# Minimal analysis - exclude runtime-downloaded dependencies
a = Analysis(
    ['launch_gui.py'],
    pathex=[str(project_root)],
    binaries=[],
datas=[
    # Include config files
    ('pocket_tts/config/*.yaml', 'pocket_tts/config/'),
    # Include voice files (both MP3 and WAV supported via voice converter)
    ('Voices/George Adam.mp3', 'Voices/'),
    ('Voices/fenric.wav', 'Voices/'),
],
    hiddenimports=[
        # Core Python/Qt dependencies only
        'asyncio',
        'pathlib',
        'urllib',
        'json',
        'logging',
        'sys',
        'os',

        # Qt/PyQt core
        'qtpy',
        'qtpy.QtWidgets',
        'qtpy.QtCore',
        'qtpy.QtGui',

        # Pocket TTS core modules (exclude ML dependencies)
        'pocket_tts.utils.path_manager',
        'pocket_tts.utils.download_manager',
        'pocket_tts.gui.setup_window',
        'pocket_tts.gui.main_window',

        # Preprocessing modules (required by main_window.py)
        'pocket_tts.preprocessing.structure_detector',
        'pocket_tts.preprocessing.chunker',
        'pocket_tts.preprocessing.emotion_analyzer',
        'pocket_tts.preprocessing.parameter_mapper',
        'pocket_tts.preprocessing.schema',
        'pocket_tts.preprocessing.text_normalizer',
        'pocket_tts.preprocessing.contraction_expander',

        # Download manager dependencies
        'aiohttp',
        'aiohttp.http',
        'aiohttp.client',
        'aiohttp.connector',
        'aiohttp.resolver',
        'aiohttp.streams',
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',

        # Basic dependencies that should be available
        'importlib',
        'pkg_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude runtime-downloaded ML dependencies
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'librosa',
        'scipy',
        'numpy',
        'einops',
        'safetensors',
        'sentencepiece',
        'huggingface_hub',
        'tokenizers',
        'accelerate',

        # Exclude unused libraries
        'matplotlib',
        'pandas',
        'PIL',
        'cv2',
        'sklearn',
        'tensorflow',
        'jax',
        'flask',
        'django',

        # Exclude development tools
        'pytest',
        'coverage',
        'black',
        'ruff',
        'mypy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pocket-tts-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console for GUI app
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon if available
)