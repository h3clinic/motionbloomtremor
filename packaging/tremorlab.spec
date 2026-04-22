# PyInstaller spec for TremorLab.
# Build: pyinstaller packaging/tremorlab.spec
# Output: dist/TremorLab/   (one-dir, faster startup than one-file)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = []
datas += collect_data_files("mediapipe")  # .tflite / .binarypb models
datas += collect_data_files("cv2")

hiddenimports = []
hiddenimports += collect_submodules("mediapipe")
hiddenimports += ["ffpyplayer", "ffpyplayer.player", "ffpyplayer.tools"]
hiddenimports += ["PIL._tkinter_finder"]

a = Analysis(
    ["../tremorlab_run.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tests", "pytest"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TremorLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="Assets/TremorLab.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="TremorLab",
)

# macOS .app bundle
import sys
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="TremorLab.app",
        icon="Assets/TremorLab.icns",
        bundle_identifier="com.motionbloom.tremorlab",
        info_plist={
            "NSCameraUsageDescription": "TremorLab uses the camera to analyze hand tremor locally on your device.",
            "NSMicrophoneUsageDescription": "Audio is not recorded. Required for video playback.",
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "LSMinimumSystemVersion": "11.0",
        },
    )
