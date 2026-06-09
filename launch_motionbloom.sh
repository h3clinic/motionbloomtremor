#!/bin/bash
# MotionBloom UI - Launch Script
# This script sets up and launches MotionBloom with the new bright red/white UI

set -e  # Exit on error

PROJECT_DIR="/Users/aharshi/MotionBloomAppVersion/motionbloomtremor"
PYTHON3="/opt/homebrew/bin/python3"

echo "🚀 MotionBloom UI Launcher"
echo "=================================================="
echo ""
echo "Project Directory: $PROJECT_DIR"
echo "Python: $PYTHON3 ($($PYTHON3 --version))"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Step 1: Check Python
echo "✓ Step 1: Python Version"
$PYTHON3 --version
echo ""

# Step 2: Check if venv exists
echo "✓ Step 2: Virtual Environment"
if [ -d ".venv" ]; then
    echo "  Virtual environment found (.venv)"
    source .venv/bin/activate
    echo "  Activated: $(.venv/bin/python --version)"
else
    echo "  No virtual environment found"
    echo "  Create one with: python3 -m venv .venv"
    echo "  Then activate with: source .venv/bin/activate"
    echo ""
fi

# Step 3: Check dependencies
echo "✓ Step 3: Checking Dependencies..."
echo ""

# Try to import key modules
MODULES=("PyQt6" "cv2" "mediapipe" "numpy" "scipy")
MISSING=()

for module in "${MODULES[@]}"; do
    if python3 -c "import ${module}" 2>/dev/null; then
        echo "  ✓ $module installed"
    else
        echo "  ✗ $module MISSING"
        MISSING+=("$module")
    fi
done

echo ""

# If missing dependencies
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "⚠️  Missing dependencies detected!"
    echo ""
    echo "Install with:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Or install specific packages:"
    for module in "${MISSING[@]}"; do
        case "$module" in
            "PyQt6")
                echo "  pip install PyQt6 PyQt6-multimedia"
                ;;
            "cv2")
                echo "  pip install opencv-python"
                ;;
            "mediapipe")
                echo "  pip install mediapipe"
                ;;
            *)
                echo "  pip install $module"
                ;;
        esac
    done
    echo ""
    exit 1
fi

echo "✅ All dependencies found!"
echo ""

# Step 4: Launch MotionBloom
echo "✓ Step 4: Launching MotionBloom..."
echo ""
echo "Starting application with bright red & white theme..."
echo "This window will show the app output."
echo ""
echo "=================================================="
echo ""

# Launch the main app
python3 -m motionbloom

# If -m doesn't work, try direct launch
if [ $? -ne 0 ]; then
    echo ""
    echo "Trying alternative launch method..."
    python3 motionbloom_run.py
fi
