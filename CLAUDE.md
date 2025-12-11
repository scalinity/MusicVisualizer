# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# List available audio devices (useful for debugging)
python list_audio_devices.py
```

### System Dependencies (Linux)
```bash
sudo apt install portaudio19-dev python3-pyaudio
```

### System Dependencies (macOS)
```bash
brew install portaudio sdl2
```

## Architecture

This is a real-time 3D music visualizer using PyAudio for audio capture and PyOpenGL for rendering.

### Data Flow
```
Audio Input (PyAudio) → FFT Analysis (NumPy) → Visualization (PyOpenGL/PyGame)
```

### Core Components

**`src/audio_analyzer.py`** - Audio capture and frequency analysis
- Captures stereo audio via PyAudio (44.1kHz, 1024 sample chunks)
- Performs FFT and maps to 64 logarithmically-spaced frequency bands
- Handles device enumeration and runtime switching
- Pre-calculates Hanning window and band bin ranges for performance

**`src/visualizer.py`** - OpenGL rendering and main loop
- Implements custom `gluPerspective`, `gluLookAt`, `gluOrtho2D` (avoids GLU dependency)
- Uses VBOs for cube and terrain geometry
- Four visualization modes: Bars, Terrain, Vortex (circular), Stereo
- Terrain mode maintains 180-frame history for rolling 3D landscape

### Key Implementation Details

- Stereo channels are averaged to mono for symmetric visualization (prevents one-sided activity from panned audio)
- Terrain VBO is dynamically updated each frame with vectorized NumPy operations
- MSAA (4x multisampling) enabled for anti-aliasing
- Specular highlights disabled for matte finish on bars

### Controls
- SPACE: Cycle visualization mode
- D: Cycle audio input device
- UP/DOWN or Mouse Wheel: Zoom
- LEFT/RIGHT: Adjust sensitivity
- R: Toggle auto-rotation
- P: Pause
- F/F11: Toggle fullscreen
