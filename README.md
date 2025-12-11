# 3D Music Visualizer

A real-time 3D music visualizer built with Python, PyGame, and OpenGL.

## Overview
This application visualizes audio frequencies in a 3D environment. It supports real-time microphone input with multiple visualization modes.

## Setup

### Prerequisites
- Python 3.8+
- A working microphone
- System libraries: `portaudio`, `sdl2` (Installed via Homebrew on macOS)

### Installation

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On macOS, you might need to install system dependencies first:*
   ```bash
   brew install portaudio sdl2
   ```

### Usage

Run the application:
```bash
python main.py
```

#### Setting Up System Audio Loopback (macOS)

To visualize music playing on your computer (instead of microphone input), you need a virtual audio device:

1. **Install BlackHole** (recommended):
   ```bash
   brew install blackhole-2ch
   ```

2. **Create a Multi-Output Device** (to hear audio while visualizing):
   - Open **Audio MIDI Setup** (Applications > Utilities)
   - Click **+** button (bottom left) → Create Multi-Output Device
   - Check both **MacBook Air Speakers** and **BlackHole 2ch**
   - Right-click the Multi-Output Device → "Use This Device For Sound Output"

3. **Select BlackHole in the Visualizer**:
   - Run the visualizer and press **D** to cycle to "BlackHole 2ch"
   - Play music and watch it visualize!

4. **To switch back to normal audio**:
   - Go to System Settings > Sound > Output
   - Select "MacBook Air Speakers"

**Alternative**: You can also use [Soundflower](https://github.com/mattingalls/Soundflower) instead of BlackHole.

### Controls
- **SPACE**: Switch Visualization Mode (Bars -> Terrain -> Vortex)
- **D**: Cycle Audio Input Device (Microphone, System Audio, etc.)
- **UP/DOWN Arrows**: Zoom In/Out (Range: 10-80)
- **Mouse Wheel**: Smooth Zoom In/Out
- **LEFT/RIGHT Arrows**: Adjust Sensitivity
- **F or F11**: Toggle Fullscreen
- **P**: Pause/Resume
- **R**: Toggle Auto-Rotation
- **ESC**: Quit
- **Window Resize**: Drag window edges to resize (automatically adjusts viewport)

## Features
- **Real-time Audio Analysis**: Uses FFT to decompose audio into 64 frequency bands with optimized VBO rendering
- **Multiple Audio Input Sources**:
    - Microphone input for live audio
    - System audio loopback (with BlackHole or Soundflower)
    - Runtime device switching with the **D** key
- **3 Visualization Modes**:
    1.  **Linear Bars**: Standard spectrum analyzer with reflection effects
    2.  **3D Terrain**: Rolling history of frequency data (60 frames deep) with rainbow gradient
    3.  **Vortex**: Multi-ring spiral pattern with concentric frequency layers
- **Interactive Camera**: Smooth orbit and zoom controls (keyboard + mouse wheel)
- **Flexible Display**: Resizable window + fullscreen mode for extended monitors
- **Performance Optimized**: VBO-based rendering for 60 FPS, pre-calculated FFT operations
- **Real-time FPS Display**: Monitor performance metrics in the window title
