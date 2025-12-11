# Product Design Document: 3D Music Visualizer

## 1. Project Overview

### Core Purpose and Target Use Case
The 3D Music Visualizer is a desktop application designed to transform audio input (from files or microphone) into immersive, real-time 3D visual experiences. It targets music enthusiasts, streamers, and audio engineers who need a lightweight, aesthetically pleasing visualization tool for music playback or live broadcasting.

### Key Differentiators
- **True 3D Environment**: Unlike standard 2D frequency bars, this uses a 3D depth field with camera controls.
- **Dual Input Mode**: Seamless switching between local file playback and real-time microphone/system audio capture.
- **High Performance**: Optimized for 60+ FPS using hardware acceleration, ensuring visuals sync perfectly with audio latency <50ms.

### Success Criteria
- Audio-to-visual latency is imperceptible to the user (<50ms).
- Application maintains steady 60 FPS on standard integrated graphics.
- Visuals clearly correspond to distinct audio features (bass thumps, high-hat sizzles).
- Application handles common audio formats (MP3, WAV) and microphone input without crashing.

## 2. Technical Architecture

### System Components Diagram
```
[Audio Source] 
      │
      ▼
[Audio Engine] <───> [Analysis Module]
(PyAudio/Librosa)      (NumPy/SciPy FFT)
      │                     │
      │ (Raw Samples)       │ (Frequency/Beat Data)
      │                     ▼
      └─────────────> [State Manager]
                            │
                            ▼
                     [Rendering Engine]
                     (PyOpenGL / PyGame)
                            │
                            ▼
                        [Display]
```

### Data Flow
1.  **Input**: Audio data is captured in chunks (frames).
2.  **Processing**: 
    -   Fast Fourier Transform (FFT) converts time-domain signal to frequency domain.
    -   Beat detection algorithm analyzes amplitude peaks in low frequency ranges.
    -   Data is normalized and smoothed (decay) to prevent visual jitter.
3.  **State Update**: Visual parameters (scale, color, position) are updated based on processed frequency bands.
4.  **Rendering**: GPU renders the 3D scene using the updated parameters.

### Technology Stack
-   **Language**: Python 3.9+
-   **Windowing & Context**: `PyGame` (Robust, handles windowing, input, and MP3 playback).
-   **Graphics**: `PyOpenGL` (Hardware-accelerated 3D rendering).
-   **Audio I/O**: `PyAudio` (Real-time microphone input).
-   **Math/Analysis**: `NumPy` & `SciPy` (FFT and array manipulations).

**Justification**: Python provides excellent libraries for scientific computing (FFT). `PyGame` simplifies window management, while `PyOpenGL` grants access to low-level hardware acceleration required for 60 FPS 3D graphics.

## 3. Feature Specifications

### MVP Features (Launch Minimum)
-   **Input**:
    -   Real-time Microphone/Line-in capture.
    -   File playback (.wav, .mp3).
-   **Visualization**:
    -   "Spectrum Terrain": A rolling 3D landscape generated from frequency history.
    -   "Circular EQ": 3D cylinders arranged in a circle reacting to bands.
-   **Controls**:
    -   Toggle Input Source.
    -   Orbit Camera (Mouse drag).
    -   Sensitivity Slider.
    -   Quit/Fullscreen toggle.

### Nice-to-have Features (Post-MVP)
-   Shader-based post-processing (Bloom, Motion Blur).
-   Automatic theme cycling.
-   VR Headset support.
-   System audio loopback (capturing "what you hear" without a mic).

### Excluded Features
-   Playlist management (User handles files one by one).
-   Video export/rendering (Real-time only).
-   Network streaming input (Spotify/SoundCloud integration).

## 4. Audio Processing Requirements

### Input Sources
-   **Stream**: `PyAudio` input stream for Microphone. Chunk size ~1024-2048 samples.
-   **File**: Decoded via `PyGame` mixer or `librosa` for analysis, synchronized with playback.

### Frequency Analysis
-   **Method**: FFT (Fast Fourier Transform).
-   **Bands**: Spectrum divided into 3 main zones (Bass: 20-250Hz, Mids: 250-4kHz, Highs: 4k-20kHz), subdivided into at least 32 visual bands for detail.
-   **Smoothing**: Linear interpolation between frames and exponential decay for "falling" bars to simulate gravity/weight.

### Latency
-   Buffer size will be tuned (likely 1024 samples @ 44.1kHz ≈ 23ms) to keep total latency under 50ms.

## 5. 3D Visualization Requirements

### Aesthetic Direction
-   **Style**: Cyberpunk / Wireframe Neon. High contrast, glowing lines against dark backgrounds.
-   **Palette**: Configurable. Default: Cyan/Magenta (Synthwave).

### Camera Controls
-   **Orbital**: User can rotate around the center visualization.
-   **Zoom**: Mouse wheel to distance/close-in.

### Reactive Elements
-   **Geometry Scale**: Objects stretch on Y-axis based on amplitude.
-   **Color Intensity**: Brightness increases with volume.
-   **Particle Systems**: Emitted on beat detection (Kick drum hits).

### Performance
-   Target: 60 FPS stable.
-   Optimization: Use Vertex Buffer Objects (VBOs) to minimize CPU-GPU communication.

## 6. Implementation Plan

### Phase 1: Setup & Audio Core (Days 1-2)
-   Project skeleton.
-   Implement `AudioAnalyzer` class (Mic input + FFT).
-   Verify console output of frequency bands.

### Phase 2: Graphics Core (Days 2-3)
-   Setup `PyGame` window with OpenGL context.
-   Implement basic camera and render loop.
-   Draw a static 3D grid/cube.

### Phase 3: Integration (Days 3-4)
-   Connect Audio data to Graphics geometry.
-   Implement "Spectrum Terrain" mode.
-   Add "Circular EQ" mode.

### Phase 4: Polish & UI (Day 5)
-   Add UI overlay (FPS, Controls).
-   Tune sensitivity and smoothing.
-   Create README and documentation.

