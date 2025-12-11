# System Audio Loopback Setup Guide

## Overview
The Music Visualizer now supports switching between multiple audio input devices, including:
- Built-in microphone
- External microphones
- **System audio loopback devices** (BlackHole, Soundflower)

## What Changed

### AudioAnalyzer Enhancements
- Added device selection support via `device_index` parameter
- New `list_input_devices()` class method to enumerate available audio inputs
- New `switch_device()` method for runtime device switching
- Automatic sample rate adaptation for different devices
- Better error handling for device connections

### Visualizer UI Updates
- Press **D** to cycle through available audio input devices
- Current device name shown in window title
- Automatic device list detection on startup

## Setting Up System Audio on macOS

### Option 1: BlackHole (Recommended)

BlackHole is a modern, open-source virtual audio driver for macOS.

#### Installation
```bash
brew install blackhole-2ch
```

#### Configuration
1. **Create Multi-Output Device** (to hear audio while visualizing):
   - Open **Audio MIDI Setup** (Applications > Utilities > Audio MIDI Setup)
   - Click the **+** button (bottom left)
   - Select **"Create Multi-Output Device"**
   - In the right panel, check:
     - ✓ MacBook Air Speakers (or your preferred output)
     - ✓ BlackHole 2ch
   - Optional: Rename to "Speakers + BlackHole"

2. **Set System Output**:
   - Right-click your Multi-Output Device
   - Select **"Use This Device For Sound Output"**
   - OR: Go to System Settings > Sound > Output and select it

3. **Use in Visualizer**:
   - Launch the visualizer: `python main.py`
   - Press **D** to cycle devices until you see "BlackHole 2ch"
   - Play music from any app
   - The visualizer will react to the music!

4. **Restore Normal Audio** (when done):
   - System Settings > Sound > Output
   - Select "MacBook Air Speakers"

### Option 2: Soundflower (Alternative)

Soundflower is an older but still functional option.

#### Installation
```bash
# Download from GitHub
# https://github.com/mattingalls/Soundflower/releases
```

Follow similar steps as BlackHole to create a Multi-Output Device.

## Testing Your Setup

### Quick Test
```bash
cd "/Users/danny/Documents/Computer Work/Music Visualizer"
source venv/bin/activate
python list_audio_devices.py
```

This will show all available audio devices. You should see:
- MacBook Air Microphone
- BlackHole 2ch (if installed)
- Any other connected audio devices

### Test the Visualizer
```bash
python main.py
```

1. The visualizer starts with the default microphone
2. Press **D** to switch to BlackHole
3. Play music in Spotify, YouTube, etc.
4. Watch the visualization react to your music!

## Troubleshooting

### No Sound After Setting Up Multi-Output
- Make sure you selected the Multi-Output Device in System Settings > Sound
- Verify both speakers and BlackHole are checked in Audio MIDI Setup

### Visualizer Not Reacting to Music
- Press **D** to ensure you're on the BlackHole device
- Check that music is actually playing
- Increase sensitivity with **RIGHT ARROW** key
- Verify system output is set to the Multi-Output Device

### Device Not Listed
- Make sure BlackHole is properly installed
- Restart the visualizer after installing new audio drivers
- Check Audio MIDI Setup to verify the device exists

### Audio Quality Issues
- BlackHole supports up to 48kHz sample rate
- The visualizer automatically adapts to device sample rates
- If you hear crackling, the Multi-Output Device might need reconfiguration

## Technical Details

### How It Works
1. **BlackHole** creates a virtual audio input that captures system output
2. The **Multi-Output Device** splits audio to both speakers AND BlackHole
3. The visualizer reads from BlackHole as if it were a microphone
4. FFT analysis converts audio to frequency bands for visualization

### Sample Rates
- Default: 44100 Hz
- BlackHole default: 48000 Hz
- The analyzer automatically adapts to each device's native sample rate

### Latency
- Expected latency: < 50ms for most configurations
- BlackHole has minimal latency overhead
- Multi-Output Device adds ~10ms additional latency

## Advanced: Programmatic Device Selection

You can also specify a device when creating the AudioAnalyzer:

```python
from src.audio_analyzer import AudioAnalyzer

# List available devices
devices = AudioAnalyzer.list_input_devices()
for dev in devices:
    print(f"[{dev['index']}] {dev['name']}")

# Create analyzer with specific device
analyzer = AudioAnalyzer(device_index=2)  # Use device #2

# Switch device at runtime
analyzer.switch_device(1)  # Switch to device #1
```

## Future Enhancements

Potential improvements:
- Save preferred device in config file
- Auto-detect and prefer loopback devices
- Support for multi-channel system audio (5.1, 7.1)
- Windows/Linux loopback support (using different methods)
