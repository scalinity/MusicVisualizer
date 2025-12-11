#!/usr/bin/env python3
"""
Helper script to list all available audio devices
"""
import pyaudio

p = pyaudio.PyAudio()

print("Available Audio Devices:")
print("=" * 80)

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"\nDevice {i}: {info['name']}")
    print(f"  Max Input Channels: {info['maxInputChannels']}")
    print(f"  Max Output Channels: {info['maxOutputChannels']}")
    print(f"  Default Sample Rate: {info['defaultSampleRate']}")
    print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")

print("\n" + "=" * 80)
print(f"Default Input Device: {p.get_default_input_device_info()['name']}")
print(f"Default Output Device: {p.get_default_output_device_info()['name']}")

p.terminate()
