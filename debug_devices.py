from src.audio_analyzer import AudioAnalyzer
import pyaudio

print("Listing Audio Devices:")
devices = AudioAnalyzer.list_input_devices()
for dev in devices:
    print(f"Index: {dev['index']}, Name: {dev['name']}, Channels: {dev['channels']}, Rate: {dev['sample_rate']}")

analyzer = AudioAnalyzer()
print(f"Selected Device: {analyzer.get_current_device_name()}")
print(f"Configured Channels: {analyzer.CHANNELS}")
