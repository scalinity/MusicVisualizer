import pyaudio
import numpy as np
import math

class AudioAnalyzer:
    # Audio input constants
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024
    AUDIO_FORMAT = pyaudio.paInt16
    NUM_CHANNELS = 2

    # Frequency analysis constants
    NUM_FREQUENCY_BANDS = 64
    LOW_FREQ_CUTOFF_BINS = 2  # Skip first 2 FFT bins (~0-40Hz)
    SMOOTHING_FACTOR = 0.7
    NOISE_THRESHOLD = 0.05

    def __init__(self, device_index=None):
        self.RATE = self.SAMPLE_RATE
        self.CHUNK = self.CHUNK_SIZE
        self.FORMAT = self.AUDIO_FORMAT
        self.CHANNELS = self.NUM_CHANNELS

        self.p = pyaudio.PyAudio()
        self.current_device_index = device_index
        self.stream = None

        # Open initial stream
        self._open_stream(device_index)

        # Frequency bands configuration
        self.num_bands = self.NUM_FREQUENCY_BANDS
        self.bands = np.zeros((self.NUM_CHANNELS, self.num_bands))
        self.smoothing_factor = self.SMOOTHING_FACTOR

        # Pre-calculate Hanning window (same for every frame)
        self.window = np.hanning(self.CHUNK)

        # Pre-calculate FFT normalization factor
        self.fft_norm_factor = 1.0 / (self.CHUNK / 2)

        # Pre-calculate band bin ranges for logarithmic mapping
        self.low_freq_cutoff = self.LOW_FREQ_CUTOFF_BINS
        self._precalculate_band_ranges()

    def _open_stream(self, device_index=None):
        """Open audio stream with specified device"""
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()

            # Determine sample rate for the device
            if device_index is not None:
                device_info = self.p.get_device_info_by_index(device_index)
                # Some devices may not support 44100, try their default
                try:
                    self.stream = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=self.CHUNK
                    )
                except:
                    # Fallback to device's default sample rate
                    default_rate = int(device_info['defaultSampleRate'])
                    self.stream = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=default_rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=self.CHUNK
                    )
                    self.RATE = default_rate
            else:
                # Use default input device
                self.stream = self.p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )

            self.current_device_index = device_index
            print(f"Audio stream opened on device: {self.get_current_device_name()}")

        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.stream = None

    @classmethod
    def list_input_devices(cls):
        """List all available input devices"""
        p = pyaudio.PyAudio()
        devices = []

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })

        p.terminate()
        return devices

    def get_current_device_name(self):
        """Get the name of the currently selected device"""
        try:
            if self.current_device_index is None:
                return self.p.get_default_input_device_info()['name']
            else:
                return self.p.get_device_info_by_index(self.current_device_index)['name']
        except:
            return "Unknown Device"

    def switch_device(self, device_index):
        """Switch to a different input device"""
        if device_index != self.current_device_index:
            print(f"Switching to device index: {device_index}")
            self._open_stream(device_index)
            # Reset bands to avoid artifacts
            self.bands = np.zeros((self.NUM_CHANNELS, self.num_bands))

    def _precalculate_band_ranges(self):
        """Pre-calculate frequency band bin ranges to avoid recalculation each frame"""
        # Estimate FFT length (will be CHUNK//2 + 1 for real FFT)
        fft_len = self.CHUNK // 2 + 1
        fft_len_usable = fft_len - self.low_freq_cutoff

        self.band_ranges = []
        for i in range(self.num_bands):
            start_ratio = (i / self.num_bands) ** 2
            end_ratio = ((i + 1) / self.num_bands) ** 2

            start_bin = int(start_ratio * fft_len_usable) + self.low_freq_cutoff
            end_bin = int(end_ratio * fft_len_usable) + self.low_freq_cutoff

            if end_bin <= start_bin:
                end_bin = start_bin + 1

            self.band_ranges.append((start_bin, end_bin))

    def read_audio(self):
        if self.stream is None:
            return np.zeros(self.num_bands)

        try:
            # Read raw data
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            
            # Convert to numpy array
            # Data is interleaved [L, R, L, R...] for stereo
            raw_data = np.frombuffer(data, dtype=np.int16)
            
            # Use safe reshape to avoid errors if buffer size mismatch (though unlikely with read(chunk))
            if raw_data.size == 0:
                 return np.zeros((self.CHANNELS, self.num_bands))

            # Reshape to (samples, channels)
            # If mono, it will be (samples, 1). If stereo, (samples, 2)
            try:
                raw_data = raw_data.reshape(-1, self.CHANNELS)
            except ValueError:
                # Fallback if size doesn't match
                return self.bands

            channels_bands = []
            
            for ch in range(self.CHANNELS):
                # Apply window
                audio_data = raw_data[:, ch] * self.window
                
                # FFT
                fft_data = np.abs(np.fft.rfft(audio_data)) * self.fft_norm_factor
                
                # Bands
                ch_bands = np.array([
                    np.mean(fft_data[start:end]) if end < len(fft_data) else 0.0
                    for start, end in self.band_ranges
                ])
                channels_bands.append(ch_bands)
            
            new_bands = np.stack(channels_bands) # Shape (2, 64)

            # Apply noise gate
            new_bands[new_bands < self.NOISE_THRESHOLD] = 0

            # Exponential smoothing
            self.bands = self.bands * self.smoothing_factor + new_bands * (1 - self.smoothing_factor)

            return self.bands

        except Exception as e:
            print(f"Error processing audio: {e}")
            return np.zeros(self.num_bands)

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    # List available devices
    print("Available Input Devices:")
    print("=" * 60)
    devices = AudioAnalyzer.list_input_devices()
    for dev in devices:
        print(f"[{dev['index']}] {dev['name']} ({dev['channels']} ch, {dev['sample_rate']} Hz)")
    print("=" * 60)

    # Simple test
    analyzer = AudioAnalyzer()
    print(f"\nListening on: {analyzer.get_current_device_name()}")
    print("Press Ctrl+C to stop")
    try:
        while True:
            bands = analyzer.read_audio()
            # improved visualization for terminal
            max_val = np.max(bands) if np.max(bands) > 0 else 1
            visual = ["#" * int((b / max_val) * 20) for b in bands[:10]] # Show first 10 bands
            print(f"Bands: {'|'.join(visual)}", end='\r')
    except KeyboardInterrupt:
        analyzer.close()
        print("\nStopped.")
