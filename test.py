import pyaudio

p = pyaudio.PyAudio()
device_count = p.get_device_count()

if device_count == 0:
    print("No audio input devices detected. Check your microphone and settings.")
else:
    print(f"Detected {device_count} audio devices:")
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']} - Input Channels: {device_info['maxInputChannels']}")

p.terminate()
