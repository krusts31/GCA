import sounddevice as sd

# List all audio devices
devices = sd.query_devices()
for device in devices:
    print(device)

