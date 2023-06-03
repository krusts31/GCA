import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa.display
import librosa

# Define the duration of audio to record in seconds
duration = 1.0  # for example, 2 seconds
# Define the sample rate
sample_rate = 44100  # typical value for audio processing

sd.default.device = 11

# Record audio for 'duration' seconds
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
print(audio, type(audio), audio.shape)

# Wait for the recording to finish
sd.wait()

# Since the audio is recorded as 2D array, we take the first dimension to get 1D audio data
audio = audio[:, 0]

# Compute the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
