import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('./3noteswav/B2&E3&F#3.wav')

# Compute the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#D = librosa.amplitude_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)


# Display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.title('Spectrogram')
plt.show()
