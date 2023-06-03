import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa.display
import librosa
from tensorflow.keras.models import load_model
from skimage.transform import resize

# Define the duration of audio to record in seconds
duration = 1.0  # for example, 2 seconds
# Define the sample rate
sample_rate = 44100  # typical value for audio processing

# Load your model
model = load_model('my_model.h5')

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

# Resize the spectrogram to the input size of your CNN
D_resized = resize(D, (256, 256))

# Add extra dimension for grayscale and for batch size
D_resized = D_resized[np.newaxis, ..., np.newaxis]

# Display the spectrogram
librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')

# Predict the label using your model
prediction = model.predict(D_resized)
predicted_label = np.argmax(prediction)

# print the predicted label
print(predicted_label)
