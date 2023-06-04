import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from skimage.transform import resize

notes = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3',
         'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4',
         'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5',
         'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6']

# Load your model
model = load_model('my_model_masive.h5')

# Define the sample rate
sample_rate = 44100  # typical value for audio processing

# Directory containing your audio files
directory = './mono_tone/'

for filename in os.listdir(directory):
    if filename.endswith(".wav"):  # assuming the files are all .wav, modify as needed
        audio_file_path = os.path.join(directory, filename)

        # Load audio file
        audio_data, _ = librosa.load(audio_file_path, sr=sample_rate)

        # Compute the spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)

        # Resize the spectrogram to the input size of your CNN
        D_resized = resize(D, (256, 256))

        # Add extra dimension for grayscale and for batch size
        D_resized = D_resized[np.newaxis, ..., np.newaxis]

        # Predict the label using your model
        prediction = model.predict(D_resized)

        predicted_label = np.argmax(prediction)

        # Print the predicted label and the name of the file
        print(f'File: {filename}, Predicted note: {notes[predicted_label]}')

