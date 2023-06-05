import numpy as np
import librosa
from tensorflow.keras.models import load_model
from skimage.transform import resize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


notes = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3',
         'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4',
         'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5',
         'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6']

# Load your model
model = load_model('my_model_masive.h5')

# Define the sample rate
sample_rate = 44100  # typical value for audio processing

# Path of your audio file
audio_file_path = "pitch_follower_Slide-G.mp3"

# Load audio file
audio_data, _ = librosa.load(audio_file_path, sr=sample_rate)

# Define chunk size (in seconds)
chunk_size = 0.02  # 20 ms

# Calculate the number of samples per chunk
samples_per_chunk = int(sample_rate * chunk_size)

# Calculate the number of chunks
num_chunks = len(audio_data) // samples_per_chunk

# Reshape audio data into chunks
audio_chunks = np.reshape(audio_data[:num_chunks * samples_per_chunk], (num_chunks, samples_per_chunk))

# Process each chunk
for chunk in audio_chunks:
    # Compute the spectrogram of the chunk
    D_chunk = librosa.amplitude_to_db(np.abs(librosa.stft(chunk)), ref=np.max)

    # Resize the spectrogram to the input size of your CNN
    D_resized = resize(D_chunk, (256, 256))

    # Add extra dimension for grayscale and for batch size
    D_resized = D_resized[np.newaxis, ..., np.newaxis]

    # Predict the label using your model
    prediction = model.predict(D_resized, verbose=0)

    # Get the maximum probability
    max_prob = np.max(prediction)

    # If the maximum probability is greater than a threshold (e.g. 0.5), print the predicted label
    if max_prob > 0.5:
        predicted_label = np.argmax(prediction)
        print(notes[predicted_label])

