import numpy as np
import joblib
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa.display
import librosa
from tensorflow.keras.models import load_model
from skimage.transform import resize
from threading import Thread
import queue

notes = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3',
         'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4',
         'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5',
         'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6']

# Load your model
model = load_model('my_model_masive.h5')

audio_queue = queue.Queue()

# Define the duration of audio to record in seconds
duration = 0.1  # for example, 2 seconds
# Define the sample rate
sample_rate = 44100  # typical value for audio processing

sd.default.device = 11
#fillter some of the noise
#dynaic filter
#static filter with a treash hold

from scipy.signal import butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def record_audio(audio_queue):
    while True:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=6)#here withe the id of audio interface

        sd.wait()
        audio = audio[:, 0]
        if (np.abs(librosa.stft(audio)).max() > 50):
            print("max: ", np.abs(librosa.stft(audio)).max())
            audio_queue.put(audio)

def process_audio(audio_queue):
    while True:
        audio = audio_queue.get()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        D_resized = resize(D, (256, 256))
        D_resized = D_resized[np.newaxis, ..., np.newaxis]
        prediction = model.predict(D_resized)
        predicted_label = np.argmax(prediction)
        # print the predicted label
        print(notes[predicted_label])

# Start recording and processing threads
record_thread = Thread(target=record_audio, args=(audio_queue,))
process_thread = Thread(target=process_audio, args=(audio_queue,))
record_thread.start()
process_thread.start()
