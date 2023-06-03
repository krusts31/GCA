import os
from tensorflow.keras.models import load_model
import librosa
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define your audio directory and get the list of all files
audio_dir = './music'
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

# Create a list to hold your spectrograms and a list to hold your labels
spectrograms = []
labels = []

# For each audio file, compute the spectrogram and label
for index, f in enumerate(audio_files):
    # Load the audio file
    y, sr = librosa.load(os.path.join(audio_dir, f))

    # Compute the spectrogram and convert it to dB
    D = np.abs(librosa.stft(y))**2

    spectrogram = librosa.amplitude_to_db(D, ref=np.max)

    #print(spectrogram.shape)

    # Resize the spectrogram to the size your model expects (e.g., 256 X 256)
    spectrogram = resize(spectrogram, (256, 256), anti_aliasing=True)
    
    # Add an extra dimension for the channel (since we're working with grayscale images)
    spectrogram = np.expand_dims(spectrogram, axis=-1)

    # Append the spectrogram to your list
    spectrograms.append(spectrogram)

    # Parse out the label from the file name and append it to your labels list
    label = os.path.splitext(f)[0].split('_')[0]  # assuming the note is the first part of the file name, before an underscore
    #print(label)
    labels.append(label)

# Convert your lists into numpy arrays
X = np.array(spectrograms)
y = np.array(labels)

# Encode your labels as integers
le = LabelEncoder()
y = le.fit_transform(y)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model('my_model.h5')


"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))  # number of classes is the number of unique labels

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
model.save('my_model.h5')
"""

prediction = model.predict(X_test)
predicted_label = np.argmax(prediction)
y_pred = le.inverse_transform(y)
y_test = le.inverse_transform(y_test)

for i in range(len(y_pred) - 1):
    print(y_pred[i], y_test[i])
