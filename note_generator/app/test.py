import numpy as np
from scipy.io.wavfile import write

# Define constants
sample_rate = 44100  # standard audio sample rate
duration = 5.0  # duration of each note in seconds
variations = 1000  # number of variations for each note

# Define frequencies for guitar strings in standard tuning
# Using equal temperament tuning formula
base_guitar_frequencies = [(2**((n-49)/12))*440 for n in range(28, 40)]
guitar_notes = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3']

# Generate each note and save to .wav file
for base_freq, note in zip(base_guitar_frequencies, guitar_notes):
    for i in range(variations):
        # Generate slight variations in frequency
        freq = base_freq + (np.random.rand() - 0.5) * 2 * base_freq * 0.05  # vary up to 5% 

        # Generate samples for the sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        note_samples = 0.5 * np.sin(2 * np.pi * freq * t)

        # Ensure that highest value is in 16-bit range
        audio = note_samples * (2**15 - 1) / np.max(np.abs(note_samples))
        audio = audio.astype(np.int16)

        # Write .wav file for the note
        write(f'{note}_{i}.wav', sample_rate, audio)
