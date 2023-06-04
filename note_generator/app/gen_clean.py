import os
import numpy as np
from scipy.io.wavfile import write
from scipy import signal

# Define constants
sample_rate = 44100  # standard audio sample rate
duration = 0.02  # duration of each note in seconds
variations = 1000  # number of variations for each note
waveforms = ['sine', 'square', 'triangle', 'sawtooth']

# Generate frequencies for all possible notes on a standard-tuned guitar
notes = ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3',
         'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4',
         'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5',
         'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6']
frequencies = [(2**((n-49)/12))*440 for n in range(16, 69)]
note_freq_dict = dict(zip(notes, frequencies))

# Create directories to store files
for waveform in waveforms:
    os.makedirs(waveform + "_20ms", exist_ok=True)

# Generate each note and save to .wav file
for note, freq in note_freq_dict.items():
    for i in range(variations):
        # Generate slight variations in frequency
        freq_variation = freq + (np.random.rand() - 0.5) * 2 * freq * 0.05  # vary up to 5% 

        # Generate samples for the wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        for waveform in waveforms:
            if waveform == 'sine':
                note_samples = 0.5 * np.sin(2 * np.pi * freq_variation * t)
            elif waveform == 'square':
                note_samples = 0.5 * signal.square(2 * np.pi * freq_variation * t)
            elif waveform == 'triangle':
                note_samples = 0.5 * signal.sawtooth(2 * np.pi * freq_variation * t, 0.5)
            elif waveform == 'sawtooth':
                note_samples = 0.5 * signal.sawtooth(2 * np.pi * freq_variation * t)

            # Ensure that highest value is in 16-bit range
            audio = note_samples * (2**15 - 1) / np.max(np.abs(note_samples))
            audio = audio.astype(np.int16)

            # Write .wav file for the note
            write(f'{waveform}_20ms/{note}_{i}.wav', sample_rate, audio)
