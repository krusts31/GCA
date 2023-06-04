import os
import numpy as np
import librosa
import soundfile as sf

# Directory containing your original audio files
input_directory = './mono_tone/'

# Directory where you want to save your new audio files
output_directory = './generated_notes/'

# Make sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):  # assuming the files are all .wav, modify as needed
        audio_file_path = os.path.join(input_directory, filename)

        # Extract note name from the filename
        note_name = filename.split('-')[1].split('.')[0]

        # Load audio file
        audio_data, sr = librosa.load(audio_file_path, sr=None)

        # Create 200 new samples with variation in pitch
        for i in range(200):
            # Generate a random pitch shift between -1/2 and +1/2 semitones
            pitch_shift_semitones = np.random.uniform(-0.1, 0.1)

            # Apply the pitch shift to the audio data
            new_audio_data = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=pitch_shift_semitones)

            # Define the output filename, note that the note name is now separated by underscore for easy extraction
            new_filename = f"raw_{i}_{note_name}_{int(pitch_shift_semitones*100)}p.wav"

            # Full path for the output file
            output_file_path = os.path.join(output_directory, new_filename)

            # Save the new audio data to a file
            sf.write(output_file_path, new_audio_data, sr)

