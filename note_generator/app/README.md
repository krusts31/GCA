### Before starting do
```bash
conda activate GCA-env
```

> This will list all the devices connected to your machine.
> Find the index of your audio interface and remember the number.
```bash
python3 device.py
```

> Running this will train a model based on the sound files.
> In this case, all files in ./generated_notes are used.
> The note name has to be included in the file name in ./generated_notes.
> For instance, something like test_A3.wav, etc.
> This is because the name of the sound also serves as a label.

```bash
python3 train.py
```

> Use this command to generate notes. You can adjust the variation, duration, and sample rate as per your preference.

```bash
python3 gen.py
```

> gen_clean is used to take a sample of a note and then distort it slightly.
> The goal here is to use recorded notes and then multiply them to have more data for the model.

```bash
python3 gen_nice.py
```

> If you run this, you need to set your audio interface ID here!
> audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=your_device_id)
> It will listen and then pass data to the network.
> The output will be a note.
> This line acts as a filter, so you might need to adjust it or adjust the gain.
> If (np.abs(librosa.stft(audio)).max() > 50) > Change 50 to a value that filters out the noise but allows you to see the data when played.
> Here you pass your model.
> You can download the models from here (https://drive.google.com/drive/folders/1nAel5tOW0dtZzun5ydqDWmQXauPxRYlZ) or train your own by using train.py.
> model = load_model('my_model_masive.h5')
> model = load_model('your_model_here.h5')

```bash
python3 live_data.py
```

> use this to pass a single file to the nn

```bash
python3 single_file.py
```

> use this to se a spectorgram from a file

```bash
python3 spectorgram.py
```
