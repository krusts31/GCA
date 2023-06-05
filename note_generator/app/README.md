```bash
#this will tell you all the devices connected to your machine
#find your audio interface index and rembear the number
python3 device.py

#if you run this it will train a model based on the sound files
#in this case all the files in ./generated_notes
#the note name has to be in the file name in ./generated_notes
#so something like test_A3.wav etc.
#this needs to be so because the name of the sound also serves as a lable
python3 train.py

#this is used to generate notes you can change the variation and duration
#sample rate to what you prefer
python3 gen.py

#gen clean is used to take a sample of a note and then distortit slightly
#the goal hear is to use recorded notes and then multiply them to have
#more data for the model
python3 gen.py

#if you play this you need to set your audi interface id here!
#audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=your_device_id)
#it will listen and then pass data to the network
#out put will be a note
#this line acts as a filter so you might need to adjust it or adjust the gain
#if (np.abs(librosa.stft(audio)).max() > 50)#change 50 to some value that filter out the noise but when played allows you the see the data
#here you pass your model 
#you can download the models from here(https://drive.google.com/drive/folders/1nAel5tOW0dtZzun5ydqDWmQXauPxRYlZ) or train your own by using train.py
#model = load_model('my_model_masive.h5')
#model = load_model('your_model_here.h5')

python3 live_data.py

```


