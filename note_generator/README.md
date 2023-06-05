```sh
#build docker image for training the data set
sudo docker build -t test .
#train the data set with GPU
sudo docker run -it --volume="./app:/app:rw" --gpus all test /bin/bash
```
Trained nreual networks -> https://drive.google.com/drive/folders/1nAel5tOW0dtZzun5ydqDWmQXauPxRYlZ?usp=sharing
