```sh
#build docker image for training the data set
sudo docker build -f Dockerfile-gpu -t trainingcontainers .
#start the docker container
sudo docker run -it --volume="./app:/app:rw" --gpus all trainingcontainers /bin/bash
#to train the model
python3 ./train.py
```
Trained nreual networks -> https://drive.google.com/drive/folders/1nAel5tOW0dtZzun5ydqDWmQXauPxRYlZ?usp=sharing
