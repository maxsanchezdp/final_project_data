# Music classifier
The dataset I'll be using for this project can be found here: http://marsyas.info/downloads/datasets.html
(credits pending)

It's made of 1000 audio clips, each of them 30s long and labeled by music genre.
Develop/train an algorithm that takes an audio file like the ones in the dataset and classifies it into one of the trained genres, also giving a % o similarity with the rest of genres.

I had to batch convert all files from .au to .wav using SoX, as .au is not supported by Librosa.
