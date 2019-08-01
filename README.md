# Music classifier
The dataset I'll be using for this project can be found here: http://marsyas.info/downloads/datasets.html
(credits pending). I had to convert all files to .wav format as .au was not supported byt the libraries I used to process files. You can finde the converted dataset in here: https://drive.google.com/open?id=1FZXHpyulIBo6G6Gu8HcK7fcL1mmhqE4F.

It's made of 1000 audio clips, each of them 30s long and labeled by music genre.
Develop/train an algorithm that takes an audio file like the ones in the dataset and classifies it into one of the trained genres, also giving a % o similarity with the rest of genres.

I had to batch convert all files from .au to .wav using SoX, as .au is not supported by Librosa.
