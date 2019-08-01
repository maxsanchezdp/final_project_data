# Music classifier
The dataset I'll be using for this project can be found here: http://marsyas.info/downloads/datasets.html
(credits pending). I had to convert all files to .wav format (using SoX) as .au was not supported by the libraries I used to process files. You can find the converted dataset in here: https://drive.google.com/open?id=1FZXHpyulIBo6G6Gu8HcK7fcL1mmhqE4F. It's made of 1000 audio clips, each of them 30s long and labeled by music genre.

Develop/train an algorithm that takes an audio file like the ones in the dataset and classifies it into one of the trained genres, also giving a % o similarity with the rest of genres.

First we need to "translate" the audio files into something we can analyse and find patterns in. I need to extract key features that will help a model identify different genres.

El espectrograma se puede interpretar como una proyección en dos dimensiones de una sucesión de Transformadas de Fourier de tramas consecutivas, donde la energía y el contenido frecuencial de la señal va variando a lo largo del tiempo. (https://es.wikipedia.org/wiki/Espectrograma)





