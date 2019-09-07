# Music Classifier

As my final project for Ironhack's Data Analytics Bootcamp I decided to develop a music classification system that would use Neural Networks to predict the genre of new songs. In this repository you can find all the necessary files to run the script and test it yourself. You can also train/test the model with your own music if you wish, I've added a feature extraction and a train/test mode using **argparse** for that purpose(I will explain how later on).

So, this is a step by step guide for you to follow if you wish to try my **Music Classifier** yourself:

## 0. Fork/clone this repo:




The dataset I'll be using for this project can be found here: http://marsyas.info/downloads/datasets.html
(credits pending). I had to convert all files to .wav format (using SoX) as .au was not supported by the libraries I used to process files. You can find the converted dataset in here: https://drive.google.com/open?id=1FZXHpyulIBo6G6Gu8HcK7fcL1mmhqE4F. It's made of 1000 audio clips, each of them 30s long and labeled by music genre.

Develop/train an algorithm that takes an audio file like the ones in the dataset and classifies it into one of the trained genres, also giving a % o similarity with the rest of genres.

First we need to "translate" the audio files into something we can analyse and find patterns in. I need to extract key features that will help a model identify different genres.

# (THIS IS A WORK IN PROGRESS)





