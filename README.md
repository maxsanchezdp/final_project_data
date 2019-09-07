# Music Classifier

As my final project for Ironhack's Data Analytics Bootcamp I decided to develop a music classification system that would use Neural Networks to predict the genre of new songs. In this repository you can find all the necessary files to run the script and test it yourself. You can also train/test the model with your own music if you wish, I've added a *feature extraction* and a *train/test* mode using **argparse** for that purpose (I will explain how later on).

So, this is a step by step guide for you to follow if you wish to try my **Music Classifier** yourself. On this guide I will also explain the methodology I followed and the results I got:

## 0. Fork/clone this repo:
As usual, the first step is to fork this repo and clone/download it on your computer. That way you will have the same folder structure and everything should run smoothly if you take into account the following considerations:

### i. Train data acquisition and conversion:

I haven't uploaded the songs data set I used to train and test the model, but you can find it here: http://opihi.cs.uvic.ca/sound/genres.tar.gz.

Just dowload and extract the files in the **Data** folder of this project. The structure should look like this:

![file_structure](./for_md/0_file_structure.png)

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format. You will also need to convert each file from .au to .wav (I used SoX for this: http://sox.sourceforge.net/).

Also, I left out of one song of each genre so I could use them later for testing (you can find these in the **Data/test_songs/** folder), so the actual training of the neural network is done with the first 99 songs of each genre. If you wish to test the training mode, make sure to remove the last song of each genre so you can replicate my results.

### ii. Python version, libraries and software:

This project uses **Python 3.6** as more recent versions are not supported by Keras (one of the main libraries/frameworks I've used). I would recommend you to create a virtual environment (using conda, virtualenv, or whichever you prefer) and installing the Python version and libraries specified in the **requirements.txt**. You should make sure you have installed:

|  Name 	|  Description 	|
|---	|---	|
|   [Python 3.6](https://www.python.org/downloads/release/python-369/)	|   Python version currently supported by Keras	|
|   [Librosa](https://librosa.github.io/librosa/)	|   Python package for music and audio analysis	|
|   [Pandas](https://pandas.pydata.org/)	|   Provides easy-to-use data structures and data analysis tools for Python	|
|  [Numpy](https://numpy.org/)	|   Python package for scientific computing	|
|   [Scipy](https://www.scipy.org/)	|   Provides many user-friendly and efficient numerical routines for python 	|
|   [Matplotlib](https://matplotlib.org/)	|   Python 2D plotting library	|
|   [Scikit-learn](https://scikit-learn.org/stable/)	|   Machine learning library for Python	|
|   [TensorFlow](https://www.tensorflow.org/)	|   Machine learning library for Python	|
|   [Keras](https://keras.io/)	|   Neural Networks library for Python that runs on TensorFlow	|
|   [Urllib3](https://urllib3.readthedocs.io/en/latest/)	|   HTTP client for Python	|
|   [Python-dotenv](https://pypi.org/project/python-dotenv/)	|   Reads the key,value pair from .env file and adds them to environment variable	|
|   [Spotipy](https://spotipy.readthedocs.io/en/latest/)	|   Python library for the Spotify Web API	|
|   [mpg123](https://www.mpg123.de/index.shtml)	|   MPEG console audio player and decoder for Linux, Windows and Mac	|

### iii. Spotify developer account:

I've developed an integration with Sporify's Web API that allows the user to search and download previews of new songs and then test the model with these. It's a pretty cool feature and I highly recommend you to try it. You'll need **Spotify developer** credentials to make this feature work, but don't worry, it's very easy to set it all up:

**1. Connect Spotify developer to your Spotify account:** it takes less than 5 minutes, you just need to go to this [website](https://developer.spotify.com/dashboard/) and click on *"Sign up for a free Spotify account here."*. Then click on *Create a Client ID* and follow the instructions. You will get a *Client ID* and a *Client Secret*.

**2. Add your ID and Secret to environment variables:** create a *.env* file in the **root** folder, open it with your favourite text editor and type your ID as *CID = "WhateverYourClientIdIs"* and your Secret as *SECRET = "WhateverYourSecretIs"*.

### Once you're done with this, everything should work like a charm

## 1. *Feature extraction* and *train* mode:

As I mentioned before, you can train the model with your own music if you want to. You just need to make sure to put your music in the **Data/genre/** folder as I previously showed. All files must be correctly labelled. However, if you wish to add new/different genres you'll to tweak the code a little bit (just modify the *GENRES* dictionary on **feature_extraction.py** and **demo1.py**).

I will explain how it works with the original dataset:

### Feature extraction:

This is the first step of my project and also the first challenge. We are dealing with audio files which we need to "translate" into numbers and other types of data that we can later use to train the Neural Network. I used a library called **Librosa** 
# (THIS IS A WORK IN PROGRESS)





