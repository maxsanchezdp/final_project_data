# Music Classifier

As my final project for Ironhack's Data Analytics Bootcamp I decided to develop a music classification system that would use Neural Networks to predict the genre of new songs. In this repository you can find all the necessary files to run the script and test it yourself. You can also train/test the model with your own music if you wish, I've added a *feature extraction* and a *train/test* mode using **argparse** for that purpose (I will explain how later on).

So, this is a step by step guide for you to follow if you wish to try my **Music Classifier** yourself. *On this guide I will also explain the methodology I followed and the results I got*:

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


# (THIS IS A WORK IN PROGRESS)





