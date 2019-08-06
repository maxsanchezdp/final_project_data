import pandas as pd
import numpy as np
import os
import pathlib
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew

AUDIO_DIR = './Data/genres/'  # Path to training data

# Dictionary for genres label encoding:
GENRES = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}


def split_songs(X, window=0.2, overlap=0.5):
    """
    Splits a song into several ones based on window and overlap
    """
    # Temporary lists to hold results
    temp_X = []

    # Get input song array size and calculate size of splits and overlap
    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]

    # Append new songs to temporary list
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)


# Get selected features from each song using librosa and numpy:

def get_features(y, sr, n_fft=1024, hop_length=512):
    """
    Calculate and extract features for each song
    """

    # Selected features:
    features = {'centroid': None, 'roloff': None, 'flux': None, 'rmse': None, 'zcr': None, 'chroma': None}

    # Using librosa to calculate the features
    features['centroid'] = librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['roloff'] = librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['rmse'] = librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['flux'] = librosa.onset.onset_strength(y=y, sr=sr).ravel()
    features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()

    # Treatment of MFCC feature
    mfcc = librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    for idx, v_mfcc in enumerate(mfcc):
        features[f'mfcc_{idx}'] = v_mfcc.ravel()

    # Calculate statistics for each feature:
    def get_stats(descriptors):
        result = {}
        for k, v in descriptors.items():
            result[f'{k}_mean'] = np.mean(v)
            result[f'{k}_std'] = np.std(v)
            result[f'{k}_kurtosis'] = kurtosis(v)
            result[f'{k}_skew'] = skew(v)
        return result

    dict_agg_features = get_stats(features)

    # Calculating one more feature:
    dict_agg_features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]

    return dict_agg_features


def read_process_songs(src_dir, debug=True):
    """
    Read and process songs
    """

    arr_features = []

    # Read files from the folders
    for x, _ in GENRES.items():
        folder = src_dir + x

        for root, subdirs, files in os.walk(folder):
            for file in files:
                # Read the audio file
                file_name = folder + "/" + file
                signal, sr = librosa.load(file_name)
                signal = signal[:660000]

                # Debug process
                if debug:
                    print("Reading file: {}".format(file_name))

                # Split songs:
                samples = split_songs(signal)

                # Append the result to the data structure
                for s in samples:
                    features = get_features(s, sr)
                    features['genre'] = GENRES[x]
                    arr_features.append(features)

    return arr_features

