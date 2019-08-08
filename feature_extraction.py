import pandas as pd
import numpy as np
import os
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew

# directories
AUDIO_DIR = './Data/genres/'

# Dictionary for genres label encoding:
GENRES = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}


def split_songs(X, window, overlap):
    """
    Function to split a song into multiple songs.
    """

    # Temporary lists to hold results
    temp_X = []

    # Get input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)


def get_features(y, sr, n_fft=1024, hop_length=512):
    """
    Get selected features for a song using numpy and librosa
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
    def get_moments(descriptors):
        result = {}
        for k, v in descriptors.items():
            result[f'{k}_mean'] = np.mean(v)
            result[f'{k}_std'] = np.std(v)
            result[f'{k}_kurtosis'] = kurtosis(v)
            result[f'{k}_skew'] = skew(v)
        return result
    
    dict_agg_features = get_moments(features)
    
    # Calculating one more feature:
    dict_agg_features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]
    
    return dict_agg_features


def read_process_labelled(src_dir, window=0.2, overlap=0.5, debug=True):
    """
    Read and process labelled songs (train/test data, demo test data)
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
                    print(f"Reading file: {file_name}")
                    
                # Split songs:
                samples = split_songs(signal, window, overlap)

                # Append the result to the data structure
                for s in samples:
                    features = get_features(s, sr)
                    features['genre'] = GENRES[x]
                    arr_features.append(features)

    return arr_features


def read_process_unlabelled(src_dir, window=1, overlap=0, debug=True):
    """
    Read and process unlabelled songs (spotify data)
    """
    
    arr_features = []

    for root, subdirs, files in os.walk(src_dir):
        for file in files:
            # Read the audio file
            file_name = src_dir + file
            signal, sr = librosa.load(file_name)
            signal = signal[:660000]

            # Debug process
            if debug:
                print("Reading file: {}".format(file_name))
                
            # Split songs:
            samples = split_songs(signal, window, overlap)

            # Append the result to the data structure
            for s in samples:
                features = get_features(s, sr)
                arr_features.append(features)
    return arr_features


def read_process_song(path, window=1, overlap=0, debug=True):
    """
    Read and process a single song
    """

    arr_features = []

    signal, sr = librosa.load(path)
    signal = signal[:660000]

    # Debug process
    if debug:
        print("Reading file: {}".format(path))

    # Split songs:
    samples = split_songs(signal, window, overlap)

    # Append the result to the data structure
    for s in samples:
        features = get_features(s, sr)
        arr_features.append(features)
    return arr_features


# FUNCTIONS TO INCLUDE IN main.py


def create_train_feats():
    """
    Builds .csv used to train the model
    """
    features = read_process_labelled(AUDIO_DIR, debug=True)
    df = pd.DataFrame(features)
    df.to_csv('./Features/dataset_features/data_features.csv', index=False)


def create_demo_feats():
    """
    Builds .csv used to test the model (demo)
    """
    features = read_process_labelled(TEST_DIR, window=1, overlap=0, debug=True)
    df = pd.DataFrame(features)
    df.to_csv('./Features/test_songs_features/test_features.csv', index=False)
    features = read_process_labelled(TEST_DIR, window=1/3, overlap=0, debug=True)
    df = pd.DataFrame(features)
    df.to_csv('./Features/test_songs_features/test_features_split.csv', index=False)


def create_song_feats(path):
    """
    Builds .csv used to test the model on a single song
    """
    features = read_process_song(path, debug=True)
    df = pd.DataFrame(features)
    df.to_csv('./Features/single_song_features/song_features.csv', index=False)
