import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import tensorflow as tf
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)

# Paths to data features, test song features and model:

TRAIN = './Features/dataset_features/data_features.csv'
TEST = './Features/test_songs_features/test_features.csv'
TEST_SPLIT = './Features/test_songs_features/test_features_split.csv'
MODEL = './Models/FC_NN.h5'

# Dictionary for genres label encoding:
GENRES = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
          5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}


def model_load(path=MODEL, show=False):
    """
    Load previously trained model
    """
    modelo = load_model(path)
    if show:
        print(modelo.summary())
    return modelo


def normalize(path):
    """
    Normalize test data for prediction
    """
    # Training data:
    data = pd.read_csv(TRAIN)
    X = data.drop('genre', axis=1)
    # Test data:
    tdata = pd.read_csv(path)
    Xt = tdata.drop('genre', axis=1)
    yt = tdata.genre
    rows = Xt.shape[0]
    # Append original data and test data for normalization:
    Xt = X.append(Xt, ignore_index=True)
    # Normalize:
    sc = StandardScaler().fit_transform(Xt.values)
    Xt = pd.DataFrame(sc[-rows:], index=Xt[-rows:].index, columns=Xt.columns)
    splits = rows/10
    return Xt, yt, splits


def predict(model, Xt, yt, splits):
    """
    Predicts labels for each song
    """
    g = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    evaluation = model.evaluate(Xt, yt)
    accuracy = round(evaluation[1], 3)
    preds = model.predict_classes(Xt)
    predictions = [(g[i], GENRES.get(preds[i])) for i in range(len(preds))]
    probs = model.predict(Xt)
    return accuracy, predictions, probs


def present_results(acc, preds, probs):
    """
    Present results of demo1
    """

