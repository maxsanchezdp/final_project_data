import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Paths to data features, test song features and model:

TRAIN = './Features/dataset_features/data_features.csv'
TEST = './Features/test_songs_features/test_features.csv'
TEST_SPLIT = './Features/test_songs_features/test_features_split.csv'
MODEL = './Models/FC_NN.h5'


def model_load(path, show=False):
    modelo = load_model(path)
    if show:
        print(modelo.summary())
    return modelo



