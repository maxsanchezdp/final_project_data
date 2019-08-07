import pandas as pd
import matplotlib.pyplot as plt
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
    return Xt, yt


def predict(model, Xt, yt, splits):
    """
    Predicts labels for each song
    """
    preds = model.predict_classes(Xt)
    predictions = [(GENRES.get(yt[i]), GENRES.get(preds[i])) for i in range(len(preds))]
    probs = model.predict(Xt)
    return predictions, probs


def present_results(preds, probs):
    """
    Present results of demo1
    """
    for index, values in enumerate(probs):
        plt.subplot(5, 2, index + 1)
        plt.title(f'Real: {preds[index][0]} - Predicted: {preds[index][1]}')
        plt.bar(GENRES.values(), values)
    plt.savefig('./Output/demo1.png')
    plt.show()


# FUNCTIONS TO INCLUDE IN main.py

def execute_demo1():
    modelo=model_load()
    Xt, yt, splits = normalize(TEST)
    predictions, probs = predict(modelo, Xt, yt, splits)
    present_results(predictions,probs)



