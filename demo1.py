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
TEST = './Features/single_song_features/song_features.csv'
MODEL = './Models/FC_NN.h5'

# Dictionary for genres label encoding:
genres = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
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
    Xt = pd.read_csv(path)
    rows = Xt.shape[0]
    # Append original data and test data for normalization:
    Xt = X.append(Xt, ignore_index=True)
    # Normalize:
    sc = StandardScaler().fit_transform(Xt.values)
    Xt = pd.DataFrame(sc[-rows:], index=Xt[-rows:].index, columns=Xt.columns)
    return Xt


def predict(model, Xt):
    """
    Predicts labels for each song
    """
    preds = model.predict_classes(Xt)
    predicted = genres.get(preds[0])
    probs = model.predict(Xt)[0]
    return predicted, probs


def present_results(preds, probs):
    """
    Present results of demo1
    """
    plt.figure(figsize=(10,10))
    plt.title(f'Predicted genre: {preds}')
    plt.bar(genres.values(), probs, color='g')
    for j in range(len(probs)):
        plt.text(x=j - 0.1, y=probs[j], s='{:.2f} %'.format((probs[j]) * 100), size=10)
    plt.savefig('./Output/demo1.png')
    plt.show()


# FUNCTIONS TO INCLUDE IN main.py

def execute_demo1():
    modelo=model_load()
    Xt = normalize(TEST)
    predictions,probs = predict(modelo, Xt)
    present_results(predictions,probs)
