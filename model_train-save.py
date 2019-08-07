import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models
from keras import layers
import warnings
warnings.filterwarnings('ignore')

# Path to training data:
PATH = "./Features/dataset_features/data_features.csv"


def prepare_data(path, test_size=0.2):
    """
    Prepares data in given path to feed the model
    """
    data = pd.read_csv(path)
    X = data.drop('genre', axis=1)
    y = data.genre

    # Normalize data in X:
    scaled_features = StandardScaler().fit_transform(X.values)
    X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def define_model(x_train, show=True):
    """
    Defines Keras model to be trained
    """
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dropout(rate=0.25, noise_shape=None, seed=None))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.25, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(rate=0.25, noise_shape=None, seed=None))
    model.add(layers.Dense(10, activation='softmax'))
    if show:
        print(model.summary())
    return model


def train_save_model(model, xtrain, xtest, ytrain, ytest, epochs=60, batch_size=265, val_split=0.1):
    """
    Trains the model with the given data and set parameters (epochs)
    """
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(xtrain,
                        ytrain,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=val_split,
                        verbose=0)
    train_acc = round(history.history['acc'][-1], 3)
    train_loss = round(history.history['loss'][-1], 3)
    val_acc = round(history.history['val_acc'][-1], 3)
    val_loss = round(history.history['val_loss'][-1], 3)
    print(f'Train accuracy: {train_acc} - Train loss: {train_loss}')
    print(f'Validation accuracy: {val_acc} - Validation loss: {val_loss}')
    test_loss, test_acc = model.evaluate(xtest, ytest)
    test_loss = round(test_loss, 3)
    test_acc = round(test_acc, 3)
    print(f'Test accuracy: {test_acc} - Test loss: {test_loss}')
    file_name = './Models/FN_NN_n.h5'
    model.save(file_name)
    return file_name

# FUNCTIONS TO INCLUDE IN main.py


def execute_mts(path):
    print('Preparing train data...')
    X_train, X_test, y_train, y_test = prepare_data(path)
    print('Defining, training and saving model...')
    modelo = define_model(X_train)
    file = train_save_model(modelo, X_train, X_test, y_train, y_test)
    print(f'Done! You can find your model at: {file}')






