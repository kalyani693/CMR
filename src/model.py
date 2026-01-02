from sklearn.ensemble import IsolationForest
import tensorflow as tf
from keras import layers,models
from keras.models import Model

def train_model(x):
    assert x.ndim==2
    model=IsolationForest(n_estimators=200,contamination=0.05,random_state=42)
    model.fit(x)
    return model


#1D convolutional autoencoder/dense

def build_autoencoder(window_size):
    input_layer=layers.Input(shape=(window_size,1))
    x=(input_layer).flatten()
    x=layers.Dense(64,activation="relu")(x)
    encoded=layers.Dense(16,activation="relu")(x)


    x=layers.Dense(64,activation="relu")(encoded)
    x=layers.Dense(256,activation="linear")(x)
    decoded=(window_size,1)(x)
    
    autoencoder=Model(input_layer,decoded)
    autoencoder.compile(optimizer="adam",loss="mse")

    return autoencoder

