import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.python.client import device_lib
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import random
import os
import json
import numpy as np
import pickle
from time import strftime, localtime

from dbg.logger import Logger

l = Logger("learning.py")
BATCH_SIZE = 64

class Learning:
    def __init__(self, batch_size:int=64):
        self.batch_size = batch_size
        pass

    def train_based_off_sentences(self,sentences:list, iterations=10, new_model_path:str=None) -> str:
        if new_model_path == None:
            new_model_path = f"model-{strftime('%d_%m_%Y-%H_%M', localtime())}.h5"
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences) # builds a vocab based off of sentences

        l.debug("Tokenizer converted sentences into sequences")

        X = []
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        l.debug("Inputs and labels generated")

        maxlen = max([len(x) for x in X]) 
        X_padded = pad_sequences(X, maxlen=maxlen, padding="pre")
        y = np.array(y) 

        l.debug("Padded sequences")

        model = Sequential()
        model.add(Embedding(input_dim=100000, output_dim=128, input_length=maxlen)) # set to 100_000 just to future proof as much as possible
        model.add(LSTM(64)) 
        model.add(Dense(100000, activation="softmax")) 

        l.debug("Configured model")

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        l.debug("Model compiled")

        model.fit(X_padded, y, epochs=iterations, batch_size=self.batch_size)
        model.save(new_model_path) 

        l.debug("Finished model training")

        with open("tokenizer.pkl", "wb") as handle: 
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return new_model_path
    
    def add_training_to_model(self, tokenizer, model_path:str, new_sentences:list, new_model_path:str=None):
        if new_model_path == None:
            new_model_path = f"model-{strftime('%d_%m_%Y-%H_%M', localtime())}.h5"

        model = load_model(model_path)
        sequences = tokenizer.texts_to_sequences(new_sentences)

        X = []
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        X_padded = pad_sequences(X, maxlen=model.layers[0].input_shape[1], padding="pre")
        y = np.array(y)

        model.fit(X_padded, y, epochs=8, validation_data=(X_padded, y), batch_size=self.batch_size)
        model.save(new_model_path)


    def continious_training_start(self,tokenizer,model_path:str, iterations:int, sentences:list, new_model_path:str = None) -> None:
        if new_model_path == None:
            new_model_path = f"model-{strftime('%d_%m_%Y-%H_%M', localtime())}.h5"

        sequences= tokenizer.texts_to_sequences(sentences)
        X = []
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        model = load_model(model_path)
        X_padded = pad_sequences(X, maxlen=model.layers[0].input_shape[1], padding="pre")
        y = np.array(y)

        try:
            model.fit(X_padded, y, epochs=iterations,batch_size=self.batch_size, validation_data=(X_padded, y)) 
        except KeyboardInterrupt:
            model.save("BACKUP_INTERRUPTED_MODEL.h5")
        model.save(new_model_path)

