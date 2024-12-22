import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random
import os
import json
import numpy as np
import pickle
import logging

logger = logging.getLogger("ma")

class Learning:
    def __init__(self):
        print("Available devices:")
        print(device_lib.list_local_devices())
        pass

    def train_based_off_sentences(self,sentences:list):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences) # builds a vocab based off of sentences

        print("model init")

        X = [] # subsequences of word indexes 
        y = [] # next word in sentence for each subseq
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        maxlen = max([len(x) for x in X]) 
        X_padded = pad_sequences(X, maxlen=maxlen, padding="pre")  # make every subseq the same length
        y = np.array(y) 

        model = Sequential()
        model.add(Embedding(input_dim=100000, output_dim=128, input_length=maxlen)) # meaning of words
        model.add(LSTM(64)) # word "links", aka certain connected words, aka which words follow eachother
        model.add(Dense(100000, activation="softmax")) # finds the right words in vocab (tokenizer.texts_to_sequences)

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(X_padded, y, epochs=10, batch_size=64)
        model.save("model.h5") # save model for later use

        with open("tokenizer.pkl", "wb") as handle:  # save in case of emergency
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def add_training_to_model(self, tokenizer, model_path:str, new_sentences:list):
        model = load_model(model_path)
        sequences = tokenizer.texts_to_sequences(new_sentences)

        X = []
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        print("model init")

        X_padded = pad_sequences(X, maxlen=379, padding="pre")
        y = np.array(y)

        history = model.fit(X_padded, y, epochs=8, validation_data=(X_padded, y))
        checkpoint = ModelCheckpoint("nevalaonni_checkpoint.h5",save_best_only=True, monitor="val_loss", verbose=1)

        model.save(model_path + ".new")


    def continious_training_start(self,tokenizer,model_path:str, iterations:int, sentences:list, new_model_path:str = "new_trained_model.h5") -> None:
        sequences= tokenizer.texts_to_sequences(sentences)
        X = []
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        maxlen = max([len(x) for x in X])
        X_padded = pad_sequences(X, maxlen=maxlen, padding="pre")
        y = np.array(y)

        model = load_model(model_path)
        model.fit(X_padded, y, epochs=iterations,batch_size=64, validation_data=(X_padded, y)) # tweak batch_size according to ram size and gpu power
        model.save(new_model_path)


if __name__ == "__main__":
    with open("messages.txt","r") as f:
        a = np.array(json.load(f))
    Learning().train_based_off_sentences(a)