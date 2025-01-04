import numpy as np
print("Loading TensorFlow library... This might take a bit")
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.backend import clear_session
import tensorflow as tf
from keras.engine.sequential import Sequential
import os
from dbg.logger import Logger
import pickle

l = Logger("generation.py")

class Generation:
    def __init__(self, model_path:str):
        self.model:Sequential = load_model(model_path)
        self.model_path = model_path # used for bot statistics
        with open(os.getenv("TOKENIZER_PATH"), "rb") as f:
            self.tokenizer = pickle.load(f)

    def generate(self,seed:str, word_amount:int = 5):
        for _ in range(word_amount):
            token_list = self.tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.model.layers[0].input_shape[1], padding="pre")

            output_word = "" # Sometimes model fails to predict the word, so using a fallback 

            predicted_probs = self.model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

            for word, index in self.tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break

            seed += " " + output_word
        return seed
    
    def unalloc_model(self):
        l.debug("Clearing session...")
        clear_session()
    
    def reinit_model(self):
        l.debug("Reinit session...")
        self.model = load_model(self.model_path)

if __name__ == "__main__":
    print(Generation(r"C:\Users\nevalaonni\Desktop\MessageAi\src\nevalaonni.h5.new").generate(""))