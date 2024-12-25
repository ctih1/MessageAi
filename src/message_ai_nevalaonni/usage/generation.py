import numpy as np
print("Loading TensorFlow library... This might take a bit")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
import gc

import pickle

class Generation:
    def __init__(self, model_path:str):
        self.model = load_model(model_path)
        self.model_path = model_path # used for bot statistics
        with open(os.getenv("TOKENIZER_PATH"), "rb") as f:
            self.tokenizer = pickle.load(f)

    def generate(self,seed:str, word_amount:int = 5):
        for _ in range(word_amount):
            token_list = self.tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.model.layers[0].input_shape[1], padding="pre")

            output_word = ""

            predicted_probs = self.model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

            for word, index in self.tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break
            
            output_word = output_word or "Ion know brah"

            seed += " " + output_word
        return seed
    
    def free(self): 
        tf.keras.backend.clear_session()
    
    def reinit(self):
        self.model = load_model(self.model_path)

if __name__ == "__main__":
    print(Generation(r"C:\Users\nevalaonni\Desktop\MessageAi\src\nevalaonni.h5.new").generate(""))