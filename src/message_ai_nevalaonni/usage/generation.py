import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

class Generation:
    def __init__(self, model_path:str):
        self.model = load_model(model_path)
        with open("/Users/nevalaonni/MessageAi/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

    def generate(self,seed:str, word_amount:int = 5):
        for _ in range(word_amount):
            token_list = self.tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.model.layers[0].input_shape[1], padding="pre")

            predicted_probs = self.model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

            for word, index in self.tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break

            seed += " " + output_word
        return seed

if __name__ == "__main__":
    print(Generation("/Users/nevalaonni/MessageAi/nevalaonni.h5").generate("I love penis they are "))