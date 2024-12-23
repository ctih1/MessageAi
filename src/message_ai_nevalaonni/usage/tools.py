from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np

logger = logging.getLogger("ma")
class Tools:
    @staticmethod
    def save_from_folder(folder_location:str, new_file_name:str):
        if not new_file_name.endswith(".h5"):
            logger.warning("Models that are compiled as models need to have the h5 file extension")
            return False
        model = load_model(folder_location)
        model.save(new_file_name)
        return True

    @staticmethod
    def evaluate(model_path,tokenizer, sentences):
        sequences = tokenizer.texts_to_sequences(sentences)
        X = []
        y = []

        model = load_model(model_path)
        
        for seq in sequences:
            for i in range(1, len(seq)):
                X.append(seq[:i])
                y.append(seq[i])

        X = pad_sequences(X,maxlen=model.layers[0].input_shape[1], padding="pre")
        y = np.array(y)

        l, accuracy = model.evaluate(X,y)

        return accuracy