from tensorflow.keras.models import load_model
import logging

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
