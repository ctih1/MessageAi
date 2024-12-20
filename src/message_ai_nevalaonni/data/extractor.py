import os
from typing import TypedDict
import json
import logging

logger = logging.getLogger("dma")
logger.level = logging.DEBUG

class Extractor:
    def __init__(self, file_location: str):
        """
        file_location should be the root of the discord package (README.md, messages)
        """
        self.file_location = file_location
        self.valid = self.__check_is_valid()
        self.convo_data = {} # {str id: str name}

        self.sentences = []

        self.get_channel_name_id()
        logger.debug(f"Is valid? {self.valid}")

    def __check_is_valid(self) -> bool:
        try:
            return "messages" in os.listdir(self.file_location)
        except FileNotFoundError:
            return False
        
    def get_channel_name_id(self):
        if not self.valid:
            raise AttributeError("File path is not valid!")
        with open(os.path.join(self.file_location,"messages","index.json"),"r", encoding="utf-8") as f:
            self.convo_data = json.load(f)
        return self.convo_data

    def loop_over_folders(self):
        message_path = os.path.join(self.file_location,"messages")
        for folder in [f.path for f in os.scandir(message_path) if f.is_dir()]:
            folder_path = os.path.join(message_path, folder)
            logger.debug(folder_path)

            with open(os.path.join(folder_path,"channel.json") ,"r") as f:
                channel_id = json.load(f)["id"]
                logger.info(f"Processing channel {self.convo_data[channel_id]} (folder {folder})")

            with open(os.path.join(folder_path,"messages.json") ,"r", encoding="utf-8") as f:
                messages = json.load(f)
            
            for message in messages:
                self.sentences.append(message["Contents"])
    
    def get_messages(self) -> list:
        return self.sentences