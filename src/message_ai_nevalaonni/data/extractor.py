import os
from typing import TypedDict
import json
import logging
from data.discord.extraction import Extraction as dc
from data.telegram.extraction import Extraction as tg

logger = logging.getLogger("ma")
logger.level = logging.DEBUG

class Extractor:
    def __init__(self, file_location: str, **kwargs):
        self.file_location = file_location
        self.sentences = []
        self.kwargs = kwargs

    def extract(self,apps:dict):
        if "discord" in apps.keys():
            logger.info("Extracting data from discord package...")
            self.sentences.extend(dc(apps["discord"]).loop_over_folders().get_messages())

        if "telegram" in apps.keys():
            logger.info("Extracting data from telegram package...")
            self.sentences.extend(tg(apps["telegram"]).loop_over_folders(self.kwargs["author"]).get_messages())
        
        with open("messages.txt","w") as f:
            json.dump(self.sentences,f)



