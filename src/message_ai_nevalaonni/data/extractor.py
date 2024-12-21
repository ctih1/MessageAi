import os
from typing import TypedDict
import json
import logging
from data.discord.extraction import Extraction as dc
from data.telegram.extraction import Extraction as tg

logger = logging.getLogger("dma")
logger.level = logging.DEBUG

class Extractor:
    def __init__(self, file_location: str, **kwargs):
        self.file_location = file_location
        self.sentences = []
        self.kwargs = kwargs
    def extract(self,apps:dict):
        if "discord" in apps.keys():
            self.sentences.append(dc(apps["discord"]).loop_over_folders().get_messages())
        if "telegram" in apps.keys():
            self.sentences.append(tg(apps["telegram"]).loop_over_folders(self.kwargs["author"]).get_messages())
        
        with open("messages.txt2","w") as f:
            json.dump(self.sentences[0],f)



