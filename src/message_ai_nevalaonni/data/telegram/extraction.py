import os
import json
import logging
from typing import TypedDict, List
from tqdm import tqdm


class Text(TypedDict):
    type_: str
    text: str

class TextEntity(TypedDict):
    type_: str
    text: str

class Message(TypedDict):
    id: int
    type_: str
    date: str
    date_unixtime: int
    from_: str
    from_id: str
    text: List[Text]
    text_entities: List[TextEntity]

class MessageObject(TypedDict):
    name: str
    type_: str
    id: int
    messages: List[Message]


logger = logging.getLogger("dma")


class Extraction:
    def __init__(self, file_location:str):
        self.file_location = file_location
        self.valid = self.__check_is_valid()
        self.sentences = []

        logger.debug(f"Is valid? {self.valid}")

    def __check_is_valid(self) -> bool:
        try:
            return "result.json" in os.listdir(self.file_location)
        except FileNotFoundError:
            return False

    def loop_over_folders(self, target_author:str): 
        with open(os.path.join(self.file_location,"result.json"),"r",encoding="UTF-8") as f:
            json_data = json.load(f)
        chats:MessageObject = json_data["chats"]["list"]

        for chat in tqdm(chats):
            if chat.get("name") is None: # chat[name] might itself be null, since saved messages
                continue
            if chat.get("type") == "bot_chat": 
                continue
            for message in chat["messages"]:
                message:Message = message 
                if message["type"] != "message": 
                    continue
                if not message["text"]:
                    continue

                if isinstance(message["text"], list):
                    a = message["text"].copy()
                    message["text"] = ""
                    for item in a:
                        if isinstance(item,dict):
                            if item["type"] != "plain": 
                                continue
                            message["text"] += item["text"]
                        else:
                            message["text"] += item
                if not isinstance(message["text"],str):
                    if message["text"]["type"] != "plain":
                        continue
                if message["from"].lower() != target_author.lower():
                    continue
                self.sentences.append(message["text"])

        return self
    
    def get_messages(self) -> list:
        return self.sentences
