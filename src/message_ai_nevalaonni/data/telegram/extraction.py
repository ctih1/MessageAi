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


logger = logging.getLogger("ma")


class Extraction:
    def __init__(self, file_location:str, ignored:list=[]):
        self.file_location = file_location
        self.valid = self.__check_is_valid()
        self.sentences = []
        self.ignored_chats = ignored

        logger.debug(f"Is valid? {self.valid}")

    def __check_is_valid(self) -> bool:
        try:
            return "result.json" in os.listdir(self.file_location)
        except FileNotFoundError:
            return False

    def loop_over_folders(self, target_author:str): 
        logger.debug("Opening telegram package...")
        with open(os.path.join(self.file_location,"result.json"),"r",encoding="UTF-8") as f:
            json_data = json.load(f)
        logger.debug("Loaded package succesfully")

        chats:MessageObject = json_data["chats"]["list"]

        for chat in tqdm(chats):
            logger.debug("Iterating over chats")

            if chat.get("name") is None: # chat[name] might itself be null, since saved messages
                tqdm.write("Chat has no name. If you have the 'saved messages' feature turned on, you can safely ignore this error.")
                continue

            if chat.get("name") in self.ignored_chats:
                continue

            if chat.get("type") == "bot_chat": 
                logger.debug("Skipping over chat: reason = bot")
                continue

            for message in chat["messages"]:
                message:Message = message 
                if message["type"] != "message": 
                    logger.debug("Skipping over message: reason = not a message")
                    continue
                if not message["text"]:
                    logger.debug("Skipping over message: reason = empty message")
                    continue

                if isinstance(message["text"], list):
                    logger.debug("Message is in list format")
                    a = message["text"].copy()
                    message["text"] = ""
                    for item in a:
                        if isinstance(item,dict):
                            logger.debug("Message item is in dict format")
                            if item["type"] != "plain": 
                                logger.debug("Skipping over chat: reason = not plain")
                                continue
                            message["text"] += item["text"]
                        else:
                            message["text"] += item

                if not isinstance(message["text"],str):
                    if message["text"]["type"] != "plain":
                        continue

                if message["from"].lower() != target_author.lower():
                    logger.debug("Skipping over message: reason = not correct author")
                    continue
                self.sentences.append(message["text"])

        return self
    
    def get_messages(self) -> list:
        return self.sentences
