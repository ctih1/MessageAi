import os
import json
from dbg.logger import Logger
from typing import TypedDict, List
from tqdm import tqdm

l = Logger("telegram/extraction.py")

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
class Extraction:
    def __init__(self, file_location:str, ignored:list=[]):
        self.file_location = file_location
        self.valid = self.__check_is_valid()
        self.sentences = []
        self.ignored_chats = ignored

        l.debug(f"Is valid? {self.valid}")

    def __check_is_valid(self) -> bool:
        try:
            return "result.json" in os.listdir(self.file_location)
        except FileNotFoundError:
            return False

    def loop_over_folders(self, target_author:str): 
        l.debug("Opening telegram package...")
        with open(os.path.join(self.file_location,"result.json"),"r",encoding="UTF-8") as f:
            json_data = json.load(f)
        l.debug("Loaded package succesfully")

        chats:MessageObject = json_data["chats"]["list"]

        for chat in tqdm(chats):
            tqdm.write("Iterating over chats")

            if chat.get("name") is None: # chat[name] might itself be null, since saved messages
                tqdm.write("Chat has no name. If you have the 'saved messages' feature turned on, you can safely ignore this error.")
                continue

            if chat.get("name") in self.ignored_chats:
                continue

            if chat.get("type") == "bot_chat": 
                l.write("Skipping over chat: reason = bot")
                continue

            for message in chat["messages"]:
                message:Message = message 
                if message["type"] != "message": 
                    l.write("Skipping over message: reason = not a message")
                    continue
                if not message["text"]:
                    l.write("Skipping over message: reason = empty message")
                    continue

                if isinstance(message["text"], list):
                    l.debug("Message is in list format")
                    a = message["text"].copy()
                    message["text"] = ""
                    for item in a:
                        if isinstance(item,dict):
                            l.debug("Message item is in dict format")
                            if item["type"] != "plain": 
                                l.debug("Skipping over chat: reason = not plain")
                                continue
                            message["text"] += item["text"]
                        else:
                            message["text"] += item

                if not isinstance(message["text"],str):
                    if message["text"]["type"] != "plain":
                        continue

                if message["from"].lower() != target_author.lower():
                    l.debug("Skipping over message: reason = not correct author")
                    continue
                self.sentences.append(message["text"])

        return self
    
    def get_messages(self) -> list:
        return self.sentences
