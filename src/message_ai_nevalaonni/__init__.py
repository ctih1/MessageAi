from usage.generation import Generation
from learning.learning import Learning
from data.extractor import Extractor
from webserver import server
from discordbot import bot
import os
from dotenv import load_dotenv
import pickle
import json
import logging
import sys

load_dotenv()

def main():                                                             
    # loc:str = input("Enter location of your discord data request: ")
    #valid = Extractor(loc).valid
    #while not valid:
        #loc:str = input("Invalid path! Enter a suitable path.")
        #valid = Extractor(loc).valid

    #with open(r"C:\Users\nevalaonni\Desktop\MessageAi\src\tokenizer.pkl", "rb") as f:
    #    tokenizer = pickle.load(f)

    #with open(r"C:\Users\nevalaonni\Desktop\MessageAi\messages.txt2", "r") as f:
    #    new_sentences = json.load(f)

    #Learning().add_training_to_model(tokenizer,r"C:\Users\nevalaonni\Desktop\MessageAi\src\nevalaonni.h5",new_sentences)

    #return
    #server.app.run("0.0.0.0",8080,use_reloader=False)

    logging.StreamHandler(sys.stdout)
    logger = logging.getLogger("ma")
    logging.basicConfig(filename='message_ai.log', level=logging.DEBUG)

    logger.info("Starting bot")

    if not os.getenv("BOT_TOKEN"):
        logger.error("Discord bot token not defined in .env")
        return

    if not os.getenv("MODEL_PATH"):
        logger.error("Model path not defined in .env")
        return
    
    os

    if not os.getenv("TOKENIZER_PATH"):
        logger.error("Tokenizer path not defined in .env")
        return

    bot.start(os.environ["BOT_TOKEN"])

    
main()