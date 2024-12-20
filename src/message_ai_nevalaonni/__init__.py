from usage.generation import Generation
from learning.learning import Learning
from data.extractor import Extractor
from discordbot import bot
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # loc:str = input("Enter location of your discord data request: ")
    #valid = Extractor(loc).valid
    #while not valid:
        #loc:str = input("Invalid path! Enter a suitable path.")
        #valid = Extractor(loc).valid


    bot.start(os.environ["BOT_TOKEN"])
    
main()