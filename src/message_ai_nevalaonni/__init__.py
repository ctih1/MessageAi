import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from usage.generation import Generation
from usage.tools import Tools
from learning.learning import Learning
from data.extractor import Extractor
from webserver import server
from discordbot import bot
import tensorflow as tf
from dotenv import load_dotenv
import pickle
import json
import logging
import sys
import time
import json
from time import strftime, localtime
import sys


from dotenv import set_key

load_dotenv()

def b(a:str) -> bool:
    if a.lower() == "yes":
        return True
    if a.lower() == "no":
        return False
    return None

def assistant():
    print("Hello! Let's start training an AI for you")
    tg = b(input("Do you have a telegram data package downloaded? (yes/no): "))
    dc = b(input("Do you have a discord data package downloaded? (yes/no): "))

    if not dc and not tg:
        print("You need atleast one of the following packages to contiue.")
        quit(1)

    tgu = "" # if one isn't specified

    if tg:
        tgp = input("Enter the path to your telegram package. The folder you select should contain a result.json file ")
        tgu = input("Enter your telegram name (NOT @username). This is used to identify your messages. ")
    if dc:
        dcp = input("Enter the path to your discord package. The folder you select should contain a messages folder ")

    print("Beginning data extraction...")

    extract_args = {}

    if tg:
        extract_args["telegram"] = tgp
    if dc:
        extract_args["discord"] = dcp

    Extractor("",author=tgu).extract(extract_args)

    print("Data succesfully extracted!")

    with open("messages.txt","r") as f:
        sentences = json.load(f)

    if len(sentences) < 50000:
        print("Small sample size detected. It looks like you have less than 50k sentences, which might make the AI less accurate. Do you want to continue?")
        if not b(input("Continue? (yes/no): ")):
            print("Have a nice day")
            quit(1)
        

    if not tf.test.is_gpu_available():
        print("WARNING: TensorFlow could not find a suitable GPU. Do you want to continue with CPU based learning?")
        print("Note: If you have a CUDA compatible Graphics Card, usually installing NVIDIA CUDNN 8.1 and CUDA Toolkit 11.2 solves this issue")
        if not b(input("Continue with CPU based learning? (yes/no): ")):
            print("Have a nice day")
            quit(1)
        if b(input("Do you want to reduce the model quality for a faster training time? (yes/no): ")):
            pass
        
    
    print("Starting AI learning... This might take a bit. Do not turn off your computer!")

    Learning().train_based_off_sentences(sentences)

    print("Initial model created...")

    if not b(input("First model finished! Do you want to keep training the AI? (yes/no): ")):
        env_path = os.path.join(os.curdir, ".env")

        set_key(dotenv_path=env_path,key_to_set="MODEL_PATH", value_to_set=os.path.join(os.curdir, "model.h5"))
        set_key(dotenv_path=env_path,key_to_set="TOKENIZER_PATH", value_to_set=os.path.join(os.curdir, "tokenizer.pkl"))

        print("AI Configuration succesfull! Please relaunch this program to start up the bot")
        quit(0)

    print("Starting continious training...")

    iterations = int(input("How many iterations would you like? More = better. Default: 7\n"))

    Learning().continious_training_start(os.getenv("TOKENIZER_PATH"), "model.h5", iterations,sentences, "more_learned.h5")

    if b(print("Continious training done! Do you wish to delete the old model? (yes/no): ")):
        os.remove("model.h5")

    set_key(dotenv_path=env_path,key_to_set="MODEL_PATH", value_to_set=os.path.join(os.curdir, "more_learned.h5"))
    set_key(dotenv_path=env_path,key_to_set="TOKENIZER_PATH", value_to_set=os.path.join(os.curdir, "tokenizer.pkl"))

    print("AI Configuration succesfull! Please relaunch this program to start up the bot")

def find_models() -> list:
    models: list = []
    for file in os.listdir():
        if file.endswith(".h5"):
            models.append(
                {file:strftime('%d.%m.%Y %H.%M', localtime(os.stat(file).st_ctime))}
            )
    return models


def add_training(type:str):
        with open(os.getenv("TOKENIZER_PATH")) as f:
            tokenizer = pickle.load(f)
        try:
            with open("messages.txt","r", encoding="UTF-8") as f:
                sentences = json.load(f)
        except FileNotFoundError:
            location = input("Could not find messages.txt. Please provide the file path: ")
            with open(location,"r") as f:
                sentences = json.load(f)

        iterations = int(input("How many iterations would you like? More iterations make a better model, but will increase training time. Default: 7\n"))
        
        print("Available models: ")
        for model,index in enumerate(find_models()):
            print(f"{index}. {model.keys()[0]}: {model.values()[0]}")

        if type=="retrain":
             Learning().continious_training_start(tokenizer, "model.h5", iterations,sentences, "more_learned.h5")
        elif type=="addition":
            Learning().add_training_to_model(tokenizer, "")


def main():                                                             
    # loc:str = input("Enter location of your discord data request: ")
    #valid = Extractor(loc).valid
    #while not valid:
        #loc:str = input("Invalid path! Enter a suitable path.")
        #valid = Extractor(loc).valid

    #with open(r"C:\Users\nevalaonni\Desktop\MessageAi\src\tokenizer.pkl", "rb") as f:
    #    

    #with open(r"C:\Users\nevalaonni\Desktop\MessageAi\messages.txt2", "r") as f:
    #    new_sentences = json.load(f)

    #Learning().add_training_to_model(tokenizer,r"C:\Users\nevalaonni\Desktop\MessageAi\src\nevalaonni.h5",new_sentences)

    #return
    #server.app.run("0.0.0.0",8080,use_reloader=False)

    if len(sys.argv) >= 1:
        sys.argv.extend(["none","none","none","none"])

    if sys.argv[1] == "--easy-setup":
        assistant()

    if sys.argv[1] == "--cont-training":
        add_training("retrain")

    if sys.argv[1] == "--add-training":
        add_training("addition")

    # add_training("retrain")
       
       



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
    
   # logger.info("Compiling model...")


    with open(os.getenv("TOKENIZER_PATH"),"rb") as f:
        tokenizer = pickle.load(f)

    #with open(r"C:\Users\nevalaonni\Desktop\MessageAi\messages.txt","r") as f:
    #    sentences = json.load(f)
 
    # Learning().continious_training_start(tokenizer,os.getenv("MODEL_PATH"),7,sentences)

    # Tools.compile_from_folder(os.getenv("MODEL_PATH"), "nevalaonni-w-tg.h5")

    #Extractor("", author="Onni Nevala").extract({"discord":r"C:\Users\nevalaonni\Downloads","telegram":r"C:\Users\nevalaonni\Downloads\Telegram Desktop\DataExport_2024-12-21"})

    if not os.getenv("TOKENIZER_PATH"):
        logger.error("Tokenizer path not defined in .env")
        return
    
    # Learning().train_based_off_sentences()

    bot.start(os.environ["BOT_TOKEN"])

    
main()