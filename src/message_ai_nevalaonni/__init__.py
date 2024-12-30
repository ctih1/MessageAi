import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from usage.generation import Generation
from usage.tools import Tools
from learning.learning import Learning
from data.extractor import Extractor
from dbg.logger import Logger
from webserver import server 
from discordbot import bot
import tensorflow as tf
from dotenv import load_dotenv
import pickle
import json
import sys
import time
import json
from time import strftime, localtime
import sys
import psutil
tf.config.optimizer.set_jit(True)
from dotenv import set_key
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
load_dotenv()

l:Logger = Logger("__init__.py")

DEFAULT_FIRST_TIME_ITERATIONS:int = 10
BATCH_SIZE = int(os.getenv("BATCH_SIZE") or 64)
env_path = os.path.join(os.curdir, ".env")

def b(a:str) -> bool:
    if a.lower() == "yes":
        return True
    if a.lower() == "no":
        return False
    return None

def clean_logs():
    files = sorted(os.listdir(os.path.join(".","logs")))
    if len(files) >= 5:
        os.remove(os.path.join(".","logs",files[0]))


def assistant(skip_extraction:bool=False, ignore_from:list=[]):
    total_mem_gb = round(psutil.virtual_memory().total / (1024**3))      
    iterations = DEFAULT_FIRST_TIME_ITERATIONS

    l.announcement("Hello! Let's start training an AI for you")

    if skip_extraction:
        l.info("Skipping data extraction")
        message_file = input("Enter the path to messages.txt (list of sentences): ")
        with open(message_file,"r") as f:
            sentences = json.load(f)
    else:
        tg = b(input("Do you have a telegram data package downloaded? (yes/no): "))
        dc = b(input("Do you have a discord data package downloaded? (yes/no): "))
        if not dc and not tg:
            l.warn("You need atleast one of the following packages to contiue.")
            quit(1)

        tgu = "" # if one isn't specified

        if tg:
            tgp = input("Enter the path to your telegram package. The folder you select should contain a result.json file ")
            tgu = input("Enter your telegram name (NOT @username). This is used to identify your messages. ")
        if dc:
            dcp = input("Enter the path to your discord package. The folder you select should contain a messages folder ")
        l.debug("Beginning data extraction...")
        extract_args = {}
        if tg:
            extract_args["telegram"] = tgp
        if dc:
            extract_args["discord"] = dcp
        Extractor("",author=tgu,ignored=ignore_from).extract(extract_args)

        l.info("Data succesfully extracted!")

        with open("messages.txt","r") as f:
            sentences = json.load(f)

    if len(sentences) < 50000:
        l.warn("Small sample size detected. It looks like you have less than 50k sentences, which might make the AI less accurate. Do you want to continue?")
        if not b(input("Continue? (yes/no): ")):
            l.announcement("Have a nice day")
            quit(1)
        

    if not tf.test.is_gpu_available():
        l.warn("TensorFlow could not find a suitable GPU. Do you want to continue with CPU based learning?")
        l.announcement("If you have a CUDA compatible Graphics Card, usually installing NVIDIA CUDNN 8.1 and CUDA Toolkit 11.2 solves this issue")
        if not b(input("Continue with CPU based learning? (yes/no): ")):
            l.announcement("Have a nice day")
            quit(1)
        if b(input("Do you want to reduce the model quality for a faster training time? (yes/no): ")):
            iterations = 8
    else:
        gpu_details = tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0])
        l.info(f"Found GPU {gpu_details.get('device_name','Unkown')}, which will be used for training...")
        compute_capability = float(f"{gpu_details.get('compute_capability',0)[0]}.{gpu_details.get('compute_capability',0)[1]}")

        if compute_capability > 8.5: # rtx 3090(ti), 4060ti, 4070ti and higher
            if b(input("Looks like your GPU is very powerful. Would you like to increase the model qualiy in exchange for longer learning time? (yes/no): ")):
                iterations = 20
        elif compute_capability > 7.5:  # rtx 20- series cards, and 1650ti
            if b(input("Looks like your GPU is powerful. Would you like to increase the model qualiy in exchange for longer learning time? (yes/no): ")):
                iterations = 17
        elif compute_capability > 7:
            if b(input("Looks like your GPU is above average in matrix multiplication. Would you like to increase the model qualiy in exchange for longer learning time? (yes/no): ")):
                iterations = 17
        elif compute_capability > 6:
            l.info("GPU Power normal... Not changing model quality")

        elif compute_capability > 5:
            if b(input("Looks like your GPU is below average in matrix multiplication. Would you like to decrease the model qualiy in exchange for shorter learning time? (yes/no): ")):
                iterations = 8
        elif compute_capability > 3:
            if b(input("Looks like your GPU is not good in matrix multiplication. Would you like to decrease the model qualiy in exchange for shorter learning time? (yes/no): ")):
                iterations = 6
        elif compute_capability > 2:
            if b(input("Looks like your GPU is bad in matrix multiplication. Would you like to decrease the model qualiy in exchange for shorter learning time? (yes/no): ")):
                iterations = 4
        
    if b(input(f"Current iterations selected: {iterations} Do you wish to override this value? Do not change unless you know what you're doing. Override? (yes/no): ")):
        iterations = int(input("How many iterations do you want to run? "))
        

    l.info("Starting training... This will take a bit. Do not turn off your computer!")

    Learning().train_based_off_sentences(sentences,iterations)

    l.debug("Initial model created...")

    if not b(input("First model finished! Do you want to keep training the AI? (yes/no): ")):
        env_path = os.path.join(os.curdir, ".env")

        set_key(dotenv_path=env_path,key_to_set="MODEL_PATH", value_to_set=os.path.join(os.curdir, "model.h5"))
        set_key(dotenv_path=env_path,key_to_set="TOKENIZER_PATH", value_to_set=os.path.join(os.curdir, "tokenizer.pkl"))

        l.announcement("AI Configuration succesfull! Please relaunch this program to start up the bot")
        quit(0)

    l.debug("Starting continious training...")

    iterations = int(input("How many iterations would you like? More = better. Default: 7\n"))

    Learning().continious_training_start(os.getenv("TOKENIZER_PATH"), "model.h5", iterations,sentences, "more_learned.h5")

    if b(input("Continious training done! Do you wish to delete the old model? (yes/no): ")):
        os.remove("model.h5")

    set_key(dotenv_path=env_path,key_to_set="MODEL_PATH", value_to_set=os.path.join(os.curdir, "more_learned.h5"))
    set_key(dotenv_path=env_path,key_to_set="TOKENIZER_PATH", value_to_set=os.path.join(os.curdir, "tokenizer.pkl"))

    l.announcement("AI Configuration succesfull! Please relaunch this program to start up the bot")

def find_models() -> list:
    models: list = []
    for file in os.listdir():
        if file.endswith(".h5"):
            models.append(
                {file:strftime('%d.%m.%Y %H.%M', localtime(os.stat(file).st_ctime))}
            )
    models.sort(key=lambda x:list(x.values())[0])
    return models


def add_training(type_:str):
        with open(os.getenv("TOKENIZER_PATH"),"rb") as f:
            tokenizer = pickle.load(f)
        try:
            with open("messages.txt","r", encoding="UTF-8") as f:
                sentences = json.load(f)
        except FileNotFoundError:
            location = input("Could not find messages.txt. Please provide the file path: ")
            with open(location,"r") as f:
                sentences = json.load(f)

        iterations = int(input("How many iterations would you like? More iterations make a better model, but will increase training time. Default: 7\n"))
        
        l.announcement("Available models: ")
        for index, model in enumerate(find_models(),1):
            l.announcement(f"{index}. {list(model.keys())[0]}  ({list(model.values())[0]})")

        model_index = int(input(f"\nWhich model would you like to use? (1-{len(find_models())}) ")) - 1
        model=str(list(find_models()[model_index].keys())[0])

        if type_=="retrain":
            Learning().continious_training_start(tokenizer,model,iterations,sentences)

        if type_ == "addition":
            new_sentences_path = input("Enter the path to your new sentences: ")

            try:
                with open(new_sentences_path,"r") as f:
                    new_sentences = json.load(f)
            except FileNotFoundError:
                l.error(f"Could not find file {new_sentences_path}.")
            except json.JSONDecodeError:
                l.error("Failed to decode JSON. Please make sure your file contains a list of sentences.")

            Learning().add_training_to_model(tokenizer,model,new_sentences)

def main():     
    clean_logs()

    if len(sys.argv) >= 1:
        sys.argv.extend(["none","none","none","none"])

    if sys.argv[1] == "--easy-setup":
        ignore_list = None
        try:
            i = sys.argv.index("--ignore-from")
            ignore_list = sys.argv[i+1].split(",")
        except ValueError:
            l.debug("No ignore from defined") 
        except IndexError:
            l.error("--ignore-from invalid syntax! Correct usage: --ignore-from person_a,person_b,person_c")
        if ignore_list is not None:
            l.debug(f"Ignore list: {ignore_list}")
        assistant("--skip-extract" in sys.argv, ignore_list)

    if sys.argv[1] == "--check-gpu":
        if not tf.test.is_gpu_available():
            l.warn("No compatible GPU found.")
            quit(1)
        else:
            gpu_details = tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0])
            compute_capability = float(f"{gpu_details.get('compute_capability',0)[0]}.{gpu_details.get('compute_capability',0)[1]}")
            l.info(f"Found GPU {gpu_details.get('device_name','Unkown')}, which has compute capability {compute_capability} ...")
            quit(0)

    if sys.argv[1] == "--cont-training":
        add_training("retrain")

    if sys.argv[1] == "--add-training":
        add_training("addition")

    if sys.argv[1] == "--evaluate":
        with open(os.getenv("TOKENIZER_PATH"),"rb") as f:
            tokenizer = pickle.load(f)

        with open("messages.txt","r") as f:
            sentences = json.load(f)

        l.announcement("Available models: ")
        for index, model in enumerate(find_models(),1):
            l.announcement(f"{index}. {list(model.keys())[0]}  ({list(model.values())[0]})")

        model_index = int(input(f"\nWhich model would you like to use? (1-{len(find_models())}) ")) - 1
        model=str(list(find_models()[model_index].keys())[0])

        l.info(Tools.evaluate(model, tokenizer, sentences))
        quit(0)

    if sys.argv[1] == "--test-log":
        l.announcement("Announcement message")
        l.debug("Debug message")
        l.info("Info message")
        l.warn("Warning message")
        l.error("Error message")
        l.critical("Critical message")
        quit(0)


    if not os.getenv("BOT_TOKEN") and not "--local" in sys.argv:
        l.error("Discord bot token not defined in .env")
        token = input("No discord bot token provided. Go to https://discord.com/developers/applications to get your bot token.\n Input your bot token: ")
        if len(token) != 72:
            l.warn("Invalid token format! not applying changes")
            quit(1)
        set_key(dotenv_path=env_path,key_to_set="BOT_TOKEN", value_to_set=token)

    if not os.getenv("MODEL_PATH"):
        l.error("Model path not defined in .env")
        return

    with open(os.getenv("TOKENIZER_PATH"),"rb") as f:
        tokenizer = pickle.load(f)

    if not os.getenv("TOKENIZER_PATH"):
        l.error("Tokenizer path not defined in .env")
        return
    
    if not "--local" in sys.argv:
        l.info("Starting bot")
        bot.start(os.environ["BOT_TOKEN"])
    else:
        running = True
        l.info("Entering local mode... Stop with CTRL+C")
        generation:Generation = Generation(os.getenv("MODEL_PATH"))
        while running:
            l.announcement(generation.generate(input("Seed: "), int(input("Number of words: "))))
    
main()