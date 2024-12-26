import discord
import os
from usage.generation import Generation
import psutil
import cpuinfo
import tensorflow as tf
import requests
import time
import platform
from time import strftime, localtime
from pathlib import Path
import logging
from ping3 import ping 
import pickle
import json
import random

training_in_progress:bool = False

intents = discord.Intents.default()
intents.message_content = True

logger = logging.getLogger("ma")

intents = intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)
generation:Generation = None
inbreeding_messages: list = []
message_amount:int = 0

TRAIN_AMOUNT=50

def save_inbreeding():
    global inbreeding_messages
    with open("inbreeding.txt","r") as f:
        messages:list = json.load(f)
    messages.extend(inbreeding_messages)
    with open("inbreeding.txt","w") as f:
        json.dump(messages,f)
    inbreeding_messages = []
    

def save_inbreeding_backup():
    with open("inbreeding.txt.bkp","w") as f:
        json.dump(inbreeding_messages,f)

@bot.event
async def on_ready():
    global inbreeding_messages
    global generation

    logger.info("Discord bot online")
    generation = Generation(os.getenv("MODEL_PATH")) 
    logger.info("Generation class initialized")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="The sound of fans spinning on the host server"))

@bot.event
async def on_message(message: discord.Message):
    global training_in_progress
    global inbreeding_messages
    global message_amount
    if training_in_progress:
        return

    if message.channel.id == 1321123050422538271 and message.author.id in [1318121553162010634, 1318198147524071504]:
        async with message.channel.typing():
            inbreeding_messages.append(message.content)
            ind = random.randint(0,len(message.content.split(" "))-3)
            generated_sentence = generation.generate(
                                                    " ".join(message.content.split(" ")[:ind][:ind+2]),
                                                    random.randint(4,16)
                                            )
        await message.channel.send(f"{generated_sentence}\n\n -# Progress to next retrain: {len(inbreeding_messages)}/{TRAIN_AMOUNT}")
            
        if len(inbreeding_messages) >= TRAIN_AMOUNT:
            training_in_progress = True
            await message.channel.send("Starting training...")
            __train()
            training_in_progress = False
            await message.channel.send("Training finished")
            inbreeding_messages.clear()
            
        

@bot.slash_command(
        description="Generates a sentence based off of my (ctih) messages",
        integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        }
    )       
async def talk(ctx:discord.ApplicationContext, seed:str, word_amount:int, timings:bool=False):
    if training_in_progress:
        ctx.respond("Training is in progress!")
        return
    start = time.time()
    await ctx.defer()
    generated_sentence = generation.generate(seed,word_amount)
    a = f'\nGeneration took {round(time.time() - start,3)} seconds' if timings else ''
    print(f"Responded to {ctx.author.name} with {generated_sentence}")
    await ctx.respond(generated_sentence + a)

@bot.slash_command(
        description="Gets information about the model",
        integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        }
    )
async def details(ctx):
    if training_in_progress:
        ctx.respond("Training is in progress!")
        return
    await ctx.defer()
    cpu = cpuinfo.get_cpu_info()
    embed = discord.Embed(
        title="Host / model information",
        description=f"Public IP: {requests.get('https://ipinfo.io/ip').text}, OS: { platform.system() if platform.system() != 'Darwin' else 'macOS' } {platform.release() if platform.system() != 'Darwin' else platform.mac_ver()[0]}",
    )

    model_date = None
    model_size = None

    if os.path.isdir(generation.model_path):
        model_size = sum(f.stat().st_size for f in Path(generation.model_path).glob('**/*') if f.is_file())
        latest_file = 0
        for file in os.listdir(generation.model_path):
            if platform.system() == "Windows":
                date_mod = os.path.getctime(os.path.join(generation.model_path,file))
            else:
                date_mod = os.stat(os.path.join(generation.model_path,file)).st_mtime
            latest_file = date_mod if latest_file < date_mod else latest_file
        model_date = latest_file

    else:
        model_size = os.path.getsize(generation.model_path)
        if platform.system() == "Windows":
            date_mod = os.path.getmtime(os.path.join(generation.model_path))
        else:
            date_mod = os.stat(os.path.join(generation.model_path)).st_mtime
        model_date = date_mod

    model_size = model_size / 1_000_000
        
    gpu_name = None
    
    model_device = "GPU" if tf.test.is_gpu_available() else "CPU"
    if model_device == "GPU": gpu_name = tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name","Unkown GPU")
    ram_percentage = psutil.Process(os.getpid()).memory_percent()
    embed.add_field(name="CPU Model", value=cpu["brand_raw"], inline=False)
    if gpu_name: embed.add_field(name="GPU Model", value=gpu_name)
    embed.add_field(name="RAM Used by bot", value=f"{round(psutil.Process(os.getpid()).memory_info().rss / (1024**2),1)}mb", inline=False)
    embed.add_field(name="Ping to frii.site headquarters (1,39€ server in Frankfurt)", value=f"{round(ping('vps.frii.site',timeout=3)*1000)}ms")
    embed.add_field(name="Model ran on", value=model_device, inline=False)
    embed.add_field(name="Model created on", value=strftime('%d.%m.%Y %H.%M', localtime(model_date)), inline=False)
    embed.add_field(name="Model size", value=f"{round(model_size,1)}mb", inline=False)
    embed.set_footer(text="Powered by TensorFlow")
    embed.set_thumbnail(url="https://i.ibb.co/syT024V/image.png")
    await ctx.respond(embed=embed)

@bot.slash_command(description="Syncs commands")
async def sync(ctx):
    if ctx.author.id != 642441889181728810:
        return
    await bot.sync_commands()
    await ctx.respond("Synced commands")
    return

inbreeding = discord.SlashCommandGroup("inbreeding","Inbreeding!")

@inbreeding.command()
async def conversate(ctx: discord.ApplicationContext):
    await ctx.defer()
    await ctx.send(generation.generate("I really like",random.randint(4,16)))

@inbreeding.command()
async def train(ctx:discord.ApplicationContext):
    if training_in_progress:
        ctx.respond("Training is in progress!")
        return
    if ctx.author.id not in [542701119948849163,642441889181728810]:
        ctx.respond("STOP ABUSING ME")
        return
    ctx.send("Brah wait a sec im loading tensorflwww")
    await ctx.defer()
    await ctx.send("Starting training. This will take around 3 minutes...")
    start = time.time()
    __train()
    await ctx.send(f"Trained model succesfully in {round(time.time() - start)}s")

def __train():
    global training_in_progress
    training_in_progress = True
    from learning.learning import Learning
    with open(os.getenv("TOKENIZER_PATH"),"rb") as f:
        tokenizer = pickle.load(f)

    model_path = r"C:\Users\nevalaonni\Desktop\MessageAi\model-inbred.h5"

    target_sentences = inbreeding_messages.copy()
    generation.free()
    Learning(128).add_training_to_model(tokenizer,model_path, target_sentences*3,3,"model-inbred.h5")
    generation.reinit()
    save_inbreeding()

@inbreeding.command()
async def reload(ctx:discord.ApplicationContext):
    if training_in_progress:
        ctx.respond("Training is in progress!")
        return
    global generation
    await ctx.defer()
    generation = Generation(os.getenv("MODEL_PATH")) 
    await ctx.respond("Reloaded model")

@inbreeding.command()
async def backup(ctx):
    save_inbreeding_backup()
    await ctx.respond("Backed up")    

bot.add_application_command(inbreeding)

try:
    def start(token:str):
        bot.run(token)
except Exception as e:
    print(e.__traceback__)
    save_inbreeding_backup()