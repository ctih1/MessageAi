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

intents = discord.Intents.default()
intents.message_content = True

logger = logging.getLogger("ma")

bot = discord.Bot()

generation:Generation = None

@bot.event
async def on_ready():
    global generation
    logger.info("Discord bot online")
    generation = Generation(os.getenv("MODEL_PATH")) 
    logger.info("Generation class initialized")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="The sound of fans spinning on the host server"))

@bot.slash_command(
        description="Generates a sentence based off of my (ctih) messages",
        integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        }
    )       
async def talk(ctx:discord.ApplicationContext, seed:str, word_amount:int, timings:bool=False):

    start = time.time()
    await ctx.defer()
    generated_sentence = generation.generate(seed,word_amount)
    a = f'\nGeneration took {round(time.time() - start,3)} seconds' if timings else ''
    await ctx.respond(generated_sentence + a)

@bot.slash_command(
        description="Gets information about the model",
        integration_types={
            discord.IntegrationType.guild_install,
            discord.IntegrationType.user_install,
        }
    )
async def details(ctx):
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
                date_mod = os.stat(os.path.join(generation.model_path,file)).st_ctime
            latest_file = date_mod if latest_file < date_mod else latest_file
        model_date = latest_file

    else:
        model_size = os.path.getsize(generation.model_path)
        if platform.system() == "Windows":
            date_mod = os.path.getctime(os.path.join(generation.model_path))
        else:
            date_mod = os.stat(os.path.join(generation.model_path)).st_ctime
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


def start(token:str):
    bot.run(token)