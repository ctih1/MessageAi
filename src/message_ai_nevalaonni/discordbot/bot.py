import discord
import os
from usage.generation import Generation

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot()

generation:Generation = None

@bot.event
async def on_ready():
    global generation
    print("Bot logged in!")
    generation = Generation("nevalaonni.h5")

@bot.slash_command(description="Test")
async def talk(ctx:discord.ApplicationContext, seed:str):
    await ctx.send(generation.generate(seed))

@bot.slash_command(description="Syncs commands")
async def sync(ctx):
    if ctx.author.id != 642441889181728810:
        return
    await ctx.bot.sync(guild=ctx.guild)
    return

def start(token:str):
    bot.run(token)