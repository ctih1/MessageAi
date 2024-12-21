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
    generation = Generation("/Users/nevalaonni/MessageAi/nevalaonni.h5")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="The sound of fans spinning on the host server"))

@bot.slash_command(description="Test")
async def talk(ctx:discord.ApplicationContext, seed:str, max_word_amount:int):
    await ctx.defer()
    await ctx.respond(generation.generate(seed,max_word_amount))

@bot.slash_command(description="Syncs commands")
async def sync(ctx):
    if ctx.author.id != 642441889181728810:
        return
    await ctx.bot.sync(guild=ctx.guild)
    return

def start(token:str):
    bot.run(token)