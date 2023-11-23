from dotenv import load_dotenv, find_dotenv
import discord
import os

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello')

@client.event
async def on_reaction_add(reaction, user):
    if reaction.discord.emoji.name == "heart":
        await reaction.message.add_reaction(discord.Emoji(name=""))

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    client.run(
        token=os.getenv("DISCORD_TOKEN", ""),
    )
