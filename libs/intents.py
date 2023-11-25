import discord

def default_intents() -> discord.Intents:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.reactions = True

    return intents
    
