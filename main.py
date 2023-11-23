import os

import discord
from dotenv import find_dotenv, load_dotenv

from libs.bot.bot import Bot


def main():
    load_dotenv(find_dotenv())

    intents = discord.Intents.default()
    intents.message_content = True
    intents.reactions = True

    bot = Bot(
        intents=intents
    )

    bot.run(
        token=os.getenv("DISCORD_TOKEN", ""),
        log_level=10 # DEBUG
    )


if __name__ == "__main__":
    main()
