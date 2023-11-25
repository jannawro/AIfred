from os import getenv
from logging import getLevelName

from dotenv import find_dotenv, load_dotenv

from libs.bot import Bot
from libs.intents import default_intents
from libs.chat import Chat


def main():
    load_dotenv(find_dotenv())

    chat = Chat()

    bot = Bot(
        intents=default_intents(),
        chat=chat
    )

    bot.run(
        token=getenv("DISCORD_TOKEN", ""),
        log_level=getLevelName(getenv("LOGLEVEL", "INFO").upper())
    )


if __name__ == "__main__":
    main()
