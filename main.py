from os import getenv
from logging import getLevelNamesMapping

from dotenv import find_dotenv, load_dotenv

from libs.bot.bot import Bot
from libs.bot.intents import default_intents


def main():
    load_dotenv(find_dotenv())

    bot = Bot(
        intents=default_intents()
    )

    bot.run(
        token=getenv("DISCORD_TOKEN", ""),
        log_level=getLevelNamesMapping()[getenv("LOGLEVEL", "INFO").upper()]
    )


if __name__ == "__main__":
    main()
