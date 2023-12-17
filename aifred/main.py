import logging
from os import getenv
from logging import getLevelName

from langchain_core.globals import set_debug, set_verbose

from core.intents import default_intents
from core.chatbot import Chatbot


def main():
    set_debug(True)
    set_verbose(True)

    chatbot = Chatbot(
        intents=default_intents(),
        qdrant_url=getenv("QDRANT_URL", "localhost"),
        logger=logging.getLogger("discord"),
    )

    chatbot.run(
        token=getenv("DISCORD_TOKEN", ""),
        log_level=getLevelName(getenv("LOGLEVEL", "INFO").upper()),
    )


if __name__ == "__main__":
    main()
