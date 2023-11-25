from os import getenv
from logging import getLevelName

from core.bot import Bot
from core.intents import default_intents
from core.chat import Chat
from chains.router_chain import prompt_router_chain


def main():
    main_chain = prompt_router_chain

    chat = Chat(main_chain=main_chain)

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
