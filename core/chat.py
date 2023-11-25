from langchain.cache import InMemoryCache
from langchain.globals import (
    set_verbose,
    set_llm_cache,
    set_debug
)
from langchain.globals import set_verbose
from langchain.schema.runnable import RunnableSerializable


class Chat(object):
    def __init__(self, main_chain: RunnableSerializable):
        # set langchain globals
        set_llm_cache(InMemoryCache())
        set_debug(True)
        set_verbose(True)

        self.main_chain = main_chain

    async def message(self, user_input: str):
        return self.main_chain.invoke({"user_input": user_input}).get('output')


