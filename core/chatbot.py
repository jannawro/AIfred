from discord import Client, DMChannel, Intents, Message
from typing import Any
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from datetime import timedelta, datetime, timezone
from langchain.globals import set_llm_cache
from langchain.memory import ConversationBufferMemory
from chains.general_chain import general_chain

# TODO: combine the chat and bot classes into a single chatbot class 
# TODO: set chat memory expiration to 15-20 minutes

class Chatbot(Client):
    # def __init__(self, conversation_chain: LLMChain, conversation_ttl=20, *, intents: Intents, **options: Any) -> None:
    def __init__(self, conversation_ttl=20, *, intents: Intents, **options: Any) -> None:
        super().__init__(intents=intents, **options)

        self.message_channel = None
        self.last_chatbot_message = None

        # self.conversation_chain = conversation_chain
        self.conversation_ttl = conversation_ttl
        self.conversation_memory = None
        self.conversation_memory_key = "conversation_history"
        set_llm_cache(InMemoryCache())


    async def on_message(self, message: Message) -> None:

        # don't respond to your own messages
        if message.author == self.user:
            self.last_chatbot_message = message
            return

        # create a direct message channel if one doesn't exist yet
        if isinstance(message.channel, DMChannel) and self.message_channel == None:
            self.message_channel = await message.author.create_dm()

        # time between now and the last message
        since_last_message = timedelta(minutes=self.conversation_ttl * 2)
        if self.last_chatbot_message:
            since_last_message = datetime.now(timezone.utc) - self.last_chatbot_message.created_at

        # create new conversation memory if none exists or last message is older than conversation TTL
        if self.conversation_memory == None or since_last_message >= timedelta(minutes=self.conversation_ttl):
            # TODO: handle transitioning old conversation memory into long-term memory. Should be async.
            self.conversation_memory = ConversationBufferMemory(
                memory_key=self.conversation_memory_key,
                return_messages=True
            )

        response = general_chain(memory=self.conversation_memory, memory_key=self.conversation_memory_key).invoke({"user_input": message.content}).get("text")
        await message.reply(content=response, mention_author=True)
