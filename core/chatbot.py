from discord import Client, DMChannel, Intents, Message
from typing import Any
from langchain.cache import InMemoryCache
from datetime import timedelta, datetime, timezone
from langchain.globals import set_llm_cache
from langchain.memory import CombinedMemory, ConversationBufferMemory, ConversationBufferWindowMemory, ConversationEntityMemory, VectorStoreRetrieverMemory
from chains.general_chain import general_chain

class Chatbot(Client):
    def __init__(self, intents: Intents, conversation_ttl=20, **options: Any) -> None:
        super().__init__(intents=intents, **options)

        self.message_channel = None
        self.last_chatbot_message = None

        # init llm cache
        self.llm_cache = InMemoryCache()
        set_llm_cache(self.llm_cache)

        # conversation ttl
        self.conversation_ttl = conversation_ttl

        # init conversation memory
        recent_messages = ConversationBufferMemory() # TODO: use ConversationBufferWindowMemory instead
        conversation_summary_memory = ConversationBufferMemory() # TODO: use ConversationSummaryMemory instead
        entities_memory = ConversationBufferMemory() # TODO: use ConversationEntityMemory instead
        long_term_memory = ConversationBufferMemory() # TODO: use VectorStoreRetrieverMemory instead
        self.chatbot_memory = CombinedMemory(memories=[
            recent_messages,
            conversation_summary_memory,
            entities_memory,
            long_term_memory,
        ])

        # conversation chain
        self.conversation_chain = general_chain(memory=self.chatbot_memory, memory_key="memory")


    async def on_message(self, message: Message) -> None:
        # don't respond to your own messages
        if message.author == self.user:
            self.last_chatbot_message = message
            return

        # create a direct message channel if one doesn't exist yet
        if isinstance(message.channel, DMChannel) and self.message_channel == None:
            self.message_channel = await message.author.create_dm()

        # time between now and the last message
        since_last_message = timedelta() 
        if self.last_chatbot_message:
            since_last_message = datetime.now(timezone.utc) - self.last_chatbot_message.created_at

        # flush conversation memory if last message is older than conversation TTL
        if since_last_message >= timedelta(minutes=self.conversation_ttl):
            # TODO: handle transitioning old conversation memory into long-term memory. Should be async.
            self.chatbot_memory.clear()
            self.llm_cache.clear()

        response = await self.conversation_chain.ainvoke({"user_input": message.content})
        await message.reply(content=response.get("text"), mention_author=True)

