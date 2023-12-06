import logging
from uuid import uuid4
from discord import Client, DMChannel, Intents, Message
from typing import Any, List
from pathlib import Path
from langchain.cache import InMemoryCache
from datetime import timedelta, datetime, timezone
from langchain.chat_models.openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.memory import (
    CombinedMemory,
    ConversationBufferWindowMemory,
    ConversationEntityMemory,
    ConversationSummaryMemory,
)
from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts.chat import PromptTemplate
from qdrant_client import QdrantClient
from chains.general import general_chain
from chains.memory import (
    MemoryCategoryList,
    input_to_memory_categories,
    input_to_memory_key,
    memory_synthesizer,
)


# TODO: add logging
class Chatbot(Client):
    def __init__(
        self,
        intents: Intents,
        qdrant_url: str,
        logger: logging.Logger,
        human_prefix="Janek",
        ai_prefix="Alfred",
        conversation_ttl=20,
        **options: Any
    ) -> None:
        super().__init__(intents=intents, **options)

        self.logger = logger

        self.message_channel = None
        self.last_chatbot_message = None

        self.logger.debug("Setting ")
        self.human_prefix = human_prefix

        # init llm cache
        self.llm_cache = InMemoryCache()
        set_llm_cache(self.llm_cache)

        # conversation ttl
        self.conversation_ttl = conversation_ttl

        # chatbot document store
        client = QdrantClient(url=qdrant_url)
        # TODO: embeddings cache
        self.doc_store = Qdrant(
            client=client, collection_name="documents", embeddings=OpenAIEmbeddings()
        )
        self.memory_schema = Path("./prompts/memory.schema").read_text()

        # init conversation memory
        self.recent_messages_memory = ConversationBufferWindowMemory(
            ai_prefix=ai_prefix,
            human_prefix=self.human_prefix,
            memory_key="recent_messages",
            k=4,
            input_key="user_input",
        )
        self.conversation_summary_memory = ConversationSummaryMemory(
            ai_prefix=ai_prefix,
            human_prefix=self.human_prefix,
            memory_key="conversation_summary",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            input_key="user_input",
        )
        self.subjects_memory = ConversationEntityMemory(
            ai_prefix=ai_prefix,
            human_prefix=self.human_prefix,
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            k=2,
            input_key="user_input",
            chat_history_key="subjects",
            entity_summarization_prompt=PromptTemplate.from_file(
                template_file="./prompts/modified_entity_summarization.py",
                input_variables=["entity", "summary", "history", "input"],
            ),
        )

        self.chatbot_memory = CombinedMemory(
            memories=[
                self.recent_messages_memory,
                self.conversation_summary_memory,
                self.subjects_memory,
            ]
        )

        # conversation chain
        self.conversation_chain = general_chain(memory=self.chatbot_memory)

    async def create_message_channel(self, message: Message) -> None:
        """create a direct message channel if one doesn't exist yet"""
        if isinstance(message.channel, DMChannel) and self.message_channel == None:
            self.message_channel = await message.author.create_dm()
        return

    async def handle_memory(self, date) -> None:
        # time between now and the last message
        since_last_message = timedelta()
        if self.last_chatbot_message:
            since_last_message = (
                datetime.now(timezone.utc) - self.last_chatbot_message.created_at
            )

        # flush conversation memory if last message is older than conversation TTL
        if since_last_message >= timedelta(minutes=self.conversation_ttl):
            _ = self.memory_transfer(self.subjects_memory.copy(deep=True), date)

            self.chatbot_memory.clear()
            self.llm_cache.clear()

        return

    async def memory_transfer(
        self, subjects_memory: ConversationEntityMemory, date: str
    ) -> None:
        documents = []
        subjects = subjects_memory.entity_store.dict()
        facts = []
        for _, v in subjects:
            facts.extend(v.split("."))
        for fact in facts:
            result = self.doc_store.similarity_search_with_score(
                query=fact,
                k=1,
            )

            score = result[0][1]
            if score < 0.9:
                memory_key = (
                    input_to_memory_key().invoke({"user_input": fact}).get("output")
                )

                documents.append(
                    {
                        "content": fact,
                        "metadata": {
                            "key": memory_key,
                            "last_updated": date,
                            "uuid": str(uuid4()),
                        },
                    }
                )
            else:
                matched_document = result[0][0]

                synthesized_memory = (
                    memory_synthesizer()
                    .invoke(
                        {
                            "old_memory": matched_document.page_content,
                            "new_memory": fact,
                        }
                    )
                    .get("output")
                )

                documents.append(
                    {
                        "content": synthesized_memory,
                        "metadata": {
                            "key": matched_document.metadata["key"],
                            "last_updated": date,
                            "uuid": matched_document.metadata["uuid"],
                        },
                    }
                )

        self.doc_store.add_texts(
            texts=[document["content"] for document in documents],
            metadatas=[document["metadata"] for document in documents],
            ids=[document["metadata"]["uuid"] for document in documents],
        )
        return

    def get_memory_categories(self, message: Message, date: str) -> MemoryCategoryList:
        response = (
            input_to_memory_categories()
            .invoke({"user_input": message.content, "date": date})
            .get("output", "")
        )

        return MemoryCategoryList.parse_raw(response)

    def mmr_find_documents(
        self, message: Message, memory_categories: MemoryCategoryList
    ) -> List[Document]:
        return self.doc_store.max_marginal_relevance_search(
            query=message.content,
            filter={"key": [category.key for category in memory_categories.categories]},
        )

    def get_ai_response(
        self, message: Message, date: str, relevant_documents: List[Document]
    ) -> str:
        return self.conversation_chain.invoke(
            {
                "user_input": message.content,
                "human_prefix": self.human_prefix,
                "date": date,
                "long_term_memory": "\n".join(
                    [document.page_content for document in relevant_documents]
                ),
            }
        ).get("output", "received empty response")

    async def on_message(self, message: Message) -> None:
        # save bot message, but don't respond to bot-generated messages
        if message.author == self.user:
            self.last_chatbot_message = message
            return

        _ = self.create_message_channel(message)

        date = datetime.now().strftime("%d/%m/%Y") + " (DD/MM/YYYY)"

        await self.handle_memory(date)

        memory_categories = self.get_memory_categories(message, date)

        relevant_documents = self.mmr_find_documents(message, memory_categories)

        response = self.get_ai_response(message, date, relevant_documents)

        await message.reply(content=response, mention_author=True)

        return
