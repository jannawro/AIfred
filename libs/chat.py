from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.globals import set_verbose
from libs.router import categorizer_chain


class Chat(object):
    def __init__(self):
        # set langchain globals
        set_llm_cache(InMemoryCache())
        set_verbose(True)

        llm = ChatOpenAI()
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human. Your responses should be polite but concise. Always answet in a valid Markdown format."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ],
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory
        )

        self.test_chain = categorizer_chain

    async def message(self, input: str):
        return await self.conversation.acall({"input": input})

    async def test(self, input: str):
        return await self.test_chain.ainvoke({"input": input})

