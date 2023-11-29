from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseMemory


def general_chain(memory: BaseMemory) -> LLMChain:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_general_chain.yaml",
                input_variables=["date"],
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    return LLMChain(
        llm=ChatOpenAI(model="gpt-4"),
        prompt=prompt,
        memory=memory,
        output_key="output",
    )
