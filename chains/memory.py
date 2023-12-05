from typing import List
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel


class MemoryCategory(BaseModel):
    key: str
    query: str


class MemoryCategoryList(BaseModel):
    categories: List[MemoryCategory]


def input_to_memory_categories() -> LLMChain:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_input_to_memory_categories.yaml",
                input_variables=["date", "memory_schema"],
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    return LLMChain(
        llm=ChatOpenAI(model="gpt-4", temperature=0.05, max_tokens=256), prompt=prompt, output_key="output"
    )

def input_to_memory_key() -> LLMChain:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_input_to_memory_key.yaml",
                input_variables=["date", "memory_schema"],
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    return LLMChain(
        llm=ChatOpenAI(model="gpt-4", temperature=0.05, max_tokens=15), prompt=prompt,output_key="output"
    )
    
