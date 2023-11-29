from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def memory_categorizer_chain() -> LLMChain:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./prompts/sys_memory_categorizer.yaml",
                input_variables=["date", "memory_schema"],
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    return LLMChain(
        llm=ChatOpenAI(model="gpt-4", temperature=0.05, max_tokens=256), prompt=prompt
    )
