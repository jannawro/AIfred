from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain.schema import BaseMemory


fake_general_chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = PromptTemplate.from_template(
        "Say: 'User input: {user_input}\n'General chain isn't implemented yet.'"
    ),
    output_key = "output"
)

def general_chain(memory: BaseMemory, memory_key: str) -> LLMChain:
    return LLMChain(
        llm = ChatOpenAI(model="gpt-4"),
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                SystemMessagePromptTemplate.from_template_file(template_file="./prompts/aifred.yaml", input_variables=[]),
                MessagesPlaceholder(variable_name=memory_key),
                HumanMessagePromptTemplate.from_template("{user_input}")
            ]
        ),
        memory = memory
    )
