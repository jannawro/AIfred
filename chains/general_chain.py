from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate


fake_general_chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = PromptTemplate.from_template(
        "Say: 'User input: {user_input}\n'General chain isn't implemented yet.'"
    ),
    output_key = "output"
)

general_chain = LLMChain(
    llm = ChatOpenAI(model="gpt-4"),
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template_file(template_file="./prompts/aifred.yaml", input_variables=[]),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    ),
    output_key = "output"
)
