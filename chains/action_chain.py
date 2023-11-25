from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


fake_action_chain = LLMChain(
    llm = ChatOpenAI(),
    prompt = PromptTemplate.from_template(
        "Say: 'User input: {user_input}\nAction chain isn't implemented yet.'"
    ),
    output_key = "output"
)
