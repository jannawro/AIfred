from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    load_prompt,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from chains.action_chain import fake_action_chain
from chains.query_chain import fake_query_chain
from chains.general_chain import general_chain

categorizer_prompt = load_prompt("./prompts/categorizer.yaml")
# categorizer_chain = (
#     PromptTemplate.from_template(categorizer_prompt.format(user_input="{user_input}"))
#     | ChatOpenAI(
#         model="text-davinci-003",
#         cache=True,
#         temperature=0.0,
#         verbose=True,
#     )
#     | StrOutputParser()
# )
categorizer_chain = LLMChain(
    llm=OpenAI(
        model="gpt-3.5-turbo",
        cache=True,
        temprature=0.0,
        max_tokens=1,
    ),
    prompt=PromptTemplate.from_template(categorizer_prompt.format(user_input="{user_input}"))
)

def category_router(x):
    if "action" in x["category"].lower():
        return fake_action_chain
    elif "query" in x["category"].lower():
        return fake_query_chain
    else:
        return general_chain

prompt_router_chain = {"category": categorizer_chain, "user_input": lambda x: x["user_input"]} | RunnableLambda(category_router)

