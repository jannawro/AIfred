from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnableLambda
from chains.action_chain import fake_action_chain
from chains.query_chain import fake_query_chain
from chains.general_chain import general_chain


def categorizer_chain() -> LLMChain:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template_file(
                template_file="./sys_query_action_categorizer.yaml", input_variables=[]
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    return LLMChain(
        llm=ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=1,
        ),
        prompt=prompt,
    )


def category_router(x):
    if "action" in x["category"].lower():
        return fake_action_chain
    elif "query" in x["category"].lower():
        return fake_query_chain
    else:
        return general_chain


prompt_router_chain = {
    "category": categorizer_chain(),
    "user_input": lambda x: x["user_input"],
} | RunnableLambda(category_router)
