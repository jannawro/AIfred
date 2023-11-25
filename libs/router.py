from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.branch import RunnableBranch

categorizer_prompt = load_prompt("./prompts/catogorizer.yaml")
categorizer_chain = (
    PromptTemplate.from_template(categorizer_prompt.format(input="{input}"))
    | ChatOpenAI(cache=True)
    | StrOutputParser()
)

prompt_router = RunnableBranch(
    (lambda x: "action" in x["category"].lower(), action_chain),
    (lambda x: "query" in x["category"].lower(), query_chain),
    general_chain,
)
