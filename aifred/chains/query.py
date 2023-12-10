from langchain.chat_models import FakeListChatModel
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

fake_query_chain = (
    PromptTemplate.from_template("Stub")
    | FakeListChatModel(responses=["Query chain isn't implemented yet."])
    | StrOutputParser()
)
