from langchain.prompts import PromptTemplate
from langchain.chat_models.fake import FakeListChatModel
from langchain.schema import StrOutputParser


fake_action_chain = (
    PromptTemplate.from_template("Stub")
    | FakeListChatModel(responses=["Action chain isn't implemented yet."])
    | StrOutputParser()
)
