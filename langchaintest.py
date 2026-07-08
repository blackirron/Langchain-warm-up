from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a coder"),
    ("user", "{ps}")
])

llm=ChatOllama(
    model="qwen2.5-coder:1.5b",
    temperature=0.2,
    num_ctx=2048#limits no. of tokens
)

parser = StrOutputParser()

chain = prompt|llm|parser

response = chain.invoke({"ps":"how to kill a mockingbird"})

print(response)
