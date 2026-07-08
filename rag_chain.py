from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
llm = ChatOllama(
    model="qwen2.5-coder:1.5b",
    temperature =0.1,
    num_ctx=2048
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local("faiss_pm_index", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k":25})

system_prompt = (
    "You are an advanced AI assistant. Answer the user's question using ONLY the provided context below. If you do not know the answer based on the context, say 'I cannot find that information in the document.'\n\n Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

query = "What is my name"

response = rag_chain.invoke({"input":query})

print(response["answer"])
