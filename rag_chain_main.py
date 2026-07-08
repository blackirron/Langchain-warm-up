from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

llm = ChatOllama(model="qwen2.5-coder:1.5b",temperature =0.1,)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local("faiss_pm_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k":5})

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_system_prompt = (
    "You are an AI assistant. Answer the user's question using ONLY "
    "the provided context. If you don't know the answer, just say so.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, qa_chain
)

store: dict[str, BaseChatMessageHistory]={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history,
    input_messages_key = "input",
    history_messages_key = "chat_history",
    output_messages_key = "answer"
)

while True:
    user_input = input("\nYou: ")
    if user_input == "exit":
        break

    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable":{"session_id":"session1"}}
    )
    print(f"AI: {response['answer']}")
