from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Product_management")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 50)
chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(chunks, embeddings)

db.save_local("faiss_pm_index")

query = "What is a white hole in general relativity"

docs_n_scores = db.similarity_search_with_score(query, k=2)

for doc, score in docs_n_scores:
    print("Match Found (Distance Score: {:.4f}) ---".format(score))
    print(doc.page_content)
