from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

url="https://en.wikipedia.org/wiki/White_hole"
loader = WebBaseLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap= 20
)

chunks = text_splitter.split_documents(docs)

print("no of chunks: ", len(chunks))
