from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path=Path(__file__).parent/"Learning_Python.pdf"

loader=PyPDFLoader(file_path=pdf_path)
# //read pdf file-> page by page load ho gya
docs=loader.load()

print("Docs :", docs[10])


# chunking krna hai
# docs ko text me 
# langchain hi krega chunking

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
split_docs=text_splitter.split_documents(documents=docs)


# vector embeddings banana hai

embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-large",
    
)

# using {embedding model} create embeddings of {split_docs} and store in DB
# (docker install krna h)--> quadrant db ko docker k andar chalana h

vector_store= QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model  
)

print("Indexing of document done!")





