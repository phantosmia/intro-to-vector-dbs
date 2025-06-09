import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting data...")
    loader = TextLoader("/home/raquel-bubuntu/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()
    print("splitting text...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    print("ingesting into Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME"),
    )
    print("ingestion complete.")