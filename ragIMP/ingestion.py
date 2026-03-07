
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()



if __name__ == "__main__":
    print("Ingesting....")
    print(os.environ['PINECONE_API_KEY'])
    loader = TextLoader("D:\\Genai\\GenAI-Langchain_RAG\\ragIMP\\mediumblog1.txt"
                        , encoding="utf-8")
    document = loader.load()

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("Embedding....")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-2.5-flash", GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY"))

    print("ingesting to pinecone....")
    PineconeVectorStore.from_documents( texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("done")






