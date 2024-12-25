import os
import boto3
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_aws import BedrockEmbeddings,ChatBedrock
from langchain_pinecone import PineconeVectorStore

# Initialize the Bedrock client with the runtime API
bedrock = boto3.client(
    service_name="bedrock-runtime",  # Changed from "bedrock" to "bedrock-runtime"
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v2:0"  # Specify the embedding model
)

def ingest_docs():
    loader = ReadTheDocsLoader(
        "langchain-docs/hyperledger-fabric.readthedocs.io/en/latest", 
        encoding="utf-8"
    )
    
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)
    
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, 
        embeddings, 
        index_name="langchain-doc-index"
    )
    
    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()