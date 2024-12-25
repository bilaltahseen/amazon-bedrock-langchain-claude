import os
import boto3
import constants

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


def run_llm(query: str):

    # Initialize the Bedrock client with the runtime API
    bedrock = boto3.client(
        service_name="bedrock-runtime",  # Changed from "bedrock" to "bedrock-runtime"
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # Initialize the embedding model
    embeddings = BedrockEmbeddings(
        client=bedrock,
        model_id="amazon.titan-embed-text-v2:0",  # Specify the embedding model
    )

    docsearch = PineconeVectorStore(index_name=constants.INDEX_NAME, embedding=embeddings)

    chat = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm(query="What is a Hyperledger Fabric")
    print(res["answer"])
