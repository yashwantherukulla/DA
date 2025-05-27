from gpt_researcher import GPTResearcher
import asyncio
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import Tuple, List, Dict, Any, Union

async def get_web_report(query: str, report_type: str = "research_report"):
    researcher = GPTResearcher(query, report_type, config_path='./configs/web-config.json')
    # print(json.dumps(researcher.cfg.__dict__, indent=2))
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report

async def get_local_report(query: str, report_type: str = "research_report"):
    # supports PDF, plain text, CSV, Excel, Markdown, PowerPoint, and Word documents. --> from official docs
    researcher = GPTResearcher(query=query,
                               report_type=report_type,
                               report_source="local",
                               config_path="./configs/local-config.json"
                            )
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report

async def get_hybrid_report(query: str, report_type: str = "research_report"):
    vector_store = setup_store()
    researcher = GPTResearcher(
        query=query,
        report_type="research_report",
        vector_store=vector_store,
        report_source="langchain_vectorstore",
        config_path="./configs/hybrid-config.json",
    )
    # print(json.dumps(researcher.cfg.__dict__, indent=2))

    await researcher.conduct_research()
    report = await researcher.write_report()
    print(report)
    print(researcher.get_costs())
    
    return report

def setup_store(client_path: str = "./storage/langchain_qdrant", essay_file_path: str = "./storage/docs/Nvidia-stock.md"):
    try:
        with open(essay_file_path, 'r', encoding='utf-8') as file:
            essay = file.read()
        print(f"Successfully read essay from {essay_file_path}")
    except FileNotFoundError:
        print(f"Essay file not found at {essay_file_path}. Using a placeholder text.")
        essay = "This is a placeholder text because the essay file wasn't found."
    except Exception as e:
        print(f"Error reading essay file: {str(e)}. Using a placeholder text.")
        essay = "This is a placeholder text due to error reading the essay file."

    document = [Document(page_content=essay)]
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    # vector_store = FAISS.from_documents(document, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    client = QdrantClient(path=client_path)

    try:
        client.create_collection(
            collection_name="demo_collection",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        collection_created = True
    except ValueError:
        print("Using existing collection...")
        collection_created = False
        
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    )

    if collection_created:
        vector_store.add_documents(documents=docs)
        print("Documents added to the collection.")
    
    return vector_store

if __name__ == "__main__":
    # query = "Should I invest in Nvidia?"
    query = "Summarize the essay into 3 or 4 succinct sections. also provide some tips for investors on this company."
    report = asyncio.run(get_local_report(query))
    
    print("Report:")
    print(report)