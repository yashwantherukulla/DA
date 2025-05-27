from gpt_researcher import GPTResearcher

from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import asyncio
import json

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def setup_store():
    essay = """
    May 2004

    (This essay was originally published in Hackers & Painters.)

    If you wanted to get rich, how would you do it? I think your best bet would be to start or join a startup.
    That's been a reliable way to get rich for hundreds of years. The word "startup" dates from the 1960s,
    but what happens in one is very similar to the venture-backed trading voyages of the Middle Ages.

    Startups usually involve technology, so much so that the phrase "high-tech startup" is almost redundant.
    A startup is a small company that takes on a hard technical problem.

    Lots of people get rich knowing nothing more than that. You don't have to know physics to be a good pitcher.
    But I think it could give you an edge to understand the underlying principles. Why do startups have to be small?
    Will a startup inevitably stop being a startup as it grows larger?
    And why do they so often work on developing new technology? Why are there so many startups selling new drugs or computer software,
    and none selling corn oil or laundry detergent?


    The Proposition

    Economically, you can think of a startup as a way to compress your whole working life into a few years.
    Instead of working at a low intensity for forty years, you work as hard as you possibly can for four.
    This pays especially well in technology, where you earn a premium for working fast.

    Here is a brief sketch of the economic proposition. If you're a good hacker in your mid twenties,
    you can get a job paying about $80,000 per year. So on average such a hacker must be able to do at
    least $80,000 worth of work per year for the company just to break even. You could probably work twice
    as many hours as a corporate employee, and if you focus you can probably get three times as much done in an hour.[1]
    You should get another multiple of two, at least, by eliminating the drag of the pointy-haired middle manager who
    would be your boss in a big company. Then there is one more multiple: how much smarter are you than your job
    description expects you to be? Suppose another multiple of three. Combine all these multipliers,
    and I'm claiming you could be 36 times more productive than you're expected to be in a random corporate job.[2]
    If a fairly good hacker is worth $80,000 a year at a big company, then a smart hacker working very hard without 
    any corporate bullshit to slow him down should be able to do work worth about $3 million a year.
    ...
    ...
    ...
    """

    document = [Document(page_content=essay)]
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    # vector_store = FAISS.from_documents(document, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    client = QdrantClient(path="./tmp/langchain_qdrant")

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



async def main():
    query = """
        Summarize the essay into 3 or 4 succinct sections.
        Make sure to include key points regarding wealth creation.

        Include some recommendations for entrepreneurs in the conclusion.
    """

    vector_store = setup_store()
    researcher = GPTResearcher(
        query=query,
        report_type="research_report",
        report_source="langchain_vectorstore",
        vector_store=vector_store,
        # config_path="./configs/hybrid-config.json",
    )
    # print(json.dumps(researcher.cfg.__dict__, indent=2))

    await researcher.conduct_research()
    report = await researcher.write_report()
    print(report)
    print(researcher.get_costs())

if __name__ == "__main__":
    asyncio.run(main())
