from typing import TypedDict # For defining state of the langgraph
import fitz # For extracting text from pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text in correct chunks - word is not broken in a chunk, separates at \n, allows for overlap so context is not lost
from langchain_openai import OpenAIEmbeddings # For embedding the text
from pymongo import MongoClient # For connecting to MongoDB
from dotenv import load_dotenv # For loading environment variables
import os # For accessing environment variables
from langgraph.graph import StateGraph, START, END # For creating the langgraph

load_dotenv()

# ---------------------------- State ----------------------------
class UploadState(TypedDict): # State of the langgraph - all the items that are passed from one node to another
    file_name: str # Name of the file
    file_bytes: bytes # Bytes of the file
    extracted_text: list[dict] # Extracted text from the file in the format [{"text": "text", "page_number": page_number}]
    chunks: list[dict] # Chunks of the extracted text in the format [{"text": "text", "page_number": page_number}]
    doc_count: int # Number of documents

# ---------------------------- Nodes ----------------------------
def extract_text(state: UploadState) -> dict:
    """Extract text from a PDF file page by page using PyMuPDF.

    Args:
        state: UploadState with 'file_bytes' (bytes) - raw PDF binary data.

    Returns:
        dict with 'extracted_text': list[dict] - each dict has 'text' (str) and 'page_number' (int).
    """
    doc = fitz.open(stream=state["file_bytes"], filetype="pdf")
    text = []
    page_number = 1
    for page in doc:
        text.append({"text": page.get_text(), "page_number": page_number})
        page_number += 1
    return {"extracted_text": text}

def chunk_text(state: UploadState) -> dict:
    """Split extracted text into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        state: UploadState with 'extracted_text' (list[dict]) - page-level text with page numbers.

    Returns:
        dict with 'chunks': list[dict] - each dict has 'text' (str) and 'page_number' (int).
        Chunks are ~1000 chars with 200 char overlap to preserve context across boundaries.
    """
    text = state["extracted_text"]
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Size of each chunk
        chunk_overlap=200, # Overlap of 200 characters between chunks so context is not lost
    )
    for page in text: 
        chunks.extend([{"text": chunk, "page_number": page["page_number"]} for chunk in splitter.split_text(page["text"])])
    return {"chunks": chunks}

def embed_and_store(state: UploadState) -> dict:
    """Generate embeddings for all chunks and store them in MongoDB Atlas.

    Args:
        state: UploadState with 'chunks' (list[dict]) and 'file_name' (str).

    Returns:
        dict with 'doc_count' (int) - number of chunks embedded and stored.
        Each MongoDB document contains: text, page_number, embedding (1536D), file_name.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")) 
    texts = [chunk["text"] for chunk in state["chunks"]]
    vectors = embeddings.embed_documents(texts) # Returns a list of vectors - each vector is a list of 1536 floats
    
    client = MongoClient(os.getenv("MONGODB_URI"))
    collection = client["rag_db"]["documents"]
    
    collection.insert_many( [
        {
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "embedding": vector,
            "file_name": state["file_name"]
        }
        for chunk, vector in zip(state["chunks"], vectors)
    ])
    return {"doc_count": len(texts)}

# ---------------------------- Graph ----------------------------
graph = StateGraph(UploadState)
graph.add_node("extract_text", extract_text)
graph.add_node("chunk_text", chunk_text)
graph.add_node("embed_and_store", embed_and_store)

graph.add_edge(START, "extract_text")
graph.add_edge("extract_text", "chunk_text")
graph.add_edge("chunk_text", "embed_and_store")
graph.add_edge("embed_and_store", END)

upload_graph = graph.compile()