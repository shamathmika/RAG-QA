from typing import TypedDict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from langgraph.graph import StateGraph, START, END
from api.cache import RedisSemanticCache

load_dotenv()
cache = RedisSemanticCache()

# ---------------------------- State ----------------------------
class SearchState(TypedDict):
    query: str
    embedding: list[float]
    results: list[dict]
    reranked_results: list[dict]
    final_answer: str
    cache_hit: bool
    cached_answer: str

# ---------------------------- Nodes ----------------------------
def embed_query(state: SearchState) -> dict:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    return {"embedding": embedding.embed_query(state["query"])}

def vector_search(state: SearchState) -> dict:
    client = MongoClient(os.getenv("MONGODB_URI"))
    collection = client["rag_db"]["documents"]
    results = collection.aggregate([ # Vector search - finds the most similar chunks to the query
        {
            "$vectorSearch": { 
                "index": "vector_index", # Name of the vector index in MongoDB Atlas
                "path": "embedding", # Field containing the vector embeddings
                "queryVector": state["embedding"], # Vector embedding of the query
                "numCandidates": 20, # Number of candidate chunks to consider in HNSW
                "limit": 5 # Number of chunks to return
            }
        },
        {
            "$project": { # Projects the desired fields
                "text": 1, # Text of the chunk
                "file_name": 1, # Name of the file
                "page_number": 1, # Page number of the chunk
                "score": { "$meta": "vectorSearchScore" } # Score of the chunk
            }
        }
    ])
    return {"results": list(results)}
    
def rerank(state: SearchState) -> dict:
    query_words = state["query"].lower().split()
    for result in state["results"]:
        text_words = result["text"].lower().split()
        match_count = sum(1 for word in query_words if word in text_words)
        keyword_score = match_count / len(query_words) if query_words else 0
        result["reranked_score"] = result["score"] * 0.7 + keyword_score * 0.3
    return {"reranked_results": sorted(state["results"], key=lambda x: x["reranked_score"], reverse=True)[:3]}

def call_llm(state: SearchState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    context = "\n\n---\n\n".join(
        [chunk["text"] for chunk in state["reranked_results"]]
    )
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the following question based on the context provided. If the answer is not in the context, say so. Do not use any external knowledge."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
    ])
    return {"final_answer": response.content}

# ---------------------------- Cache Nodes ----------------------------
def check_cache(state: SearchState) -> dict:
    cached_answer = cache.get(state["embedding"])
    if cached_answer:
        return {"cache_hit": True, "cached_answer": cached_answer, "final_answer": cached_answer}
    return {"cache_hit": False, "cached_answer": ""}

def update_cache(state: SearchState) -> dict:
    cache.set(state["query"], state["final_answer"], state["embedding"])
    return {}

def route_after_cache(state: SearchState) -> str: # Conditional edge. If cache hit, end. Else, continue.
    if state["cache_hit"]:
        return "__end__" # equals END
    return "vector_search"

# ---------------------------- Graph ----------------------------
graph = StateGraph(SearchState)
graph.add_node("embed_query", embed_query)
graph.add_node("vector_search", vector_search)
graph.add_node("rerank", rerank)
graph.add_node("call_llm", call_llm)
graph.add_node("check_cache", check_cache)
graph.add_node("update_cache", update_cache)

graph.add_edge(START, "embed_query")
graph.add_edge("embed_query", "check_cache")
graph.add_conditional_edges("check_cache", route_after_cache)
graph.add_edge("vector_search", "rerank")
graph.add_edge("rerank", "call_llm")
graph.add_edge("call_llm", "update_cache")
graph.add_edge("update_cache", END)
search_graph = graph.compile()