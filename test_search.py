from api.graphs.search_graph import search_graph

result = search_graph.invoke({
    "query": input("Ask a question: ")
})

print(f"\nAnswer: {result['final_answer']}")
