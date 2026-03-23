from api.graphs.upload_graph import upload_graph

with open("test.pdf", "rb") as f:
    pdf_bytes = f.read()

result = upload_graph.invoke({
    "file_name": "test.pdf",
    "file_bytes": pdf_bytes,
})

print(f"Uploaded {result['doc_count']} chunks to MongoDB")
