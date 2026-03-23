from rest_framework.views import APIView
from rest_framework.response import Response
from api.graphs.search_graph import search_graph
from api.graphs.upload_graph import upload_graph


class SearchView(APIView):
    """API endpoint for asking questions about uploaded documents."""

    def post(self, request):
        """Process a search query through the RAG pipeline.

        Args:
            request: POST with JSON body {"query": str}.

        Returns:
            Response with {"answer": str, "cache_hit": bool}.
        """
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query is required"}, status=400)
        result = search_graph.invoke({"query": query})
        return Response({
            "answer": result["final_answer"],
            "cache_hit": result.get("cache_hit", False)
        })


class UploadView(APIView):
    """API endpoint for uploading and processing PDF documents."""

    def post(self, request):
        """Upload a PDF, extract text, chunk it, generate embeddings, and store in MongoDB.

        Args:
            request: POST with multipart form data containing 'file' (PDF).

        Returns:
            Response with {"message": str} on success.
        """
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "File is required"}, status=400)
        result = upload_graph.invoke({"file_bytes": file.read(), "file_name": file.name})
        return Response({"message": result["message"]})
