from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.schema import Document
from collections import Counter

# # Qdrant configuration
QDRANT_URL = "https://a26f47f6-04bd-4433-bc1d-30c41f48f0cd.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ZHDktyBPFyC7qCHT-zmPwr8URuKZ3_PwB-D4xuHCwj0"
QDRANT_COLLECTION = "pdfs_embeddings_sk"

Document(
    page_content="This is a chunk of text from a PDF.",
    metadata={"source": "employee_handbook.pdf"}
)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def create_embeddings_and_store_qdrant(docs):
#     Qdrant.from_documents(
#         documents=docs,
#         embedding=embedding_model,  # ✅ USE embedding
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         collection_name=QDRANT_COLLECTION
#     )

def create_embeddings_and_store_qdrant(docs):
    Qdrant.from_documents(
        documents=docs,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION
    )

# def qdrant_similarity_search(query, k=3):
#     client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#     vectorstore = Qdrant(
#         client=client,
#         collection_name=QDRANT_COLLECTION,
#         embedding=embedding_model  # ✅ USE embedding_function
#     )
#     return vectorstore.similarity_search(query, k=k)

def qdrant_similarity_search(query, k=3):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embedding_model
    )
    retrieved_chunks = vectorstore.similarity_search(query, k=k)

    sources = [doc.metadata.get("source") for doc in retrieved_chunks]
    most_common_source = Counter(sources).most_common(1)[0][0]
    filtered_chunks = [
        doc for doc in retrieved_chunks
        if doc.metadata.get("source") == most_common_source
    ]
    return filtered_chunks
