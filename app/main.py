# Create Virtual Environment 
    #python -m venv venv
# Activate virtual Environment
    # .\venv\Scripts\activate
# install all the required lib either directly or add in requirements.txt file
# pip install -r requirements.txt
# Start the server
    #uvicorn app.main:app --reload

# Debugging FastAPI code
# From run and debug options from Run button towards top-right, choose option "Python Debugger: Debug suing launch.json"
# choose Python Debugger - Current file from seach drop down which comes in focus 
# Choose Python File
# Update launch.json file with below this content
# {
#   "version": "0.2.0",
#   "configurations": [
#     {
#       "name": "Debug FastAPI (Uvicorn)",
#       "type": "python",
#       "request": "launch",
#       "module": "uvicorn",
#       "args": [
#         "app.main:app",          // Change 'main' to your filename (no .py)
#         "--reload"
#       ],
#       "jinja": true,
#       "justMyCode": true
#     }
#   ]
# }


from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.docstore.document import Document
# Qdrant utilities moved to separate file
from app.qdrant_utils import create_embeddings_and_store_qdrant, qdrant_similarity_search
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from youtube_transcript_api import YouTubeTranscriptApi
import re
from collections import Counter 
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY2"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    k: int = 3  # number of top results to return

# ---- Helper Functions ---- #

def validate_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed."
        )


async def extract_text_from_pdf(file: UploadFile) -> str:
    file_bytes = await file.read()
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        pdf_doc.close()

        if not text.strip():
            raise ValueError("PDF has no extractable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {str(e)}")


def split_text_into_chunks(text: str, pdf_path: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc = Document(page_content=text, metadata={"source": pdf_path.split("/")[-1]})
    return splitter.split_documents([doc])


# def create_embeddings_and_store_qdrant(docs, model_name="all-MiniLM-L6-v2", collection_name="pdfs_embeddings_sk"):
#     try:
#         embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#         Qdrant.from_documents(
#             documents=docs,
#             embedding=embedding_model,
#             url="https://9ecd8ba1-06dc-4e59-9dfa-a21dba98f9a4.eu-west-1-0.aws.cloud.qdrant.io",
#             api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9FxGRzqAjpxaQxrISQMpofL6zQb80Hbzs4ToiKFFVRU",
#             collection_name=collection_name
#         )
#         return len(docs)
#     except Exception as e:
#         raise RuntimeError(f"Qdrant storage failed: {str(e)}")

# def create_embeddings_and_store(docs, model_name="all-MiniLM-L6-v2", index_dir="faiss_index"):
#     try:
#         embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#         # qdrant_client = QdrantClient(
#         #     host=host,
#         #     port=port
#         # )
#         # collection_name = "my_embeddings"
#         # qdrant_client.recreate_collection(
#         #     collection_name=collection_name,
#         #     vectors_config=VectorParams(size=384, distance=Distance.COSINE)
#         # )
#         # Qdrant.from_documents(
#         #     docs,
#         #     embedding_model,
#         #     client=qdrant_client,
#         #     collection_name=collection_name
#         # )
#         vector_db = FAISS.from_documents(docs, embedding_model)
#         vector_db.save_local(index_dir)
#         return len(docs)
#     except Exception as e:
#         raise RuntimeError(f"Qdrant storage failed: {str(e)}")
    
from collections import Counter
from typing import List, Optional

# def most_relevant_pdf(results: List, metadata_key: str = "pdf_name") -> Optional[str]:
#     pdf_counter = Counter()

#     for r in results:
#         metadata = getattr(r, "metadata", None)
#         if metadata and metadata_key in metadata:
#             pdf_counter[metadata[metadata_key]] += 1

#     if not pdf_counter:
#         return None

#     # Get the most common PDF name
#     return pdf_counter.most_common(1)[0][0]

def ask_gemini(query: str, context_chunks: list) -> str:
    load_dotenv()

    # Configure the API
    key = os.getenv("GOOGLE_API_KEY2")
    print("key:", key)

    genai.configure(api_key=key)

    context = "\n\n".join([doc.page_content for doc in context_chunks])

    prompt = f"""You are an assistant that answers questions based on the context provided.

Context:
{context}

Question:
{query}

Answer:"""
    print(prompt)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def get_youtube_transcript(video_url: str) -> str:
    try:
        video_id = extract_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text
    except Exception as e:
        raise RuntimeError(f"Transcript retrieval failed: {str(e)}")

def extract_video_id(url: str) -> str:
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url).query
    params = parse_qs(query)
    if "v" in params:
        return params["v"][0]
    else:
        # Handle youtu.be/VIDEO_ID format
        return url.split("/")[-1]


# ---- API Endpoint ---- #

@app.get("/CallLLM")
def read_root():
    # Load environment variables from .env
    load_dotenv()

    # Configure the API
    key = os.getenv("GOOGLE_API_KEY2")
    print("key:", key)

    genai.configure(api_key=key)

    # for m in genai.list_models():
    #     print(m.name, "->", m.supported_generation_methods)

    # Create a generative model (use "gemini-pro" for text)
    # model = genai.GenerativeModel(model_name="models/gemini-pro")
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define a prompt
    prompt = "Give a short note on Dynamic Programming"

    # Generate content
    response = model.generate_content(prompt)

    # Print the result
    print(response.text)

    return {"message": response.text}

@app.get("/")
def read_root():
    load_dotenv()   
    return {"message": os.getenv("SHREYA_NAME")}

@app.post("/upload-pdf/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    all_chunks = []
    upload_summaries = []

    try:
        for file in files:  
            validate_pdf(file)

            pdf_filename = file.filename
            text = await extract_text_from_pdf(file) 
            docs = split_text_into_chunks(text, pdf_path=pdf_filename)
            all_chunks.extend(docs)

            upload_summaries.append({
                "filename": pdf_filename,
                "num_chunks": len(docs)
            })

        # for i, doc in enumerate(docs):
        #     print(f"\n--- Chunk {i + 1} ---\n{doc.page_content}")

        # num_chunks = create_embeddings_and_store(docs)
        num_chunks = create_embeddings_and_store_qdrant(docs)

        return JSONResponse(content={
            "message": f"Embeddings stored in Qdrant",
            "summary": upload_summaries
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# @app.post("/query/")
# async def query_vector_db(request: QueryRequest):
#     try:
#         # Load embedding model and FAISS index

#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         # db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

#         # db = FAISS.load_local("faiss_index", embedding_model)
#         # # client = QdrantClient(host="localhost", port=6333)
#         # print(client.get_collections()) 
#         # qdrant = Qdrant(
#         #     client = QdrantClient(host="localhost", port=6333),
#         #     collection_name = "my_embeddings",
#         #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         # )
    
#         # Perform similarity search
    
#         #results = db.similarity_search(request.query, k=request.k)
#         # results = qdrant.similarity_search("your query here", k=3)
#         for r in results:
#             print(r.page_content)

#         gemini_answer = ask_gemini(request.query, results)

#         return {
#             "query": request.query,
#             "results": [r.page_content for r in results],
#             "answer": gemini_answer
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_qdrant/")
async def query_vector_db_qdrant(request: QueryRequest):
    try:
        # Use Qdrant utility for similarity search
        results = qdrant_similarity_search(request.query, k=request.k)
        gemini_answer = ask_gemini(request.query, results)
        return {
            "answer": gemini_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    #"matched_pdf": most_relevant_pdf,

class VideoRequest(BaseModel):
    video_url: str

@app.post("/upload-youtube/")
async def upload_youtube(req: VideoRequest):
    try:
        transcript = get_youtube_transcript(req.video_url)
        video_id = extract_video_id(req.video_url)
        source_name = f"video_{video_id}.txt"
        chunks = split_text_into_chunks(transcript, pdf_path=source_name)

        create_embeddings_and_store_qdrant(chunks)

        return {"message": "Transcript embedded successfully", "source": source_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


