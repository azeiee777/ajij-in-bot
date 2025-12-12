# main.py
import os
import traceback
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain-community components (document loader, splitter, vectorstore)
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Use OpenAI Python client v1+ (modern)
from openai import OpenAI

# ---------- CONFIG ----------
WEBSITE_URL = "https://abdulajij.in"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

# Choose a model you have access to; replace if necessary
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # change to gpt-4.1-mini or gpt-3.5-turbo if you don't have access

app = FastAPI(title="Portfolio RAG API (v1 OpenAI client)")

origins = [
    "https://abdulajij.in",
    "http://abdulajij.in",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
vectorstore: FAISS | None = None

# Build vectorstore from website
def build_vectorstore_from_url(url: str) -> FAISS:
    print(f"[init] Loading page: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"[init] Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[init] Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(chunks, embeddings)
    print("[init] Vector store built successfully")
    return vs

# Robust retrieval helper (tries retriever API, else falls back to similarity_search)
def retrieve_context(vs: FAISS, question: str, k: int = TOP_K) -> str:
    try:
        retriever = vs.as_retriever(search_kwargs={"k": k})
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        elif hasattr(retriever, "get_relevant_items"):
            docs = retriever.get_relevant_items(question)
        else:
            docs = None
    except Exception:
        docs = None

    if not docs:
        try:
            docs = vs.similarity_search(question, k=k)
        except Exception:
            docs = []

    if not docs:
        return ""

    pieces = []
    for i, d in enumerate(docs, 1):
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        pieces.append(f"--- CHUNK {i} ---\n{text.strip()}\n")
    return "\n".join(pieces)

# Ask OpenAI (modern client)
def ask_openai_with_context(question: str, context: str, openai_api_key: str) -> str:
    # initialize client with explicit api key to be explicit (client also reads env var)
    client = OpenAI(api_key=openai_api_key)

    system_msg = (
        "You are a helpful assistant that answers questions ONLY using the provided context. "
        "If the answer is not clearly present in the context, you MUST reply exactly: "
        "\"I don't know based on the website content.\" Do NOT use any outside knowledge. Do NOT guess."
    )
    user_msg = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (use only the context above):"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        # new v1 client returns choices with message structure
        answer = resp.choices[0].message.content.strip()
        if "I don't know based on the website content." in answer:
            return "I don't know based on the website content."
        return answer
    except Exception as e:
        print("[openai error]", e)
        traceback.print_exc()
        # graceful user-facing message
        return "OpenAI API error — check server logs."

# FastAPI endpoints
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    global vectorstore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return AskResponse(answer="Server not configured: OPENAI_API_KEY missing.")

    if vectorstore is None:
        try:
            vectorstore = build_vectorstore_from_url(WEBSITE_URL)
        except Exception as e:
            print("[init error]", e)
            traceback.print_exc()
            return AskResponse(answer="Server error during initialization. Check logs.")

    context = retrieve_context(vectorstore, req.question, k=TOP_K)
    if not context.strip():
        return AskResponse(answer="I don't know based on the website content.")

    answer = ask_openai_with_context(req.question, context, api_key)
    return AskResponse(answer=answer)
