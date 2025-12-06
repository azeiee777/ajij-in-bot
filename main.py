import os

from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env file in the project root

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ---------- CONFIG ----------
WEBSITE_URL = "https://abdulajij.in"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

app = FastAPI()

# Allow your own domain to call this API
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

vectorstore = None
qa_chain = None


def build_vectorstore_from_url(url: str) -> FAISS:
    """
    Load the website content and build a FAISS vector store.
    """
    print(f"Loading page: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(chunks, embeddings)
    print("Vector store built successfully")
    return vs


def build_qa_chain(vs: FAISS) -> RetrievalQA:
    """
    Create a RetrievalQA chain that:
    - Retrieves top-k chunks
    - Answers ONLY from those chunks
    - Says 'I don't know based on the website content.' when info is missing
    """
    llm = ChatOpenAI(
        model="gpt-4.1-mini",  # or gpt-4.1 / gpt-4o-mini etc.
        temperature=0,         # low creativity -> fewer hallucinations
    )

    template = """
You are a helpful assistant that answers questions ONLY using the provided context.

If the answer is not clearly present in the context, you MUST reply exactly:
"I don't know based on the website content."

Do NOT use any outside knowledge. Do NOT guess.

Context:
{context}

Question:
{question}

Answer (remember: use only the context above):
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return qa


@app.on_event("startup")
def startup_event():
    """
    Build vector store + QA chain when the server starts.
    """
    global vectorstore, qa_chain

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Check your .env file.")

    vectorstore = build_vectorstore_from_url(WEBSITE_URL)
    qa_chain = build_qa_chain(vectorstore)
    print("Startup complete: RAG over abdulajij.in is ready.")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Accepts a question and returns an answer based only on abdulajij.in content.
    """
    if qa_chain is None:
        return AskResponse(answer="Backend not ready yet, please try again.")

    result = qa_chain({"query": req.question})
    answer = result["result"].strip()

    # Extra safety: normalize if model ignored the instruction
    if "I don't know based on the website content." in answer:
        answer = "I don't know based on the website content."

    return AskResponse(answer=answer)
