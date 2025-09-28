# ==========================================
# main.py  —  FastAPI PDF-QnA with FAISS + Gemini/OpenAI
# ==========================================

import os
import io
import json
import time
import requests
from typing import List

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# ---- Local modules
from embedder import create_chunks, embed_chunks, create_faiss_index, search_faiss  
from query_llm import ask_llm        # fallback to OpenAI/Ollama if Gemini not desired
import google.generativeai as genai

# -------------------------------------------------------
# 1.  Config & Initialization
# -------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")
genai.configure(api_key=GEMINI_API_KEY)

TEAM_TOKEN = os.getenv("TEAM_TOKEN")           # store token in .env
if not TEAM_TOKEN:
    raise ValueError("❌ TEAM_TOKEN not found in .env")

app = FastAPI(title="HackRx PDF-QnA API")
security = HTTPBearer()

# -------------------------------------------------------
# 2.  Auth
# -------------------------------------------------------
def check_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials.credentials

# -------------------------------------------------------
# 3.  Pydantic Models
# -------------------------------------------------------
class ApiRequest(BaseModel):
    documents: str                 # PDF URL
    questions: List[str]           # list of questions

class ApiResponse(BaseModel):
    answers: List[str]

# -------------------------------------------------------
# 4.  PDF Text Extraction
# -------------------------------------------------------
def extract_pdf_text(url: str) -> str:
    """Downloads PDF and returns concatenated text of all pages."""
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("No extractable text found in PDF.")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {e}")

# -------------------------------------------------------
# 5.  LLM Helper (Gemini) — optional
# -------------------------------------------------------
def ask_with_gemini(question: str, context: str) -> str:
    """Ask Gemini using only the retrieved context."""
    model = genai.GenerativeModel("gemini-1.5-pro")   # or gemini-1.5-flash for speed

    prompt = f"""
You are a helpful assistant that answers questions **only** from the context below.

Context:
---
{context}
---

Question: {question}

Give one short, direct sentence as the answer.
If answer is not found in context, reply: "Not mentioned in the document."
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from Gemini: {e}"

# -------------------------------------------------------
# 6.  Main Endpoint
# -------------------------------------------------------
@app.post("/hackrx/run", response_model=ApiResponse)
async def process_request(req: ApiRequest, token: str = Depends(check_token)):

    # ---- Step-1: Extract text from PDF
    pdf_text = extract_pdf_text(req.documents)

    # ---- Step-2: Chunk + Embed + Build FAISS
    chunks = create_chunks(pdf_text)
    vectors = embed_chunks(chunks)
    index = create_faiss_index(vectors)

    # ---- Step-3: Answer each question
    answers: List[str] = []
    for q in req.questions:
        # Retrieve top-k most relevant chunks
        top_chunks = search_faiss(index, q, chunks, top_k=5)
        context = "\n".join(top_chunks)

        # Prefer Gemini; if you want OpenAI/Ollama, use ask_llm(q, context)
        ans = ask_with_gemini(q, context)

        # fallback to query_llm if Gemini failed
        if ans.lower().startswith("error"):
            ans = ask_llm(q, context)

        answers.append(ans)

    return ApiResponse(answers=answers)

