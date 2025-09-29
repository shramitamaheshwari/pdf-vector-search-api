# pdf-vector-search-api

## 🚀 Project Overview

The **`pdf-vector-search-api`** is a high-performance **FastAPI** application that implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions about PDF documents.

This API allows a client to submit a PDF URL and a list of questions. It processes the document by creating vector embeddings, uses a **FAISS** index for fast, semantic retrieval of context, and leverages Google **Gemini** (with an **OpenAI/Ollama** fallback) to generate precise, context-aware answers.

### Key Features

* **FastAPI** for robust and asynchronous API handling.
* **RAG Pipeline**: Semantic search using Sentence Transformers and FAISS for efficient context retrieval.
* **LLM Flexibility**: Primary generation uses **Gemini 1.5 Pro** (or Flash), with automatic fallback to **OpenAI** (GPT-3.5/GPT-4) or a local **Ollama** model.
* **Secure Endpoint**: API endpoint protected by a Bearer Token for authorization.
* **PDF Ingestion**: Handles external PDF links using `requests` and `PyPDF2` for text extraction.

## 🛠️ Setup and Installation

Follow these steps to get your development environment running locally.

### Prerequisites

* Python 3.10+ (The logs indicate Python 3.13, which is likely a pre-release version; ensure you're using a stable, supported Python version).
* Git

### 1. Clone the Repository

```bash
git clone [https://github.com/shramitamaheshwari/pdf-vector-search-api.git](https://github.com/shramitamaheshwari/pdf-vector-search-api.git)
cd pdf-vector-search-api
````

### 2\. Create and Activate Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment (PowerShell)
& ./.venv/Scripts/Activate.ps1
```

### 3\. Install Dependencies

Install the core packages, including `uvicorn` and the necessary RAG libraries.

```bash
# Install core dependencies (Faiss, LangChain, Sentence-Transformers, FastAPI, etc.)
pip install uvicorn[standard] fastapi pydantic requests PyPDF2 python-dotenv google-genai sentence-transformers langchain faiss-cpu numpy openai

# NOTE: The 'embedder' package was conflicting with a local file.
# To ensure your local code works, if you face an import error,
# run: `pip uninstall embedder`
```

### 4\. Configure Environment Variables

Create a file named **`.env`** in the root directory and populate it with your API keys and the required team token.

**.env file structure:**

```
# Gemini API Key for generation
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# OpenAI API Key for fallback
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Required Bearer Token for API authorization
TEAM_TOKEN="YOUR_SECRET_TEAM_TOKEN"
```

## ▶️ Running the Application

Start the FastAPI server using Uvicorn:

```bash
python -m uvicorn app.main:app --reload
```

The application will be accessible at: **`http://127.0.0.1:8000`**

## 💡 API Endpoint

The main endpoint for processing PDF documents is:

**POST `/hackrx/run`**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `documents` | string | The publicly accessible URL of the PDF document. |
| `questions` | `List[string]` | A list of questions to be answered based on the document. |

### Example Request Body

```json
{
  "documents": "[https://example.com/some-policy-document.pdf](https://example.com/some-policy-document.pdf)",
  "questions": [
    "What is the effective date of this policy?",
    "Who is responsible for annual reporting?"
  ]
}
```

### Authorization

All requests to this endpoint must include an **`Authorization`** header:

`Authorization: Bearer YOUR_SECRET_TEAM_TOKEN`

## 🧩 Project Structure

```
pdf-vector-search-api/
├── .venv/                   # Python Virtual Environment
├── app/                     # The main application module
│   ├── __init__.py          # Makes 'app' a Python package
│   ├── main.py              # FastAPI app and main RAG orchestration logic
│   ├── embedder.py          # Local module containing chunking, embedding, and FAISS logic
│   ├── query_llm.py         # LLM fallback module (OpenAI & Ollama implementation)
│   └── parser.py            # PDF text extraction utilities (currently unused in main.py)
├── .env                     # Environment variables (API keys, secrets)
└── README.md
```

## 📜 Technical Notes

### LLM Prioritization

The application attempts to use models in this order:

1.  **Gemini (`gemini-1.5-pro`)**: Primary model as configured in `app/main.py`.
2.  **OpenAI (GPT-3.5-turbo)**: Fallback in `app/query_llm.py`.
3.  **Ollama (`tinyllama`)**: Secondary fallback executed via subprocess in `app/query_llm.py`.

### Local Module Imports

Note that due to a namespace conflict, the local modules (`embedder.py`, `query_llm.py`) are imported within `app/main.py` using direct imports (e.g., `from embedder import ...`). For cleaner code in a package structure, these could be changed to **relative imports** (e.g., `from .embedder import ...`).

```
```
