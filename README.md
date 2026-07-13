
# Assistant TOMATE — Local RAG + Agentic AI

Intelligent technical support assistant built with Ollama, ChromaDB, LangChain and LangGraph.
Fully local deployment via Docker — no data sent to external servers.

## Prerequisites

- Docker Desktop
- Python 3.10+
- 8 GB RAM minimum (16 GB recommended)

## Setup

### 1. Clone the repository

git clone <your-repo-url></your>
cd assistant-tomate

### 2. Start Docker services

docker-compose up -d

### 3. Download Llama 3

docker exec -it tomate-ollama ollama pull llama3

### 4. Add your PDF documents

Copy your PDF files into data/documents/

### 5. Run ingestion

docker exec -it tomate-api python ingestion.py

### 6. Test the assistant

python src/test_phase1_rag.py
python src/test_phase2_agents.py

## Architecture

- Ollama      → Local LLM server (Llama 3)
- ChromaDB    → Vector database
- FastAPI     → RAG pipeline + REST API (port 8000)
- Streamlit   → User interface (port 8501)

## API Endpoints

- POST /query-rag    → Phase 1: Classic RAG
- POST /query-agent  → Phase 2: Multi-agent with LangGraph

## Stack

Python · LangChain · LangGraph · ChromaDB · Ollama · FastAPI · Streamlit · Docker
