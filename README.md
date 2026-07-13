# Assistant TOMATE — RAG local + RAG agentique

Assistant technique intelligent développé avec Ollama, ChromaDB, LangChain et LangGraph.

Déploiement entièrement local via Docker — aucune donnée n'est envoyée vers des serveurs externes.

## Prérequis

- Docker Desktop
- Python 3.10 ou version ultérieure
- 8 Go de RAM minimum (16 Go recommandés)

## Configuration

### 1. Cloner le dépôt
git clone https://github.com/josaphatstar/Assistant-Intelligent-RAG
cd assistant-tomate

### 2. Démarrer les services Docker
docker-compose up -d

### 3. Télécharger Llama 3
docker exec -it tomate-ollama ollama pull llama3

### 4. Ajoutez vos documents PDF
Copiez vos fichiers PDF dans le répertoire data/documents/

### 5. Lancez l’ingestion des données
docker exec -it tomate-api python ingestion.py

## Architecture

- Ollama      → Serveur LLM local (Llama 3)
- ChromaDB    → Base de données vectorielle
- FastAPI     → Pipeline RAG + API REST (port 8000)
- Streamlit   → Interface utilisateur (port 8501)

## Points de terminaison de l'API

- POST /query-rag    → Phase 1 : RAG classique
- POST /query-agent  → Phase 2 : Multi-agent avec LangGraph

## Stack

Python · LangChain · LangGraph · ChromaDB · Ollama · FastAPI · Streamlit · Docker
