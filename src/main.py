# src/main.py

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agents.orchestrator import create_orchestrator
from agents.state import AgentState
from rag_chain import create_rag_chain

app = FastAPI(title="Assistant TOMATE", version="3.0")

print("Initialisation de la chaine RAG (Phase 1)...")
rag_chain = create_rag_chain()
print("Chaine RAG prete.")

print("Initialisation de l'orchestrateur multi-agents (Phase 2)...")
orchestrator = create_orchestrator()
print("Orchestrateur pret.")


class QueryRequest(BaseModel):
    question: str

class RagResponse(BaseModel):
    question: str
    answer: str

class AgentResponse(BaseModel):
    question: str
    answer: str
    agents_utilises: list
    niveau_support: int


@app.get("/")
def root():
    return {
        "status": "OK",
        "version": "3.0",
        "endpoints": {
            "/query-rag": "Phase 1 - RAG classique",
            "/query-agent": "Phase 2 - Architecture Agentic AI"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================
# PHASE 1 - RAG CLASSIQUE (pour la demo seance 3)
# ============================================
@app.post("/query-rag", response_model=RagResponse)
def query_rag(request: QueryRequest):
    try:
        answer = rag_chain.invoke(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PHASE 2 - MULTI-AGENTS (pour la demo seance 4)
# ============================================
@app.post("/query-agent", response_model=AgentResponse)
def query_agent(request: QueryRequest):
    try:
        initial_state = AgentState(
            messages=[HumanMessage(content=request.question)],
            question=request.question,
            plan=[],
            current_step=0,
            past_steps=[],
            response="",
            next_agent="",
            niveau_support=1
        )

        result = orchestrator.invoke(initial_state)

        messages_agents = [
            m.content for m in result["messages"]
            if hasattr(m, "content") and m.content.startswith("[")
        ]
        answer = messages_agents[-1] if messages_agents else "Aucune reponse generee."

        return {
            "question": request.question,
            "answer": answer,
            "agents_utilises": result["past_steps"],
            "niveau_support": result["niveau_support"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))