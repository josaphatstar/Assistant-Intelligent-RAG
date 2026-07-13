# src/agents/rag_agent.py

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from agents.state import AgentState
import os

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

def rag_agent(state: AgentState) -> AgentState:
    """Recherche dans la documentation TOMATE."""

    question = state["question"]
    print(f"[Agent RAG] Traitement : {question[:50]}...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vectorstore = Chroma(
        client=client,
        collection_name="documentation_tomate",
        embedding_function=embeddings
    )

    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        return {
            **state,
            "messages": list(state["messages"]) + [
                AIMessage(content="[RAG] Aucun document pertinent trouve.")
            ],
            "past_steps": state["past_steps"] + ["RAG: aucun resultat"],
            "niveau_support": 3
        }

    contexte = "\n\n".join([
        f"[Source: {d.metadata.get('module', '?')}]\n{d.page_content}"
        for d in docs
    ])

    llm = Ollama(base_url=OLLAMA_URL, model="llama3", temperature=0.2, num_predict=250)

    prompt = f"""Assistant technique TOMATE. Reponds uniquement avec le contexte.
Contexte : {contexte}
Question : {question}
Reponse concise :"""

    reponse = llm.invoke(prompt)

    return {
        **state,
        "messages": list(state["messages"]) + [
            AIMessage(content=f"[RAG] {reponse}")
        ],
        "past_steps": state["past_steps"] + ["RAG: recherche documentaire effectuee"],
        "niveau_support": 1
    }