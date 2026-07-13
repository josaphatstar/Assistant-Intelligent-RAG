# src/agents/orchestrator.py

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agents.state import AgentState
from agents.rag_agent import rag_agent
from agents.diagnostic_agent import diagnostic_agent
from agents.escalade_agent import escalade_agent
import re

# src/agents/orchestrator.py

def classifier(state: AgentState) -> str:
    """Décide quel agent appeler selon la question."""

    question = state["question"].lower()

    # Détection de codes d'erreur explicites
    if re.search(r'err-\w+-\d+', question):
        return "diagnostic"

    # Détection de problèmes techniques (vocabulaire élargi)
    mots_diagnostic = [
        "erreur", "error", "bug", "plante", "crash", "bloque",
        "ne fonctionne pas", "probleme", "souci", "dysfonctionnement",
        "ne demarre pas", "ne repond pas", "fige", "freeze"
    ]
    if any(mot in question for mot in mots_diagnostic):
        return "diagnostic"

    # Détection de demandes procédurales
    mots_rag = [
        "comment", "procedure", "etape", "parametrer",
        "configurer", "creer", "cloturer", "saisir", "ou trouver",
        "qu'est-ce que", "quel est"
    ]
    if any(mot in question for mot in mots_rag):
        return "rag"

    # Par défaut : RAG (recherche documentaire générale)
    return "rag"

def should_escalate(state: AgentState) -> str:
    """Décide si on escalade après le diagnostic."""

    if state.get("niveau_support", 1) >= 3:
        return "escalade"
    return END

def create_orchestrator():
    """Crée et compile le graphe d'agents."""

    workflow = StateGraph(AgentState)

    # Ajouter les noeuds
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("diagnostic_agent", diagnostic_agent)
    workflow.add_node("escalade_agent", escalade_agent)

    # Point d'entrée conditionnel
    workflow.set_conditional_entry_point(
        classifier,
        {
            "rag": "rag_agent",
            "diagnostic": "diagnostic_agent"
        }
    )

    # Après RAG → fin
    workflow.add_edge("rag_agent", END)

    # Après diagnostic → escalade si nécessaire
    workflow.add_conditional_edges(
        "diagnostic_agent",
        should_escalate,
        {
            "escalade": "escalade_agent",
            END: END
        }
    )

    workflow.add_edge("escalade_agent", END)

    return workflow.compile()